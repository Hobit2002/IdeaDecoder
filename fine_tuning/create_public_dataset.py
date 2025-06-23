import os
import csv
import re
import shutil
from tqdm import tqdm
from pathlib import Path
from pydub import AudioSegment

def extract_digits(filename):
    return "".join(re.findall(r"\d+", Path(filename).stem))

AUDIO_PATH = "data/audio"
ANNOTATION_PATH = "data/annotations"
is_recording_day =  lambda name: any(int(num) in range(78, 90) for num in ''.join(c if c.isdigit() else ' ' for c in name).split())
is_real_conversation = lambda name: is_recording_day(name) and ("88" in name or "89" in name)
is_song = lambda name: is_recording_day(name) and extract_digits(name) in ["78","79","82","84","85","86","87"]
is_recording_day_other = lambda name: is_recording_day(name) and not is_real_conversation(name) and not is_ai_conversation(name)
is_ai_conversation = lambda name: any(int(num) > 21 for num in ''.join(c if c.isdigit() else ' ' for c in name).split()) and not is_recording_day(name)
is_reading = lambda name: any(int(num) <= 21 for num in ''.join(c if c.isdigit() else ' ' for c in name.replace(".mp3","")).split())

session_types = {"R":is_reading,"A":is_ai_conversation,"SC":is_real_conversation,"SS":is_song,"SO":is_recording_day_other}

applicability_condition = lambda name: "Session" in name and "checked" not in name and "clean" not in name

def create_audio_dataset(target_csv, target_audio, target_sessions, audio_path, annotation_path, applicability_condition, session_types):
    os.makedirs(target_audio, exist_ok=True)
    os.makedirs(target_sessions, exist_ok=True)
    
    with open(target_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sample_audio", "transcription", "session_type", "session", "start", "end","split"])
        
        for audio_file in tqdm(Path(audio_path).glob("*.wav")):
            if not applicability_condition(audio_file.name):
                print(f"Skipping {audio_file.name}")
                continue
            
            file_id = extract_digits(audio_file.name)
            if int(file_id) == 35 and "_a_" in audio_file.name: file_id = f"_a_{file_id}"
            elif int(file_id) == 35 and "_b_" in audio_file.name: file_id = f"_b_{file_id}" 
            annotation_file = Path(annotation_path) / f"Session{file_id}.csv"
            
            # Determine session type
            session_type = next((key for key, check in session_types.items() if check(audio_file.name)), None)
            
            if not annotation_file.exists():
                print(f"Warning: Annotation file {annotation_file} not found. Skipping {audio_file.name}.")
                continue
            
            try:
                audio = AudioSegment.from_mp3(audio_file)
            except Exception as e:
                print(f"Error loading {audio_file.name}: {e}")
                continue
            
            with open(annotation_file, mode='r', encoding='utf-8') as annfile:
                reader = csv.reader(annfile)
                for row_id, row in enumerate(reader):
                    if len(row) != 3:
                        print(f"Skipping malformed row in {annotation_file}: {row}")
                        continue
                    
                    transcription, start, end = row
                    
                    # Filter transcriptions containing personal identifiers
                    surnames = ["Gembec","Kling","Přibyl","Gabzdyl","Majer","Zvoník","Šamár","Václavík","Švanda","Adam","Kuřák","Herzig","Škorpík","Horál","Kurtyn","Pacner","Straka","Druckmiller","Jánsk","Houšk"]
                    if any([surname in transcription for surname in surnames]): continue

                    start, end = float(start), float(end)
                    try:
                        start, end = float(start), float(end)
                        segment = audio[start:end]
                    except ValueError:
                        print(f"Skipping row with invalid timestamps: {row}")
                        continue
                    
                    curated_file_id = file_id.replace("_a_35","35a").replace("_b_35","35b")
                    segment_filename = f"{curated_file_id}_{row_id}.mp3"
                    segment_path = Path(target_audio) / segment_filename
                    segment.export(segment_path, format="mp3")
                    writer.writerow([segment_filename, transcription, session_type, curated_file_id, start, end, "train"])
                    # Due to specifics of the repository used for fine-tuning, we duplicate the training set also as the validation one
                    writer.writerow([segment_filename, transcription, session_type, curated_file_id, start, end, "test"])
    
            # Export the whole session to mp3
            session_path = Path(target_sessions) / (curated_file_id + ".mp3")
            audio.export(session_path, format="mp3")

    print("Dataset creation completed.")

if __name__ == "__main__":
    create_audio_dataset("data/public_dataset/metadata.csv","data/public_dataset/samples","data/public_dataset/sessions",AUDIO_PATH,ANNOTATION_PATH,applicability_condition,session_types)