from pydub import AudioSegment
import os, csv
from datetime import datetime

ACCEPTED_TRANSCRIPTIONS = []
REWRITTEN = {}
VISITED_COUNTERS = [1e8]

extract_counter = lambda line: int(line.split("[")[2].split("]")[0].replace("#",""))

def combine_wav_files(directory, session_mark, output_path):
    # Create an empty AudioSegment object to hold the combined audio
    combined_audio = AudioSegment.empty()

    # Get all mp3 files in the directory, sorted by name
    mp3_files = sorted([file for file in os.listdir(directory) if file.endswith(".wav") and int("".join([ch for ch in file if ch.isnumeric()])) < max(VISITED_COUNTERS)], key=lambda line: int("".join([ch for ch in line if ch.isnumeric()])))
    # Load and combine the mp3 files with silence in between
    for m,mp3_file in enumerate(mp3_files):
        mp3_path = os.path.join(directory, mp3_file)
        audio_segment = AudioSegment.from_wav(mp3_path)
        combined_audio += audio_segment
    # Export the combined audio to the output path
    combined_audio.export(f"{output_path}/Session{session_mark}.wav", format="wav")


# 1. Extract feedbacks from the log
csv_rows = []
try:
    with open('whisper_client.log', 'r', encoding='utf-8') as f:
        log_lines = f.readlines()[::-1]
        for line in log_lines:
            line = line.strip() 
            if line.endswith(":accept"):
                counter = int(line.split()[-1].replace(":accept",""))
                ACCEPTED_TRANSCRIPTIONS.append(counter)
            elif ":rewrite:" in line:
                counter = int(line.split()[-1].split(":rewrite:")[0])
                ACCEPTED_TRANSCRIPTIONS.append(counter)
                REWRITTEN[counter] = line.split()[-1].split(":rewrite:")[-1]
            elif "] [#" in line:
                counter = extract_counter(line)
                if counter in VISITED_COUNTERS or counter > max(VISITED_COUNTERS): break
                elif "Transcription: " in line:
                    if counter in ACCEPTED_TRANSCRIPTIONS:
                        if counter in REWRITTEN.keys(): transcription = REWRITTEN[counter]
                        else: transcription = line.split("Transcription: ")[-1]
                        start_time, end_time = counter * 3000, (counter + 1) * 3000
                        csv_rows.append((transcription, start_time, end_time))
                else: VISITED_COUNTERS.append(counter)
except FileNotFoundError:
    print(f"Error: Log file not found. Please create it.")

# Save CSV
if csv_rows:
    session_mark = datetime.now().strftime("%d_%m_%Y_%H_%M")
    with open(f"data/annotations/Session{session_mark}.csv".replace("_",""), mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(csv_rows)

# And also the audio
    del VISITED_COUNTERS[0]
    combine_wav_files("tmp_wav", session_mark, "data/audio")