from pydub import AudioSegment
import os

def combine_wav_files(directory, f, output_path):
    # Create an empty AudioSegment object to hold the combined audio
    combined_audio = AudioSegment.empty()

    # Get all mp3 files in the directory, sorted by name
    mp3_files = sorted([file for file in os.listdir(directory) if file.endswith(".wav")], key=lambda line: int("".join([ch for ch in line if ch.isnumeric()])))

    # Load and combine the mp3 files with silence in between
    for m,mp3_file in enumerate(mp3_files):
        mp3_path = os.path.join(directory, mp3_file)
        audio_segment = AudioSegment.from_wav(mp3_path)
        combined_audio += audio_segment
    # Export the combined audio to the output path
    combined_audio.export(f"{output_path}/merged.wav", format="wav")

input_path = os.path.join(os.getcwd(),"sessions/tmp_wav_session3")
combine_wav_files(input_path, 1.5, "sessions")