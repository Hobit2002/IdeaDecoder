import os
import time
import torch
from client_whisper_transcribe import transcribe_audio
from whisper_client_tools import load_wave, compute_mel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAV_DIR = "tmp_wav" #"sessions/tmp_wav_session5"   # ← set thi

# Load speech detection models
from torch.nn.functional import sigmoid
from speech_detection.cp_resnet import get_model
speech_detection_model = get_model(n_classes=1, in_channels=1)
state_dict = torch.load("speech_detection_model_2.pth", map_location=DEVICE)
speech_detection_model.load_state_dict(state_dict)
speech_detection_model.eval()

# Keep track of already processed files
processed_files = set()

print("Listening for new audio files...")

for fidx in range(50,90):

    filepath = os.path.join(WAV_DIR, f"{fidx}.wav")
    if not os.path.exists(filepath): continue

    if filepath in processed_files:
        continue

    print(f"Processing: {fidx}")

    # Load audio
    audio = load_wave(filepath)

    mel_spec = compute_mel(audio.detach().numpy())
    prediction = sigmoid(speech_detection_model(mel_spec))[0].item() #speaker_recognition_model.predict_proba(mel_spec.reshape(1, -1))[0][1]
    print(f"[#{fidx}] Jasmi's speech probability:", prediction)
    
    transcription_starts = time.time()
    text = transcribe_audio([(fidx, audio)])
    print("Transcription took:",time.time() - transcription_starts,"seconds")
    print(text)

