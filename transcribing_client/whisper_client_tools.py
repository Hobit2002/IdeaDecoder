from datetime import datetime
import torch
import torchaudio.transforms as at
import torchaudio
import librosa
import numpy as np

# Logging function
def log_message(*messages):
    message = " ".join([str(msg) for msg in messages])
    timestamp = datetime.now().strftime("[%d:%m:%Y:%H:%M:%S]")
    full_message = f"{timestamp} {message}"
    print(full_message)
    with open("whisper_client.log", "a", encoding="utf-8") as f:
        f.write(full_message + "\n")

# Utility: Load and process wave
def load_wave(wave_path, sample_rate: int = 16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    if waveform.ndim == 2:
        waveform = torch.mean(waveform, dim=0)
    return waveform

# Helper function to compute mel spectrogram
def compute_mel(audio, sr=16_000, n_mels=64):
    mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            hop_length=256,
            n_fft=1024,
            power=2.0
        )

    # Convert to log scale (dB)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize (per sample)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    # Convert to tensor
    mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mel_tensor