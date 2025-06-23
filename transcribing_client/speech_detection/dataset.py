import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from torch.distributions.beta import Beta

def mixstyle(x, p=0.4, alpha=0.4, eps=1e-6):
    if np.random.rand() > p:
        return x
    batch_size = x.size(0)

    # changed from dim=[2,3] to dim=[1,3] - from channel-wise statistics to frequency-wise statistics
    f_mu = x.mean(dim=[1, 3], keepdim=True)
    f_var = x.var(dim=[1, 3], keepdim=True)

    f_sig = (f_var + eps).sqrt()  # compute instance standard deviation
    f_mu, f_sig = f_mu.detach(), f_sig.detach()  # block gradients
    x_normed = (x - f_mu) / f_sig  # normalize input
    lmda = Beta(alpha, alpha).sample((batch_size, 1, 1, 1)).to(x.device)  # sample instance-wise convex weights
    perm = torch.randperm(batch_size).to(x.device)  # generate shuffling indices
    f_mu_perm, f_sig_perm = f_mu[perm], f_sig[perm]  # shuffling
    mu_mix = f_mu * lmda + f_mu_perm * (1 - lmda)  # generate mixed mean
    sig_mix = f_sig * lmda + f_sig_perm * (1 - lmda)  # generate mixed standard deviation
    x = x_normed * sig_mix + mu_mix  # denormalize input using the mixed statistics
    return x

class AudioSegmentDataset(Dataset):
    def __init__(self, speech_dir, non_speech_dir, sample_rate=16000, n_mels=64, segment_length=3, augment=True):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.segment_samples = int(segment_length * sample_rate)
        self.augment = augment

        # Load file paths and labels
        self.file_paths = []
        self.labels = []

        for path in sorted(os.listdir(speech_dir)):
            if path.endswith('.npy'):
                self.file_paths.append(os.path.join(speech_dir, path))
                self.labels.append(1)

        for path in sorted(os.listdir(non_speech_dir)):
            if path.endswith('.npy'):
                self.file_paths.append(os.path.join(non_speech_dir, path))
                self.labels.append(0)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        audio = np.load(path).astype(np.float32)

        if self.augment:
            # Add random constant bias
            """bias = np.random.uniform(-0.01, 0.01)
            audio = audio + bias"""

            # Add Gaussian noise
            noise = np.random.normal(0, 0.005, size=audio.shape)
            audio = audio + noise

            # Optional gain
            gain = np.random.uniform(0.8, 1.2)
            audio = audio * gain

        # Ensure length
        if len(audio) != self.segment_samples:
            if len(audio) > self.segment_samples:
                audio = audio[:self.segment_samples]
            else:
                pad = self.segment_samples - len(audio)
                audio = np.pad(audio, (0, pad), mode='constant')

        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=256,
            n_fft=1024,
            power=2.0
        )

        # Convert to log scale (dB)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize (per sample)
        mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

        # Convert to tensor
        mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)  # shape: [n_mels, time]

        return mel_tensor, torch.tensor(label, dtype=torch.long)
