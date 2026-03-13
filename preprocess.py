import os
import random
import numpy as np
import librosa

import warnings
warnings.filterwarnings('ignore')

SR = 22050
DURATION = 30
MAX_LEN = SR * DURATION
N_MELS = 128


def load_audio(path, sr=SR):
    """Load an audio file and return as a numpy array. Returns silence on failure."""
    try:
        audio, _ = librosa.load(path, sr=sr, mono=True)
        return audio
    except:
        return np.zeros(MAX_LEN)


def pad_or_trim(audio, max_len=MAX_LEN):
    """Pad audio with zeros if too short, or trim if too long."""
    if len(audio) > max_len:
        return audio[:max_len]
    return np.pad(audio, (0, max_len - len(audio)))


def create_stem_mashup(genre, genre_path):
    """
    Create a mixed audio sample from individual stems (drums, bass, vocals, other).
    Randomly selects songs and applies time stretch and volume augmentation.
    """
    genre_dir = os.path.join(genre_path, genre)
    songs = [s for s in os.listdir(genre_dir)
             if os.path.isdir(os.path.join(genre_dir, s))]
    if not songs:
        return np.zeros(MAX_LEN)

    stems = []
    for stem_name in ["drums.wav", "bass.wav", "vocals.wav", "other.wav"]:
        song = random.choice(songs)
        audio = load_audio(os.path.join(genre_dir, song, stem_name))

        # randomly apply time stretching for augmentation
        if random.random() < 0.3:
            rate = np.random.uniform(0.95, 1.05)
            audio = librosa.effects.time_stretch(audio, rate=rate)

        audio = pad_or_trim(audio)

        # random volume scaling
        audio *= np.random.uniform(0.3, 1.5)

        # randomly mute a stem to simulate missing instruments
        if random.random() < 0.15:
            audio *= 0

        stems.append(audio)

    mix = np.sum(stems, axis=0)
    return librosa.util.normalize(mix)


def add_noise(audio, noise_path):
    """
    Add ESC-50 environmental noise to audio at a random SNR between 5-20 dB.
    Applied with 80% probability.
    """
    if random.random() > 0.8:
        return audio

    noise_files = [f for f in os.listdir(noise_path) if f.endswith('.wav')]
    noise = load_audio(os.path.join(noise_path, random.choice(noise_files)))

    if len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))

    start = random.randint(0, max(1, len(noise) - len(audio)))
    noise = pad_or_trim(noise[start:start + len(audio)], len(audio))

    snr_linear = 10 ** (random.uniform(5, 20) / 20)
    audio_norm = audio / (np.linalg.norm(audio) + 1e-8)
    noise_norm = noise / (np.linalg.norm(noise) + 1e-8)
    return librosa.util.normalize(audio_norm + noise_norm / snr_linear)


def audio_to_mel(audio, sr=SR, n_mels=N_MELS):
    """Convert raw audio to a log-scaled mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels,
        n_fft=2048, hop_length=512
    )
    return librosa.power_to_db(mel, ref=np.max)


def spec_augment(mel, freq_mask=30, time_mask=60):
    """
    Apply SpecAugment to a mel spectrogram.
    Masks random frequency and time bands with the mean value.
    Helps prevent overfitting during training.
    """
    mel = mel.copy()
    n_mels, n_frames = mel.shape
    mean_val = mel.mean()

    # apply 2 frequency masks
    for _ in range(2):
        f = np.random.randint(0, freq_mask)
        f0 = np.random.randint(0, max(1, n_mels - f))
        mel[f0:f0+f, :] = mean_val

    # apply 2 time masks
    for _ in range(2):
        t = np.random.randint(0, time_mask)
        t0 = np.random.randint(0, max(1, n_frames - t))
        mel[:, t0:t0+t] = mean_val

    return mel
