import os
import random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from sklearn.metrics import f1_score, accuracy_score

import warnings
warnings.filterwarnings('ignore')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


def find_competition_path():
    possible_paths = [
        "/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup",
        "/kaggle/input/jan-2026-dl-gen-ai-project",
        "/kaggle/input/messy-mashup",
        "/kaggle/input/messymashup",
        "/kaggle/input/messy-mashup-t12026",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            required = ["genres_stems", "mashups", "test.csv"]
            has_all = all(os.path.exists(os.path.join(path, req)) for req in required)
            if has_all:
                print(f"Found competition data at: {path}")
                return path
    print("Competition data not found!")
    if os.path.exists("/kaggle/input"):
        print("Available datasets:")
        for ds in os.listdir("/kaggle/input"):
            print(f"  - {ds}")
    raise FileNotFoundError("Competition data not found.")

ROOT = find_competition_path()

class Config:
    ROOT = ROOT
    GENRE_PATH = f"{ROOT}/genres_stems"
    NOISE_PATH = f"{ROOT}/ESC-50-master/audio"
    TEST_AUDIO_PATH = f"{ROOT}/mashups"
    TEST_CSV = f"{ROOT}/test.csv"

    SR = 22050
    DURATION = 30
    MAX_LEN = SR * DURATION
    N_MELS = 128
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 3e-4
    TRAIN_SAMPLES = 5000
    VAL_SAMPLES = 1000
    PATIENCE = 10

    WANDB_PROJECT = "23f3001069-t12026"
    WANDB_ENTITY  = "23f3001069-23f3001069-t12026"

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

paths_to_check = {
    "Genre data": config.GENRE_PATH,
    "Noise data": config.NOISE_PATH,
    "Test audio": config.TEST_AUDIO_PATH,
    "Test CSV":   config.TEST_CSV
}
all_good = True
for name, path in paths_to_check.items():
    if os.path.exists(path):
        count = len(os.listdir(path)) if os.path.isdir(path) else ""
        print(f"  {name}: {path} {f'({count} items)' if count else ''}")
    else:
        print(f"  MISSING - {name}: {path}")
        all_good = False

if not all_good:
    raise FileNotFoundError("Required data paths not found")

GENRES = sorted([d for d in os.listdir(config.GENRE_PATH)
                 if os.path.isdir(os.path.join(config.GENRE_PATH, d))])
GENRE2IDX = {g: i for i, g in enumerate(GENRES)}
IDX2GENRE  = {i: g for g, i in GENRE2IDX.items()}

print(f"Device: {config.DEVICE}")
print(f"Genres ({len(GENRES)}): {GENRES}")


def load_audio(path, sr=config.SR):
    try:
        audio, _ = librosa.load(path, sr=sr, mono=True)
        return audio
    except:
        return np.zeros(config.MAX_LEN)

def pad_or_trim(audio, max_len=config.MAX_LEN):
    if len(audio) > max_len:
        return audio[:max_len]
    return np.pad(audio, (0, max_len - len(audio)))

def create_stem_mashup(genre):
    genre_dir = os.path.join(config.GENRE_PATH, genre)
    songs = [s for s in os.listdir(genre_dir)
             if os.path.isdir(os.path.join(genre_dir, s))]
    if not songs:
        return np.zeros(config.MAX_LEN)

    stems = []
    for stem_name in ["drums.wav", "bass.wav", "vocals.wav", "other.wav"]:
        song = random.choice(songs)
        audio = load_audio(os.path.join(genre_dir, song, stem_name))
        if random.random() < 0.3:
            rate = np.random.uniform(0.95, 1.05)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        audio = pad_or_trim(audio)
        audio *= np.random.uniform(0.3, 1.5)
        if random.random() < 0.15:
            audio *= 0
        stems.append(audio)

    mix = np.sum(stems, axis=0)
    return librosa.util.normalize(mix)

def add_noise(audio):
    if random.random() > 0.8:
        return audio
    noise_files = [f for f in os.listdir(config.NOISE_PATH) if f.endswith('.wav')]
    noise = load_audio(os.path.join(config.NOISE_PATH, random.choice(noise_files)))
    if len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))
    start = random.randint(0, max(1, len(noise) - len(audio)))
    noise = pad_or_trim(noise[start:start + len(audio)], len(audio))
    snr_linear = 10 ** (random.uniform(5, 20) / 20)
    audio_norm = audio / (np.linalg.norm(audio) + 1e-8)
    noise_norm = noise / (np.linalg.norm(noise) + 1e-8)
    return librosa.util.normalize(audio_norm + noise_norm / snr_linear)

def audio_to_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=config.SR, n_mels=config.N_MELS,
        n_fft=2048, hop_length=512
    )
    return librosa.power_to_db(mel, ref=np.max)

def spec_augment(mel, freq_mask=30, time_mask=60):
    mel = mel.copy()
    n_mels, n_frames = mel.shape
    mean_val = mel.mean()
    for _ in range(2):
        f = np.random.randint(0, freq_mask)
        f0 = np.random.randint(0, max(1, n_mels - f))
        mel[f0:f0+f, :] = mean_val
    for _ in range(2):
        t = np.random.randint(0, time_mask)
        t0 = np.random.randint(0, max(1, n_frames - t))
        mel[:, t0:t0+t] = mean_val
    return mel


class MashupDataset(Dataset):
    def __init__(self, genres, num_samples, train=True):
        self.genres = genres
        self.num_samples = num_samples
        self.train = train

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        genre = random.choice(self.genres)
        label = GENRE2IDX[genre]
        audio = create_stem_mashup(genre)
        if self.train:
            audio = add_noise(audio)
        audio = pad_or_trim(audio)
        mel = audio_to_mel(audio)
        if self.train:
            mel = spec_augment(mel)
        return torch.tensor(mel, dtype=torch.float32).unsqueeze(0), label


class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self._make_block(1, 64)
        self.conv2 = self._make_block(64, 128)
        self.conv3 = self._make_block(128, 256)
        self.conv4 = self._make_block(256, 512)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 256, 1), nn.ReLU(),
            nn.Conv2d(256, 512, 1), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(512, 256), nn.ReLU(),
            nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        return self.classifier(x * self.attention(x))


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.base = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        self.base.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1408, 512), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base(x)


class CRNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),   nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, 3, padding=1),  nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
        )
        self.gru = nn.GRU(
            input_size=256 * 8,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B, T, C * F)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        return self.classifier(x)


def train_model(model, model_name, train_loader, val_loader):
    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        name=model_name,
        config={
            "model": model_name,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "lr": config.LR,
            "train_samples": config.TRAIN_SAMPLES,
            "val_samples": config.VAL_SAMPLES,
            "n_mels": config.N_MELS,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "loss": "CrossEntropyLoss(label_smoothing=0.1)"
        }
    )

    model = model.to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_f1 = 0
    patience_counter = 0
    best_model_path = f'best_{model_name.replace(" ", "_")}.pth'

    print(f"\nTraining: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(config.EPOCHS):
        model.train()
        train_losses = []

        for x, y in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}/{config.EPOCHS}"):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(config.DEVICE)
                preds = model(x).argmax(1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(y.numpy())

        train_loss = np.mean(train_losses)
        val_f1  = f1_score(val_labels, val_preds, average='macro')
        val_acc = accuracy_score(val_labels, val_preds)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, F1={val_f1:.4f}, Acc={val_acc:.4f}, LR={current_lr:.6f}")

        wandb.log({
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "val_f1":     val_f1,
            "val_acc":    val_acc,
            "lr":         current_lr
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best F1: {best_f1:.4f}")
            wandb.summary["best_f1"]  = best_f1
            wandb.summary["best_acc"] = val_acc
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"\n{model_name} done. Best F1: {best_f1:.4f}")
    wandb.finish()

    model.load_state_dict(torch.load(best_model_path))
    return model, best_f1, best_model_path


wandb.login(key="your_wandb_api_key_here")

train_ds = MashupDataset(GENRES, config.TRAIN_SAMPLES, train=True)
val_ds   = MashupDataset(GENRES, config.VAL_SAMPLES,   train=False)

train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

models_to_train = [
    ("EnhancedCNN",    EnhancedCNN(num_classes=len(GENRES))),
    ("EfficientNetB2", EfficientNetModel(num_classes=len(GENRES))),
    ("CRNN",           CRNN(num_classes=len(GENRES))),
]

results = {}

for model_name, model in models_to_train:
    trained_model, best_f1, model_path = train_model(
        model, model_name, train_loader, val_loader
    )
    results[model_name] = {
        "model": trained_model,
        "best_f1": best_f1,
        "path": model_path
    }

print("\nModel results:")
for name, res in results.items():
    print(f"  {name}: F1 = {res['best_f1']:.4f}")

best_model_name = max(results, key=lambda k: results[k]['best_f1'])
print(f"\nBest model: {best_model_name} (F1={results[best_model_name]['best_f1']:.4f})")
