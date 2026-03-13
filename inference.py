import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models

import warnings
warnings.filterwarnings('ignore')


ROOT = "/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup"
MODEL_PATH = "best_model.pth"

SR = 22050
DURATION = 30
MAX_LEN = SR * DURATION
N_MELS = 128
N_TTA = 8

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
IDX2GENRE = {i: g for i, g in enumerate(GENRES)}


def load_audio(path):
    try:
        audio, _ = librosa.load(path, sr=SR, mono=True)
        return audio
    except:
        return np.zeros(MAX_LEN)

def pad_or_trim(audio):
    if len(audio) > MAX_LEN:
        return audio[:MAX_LEN]
    return np.pad(audio, (0, MAX_LEN - len(audio)))

def audio_to_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_mels=N_MELS,
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
        self.base = models.efficientnet_b2(weights=None)
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


MODEL_CLASSES = {
    "EnhancedCNN": EnhancedCNN,
    "EfficientNetB2": EfficientNetModel,
    "CRNN": CRNN,
}

print(f"Device: {DEVICE}")

model_name = "EnhancedCNN"
ModelClass = MODEL_CLASSES[model_name]
model = ModelClass(num_classes=len(GENRES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"Loaded model: {model_name} from {MODEL_PATH}")

test_df = pd.read_csv(os.path.join(ROOT, "test.csv"))
test_audio_path = os.path.join(ROOT, "mashups")
print(f"Test samples: {len(test_df)}")

all_probs = []
sample_ids = []

for tta_pass in range(N_TTA):
    print(f"TTA Pass {tta_pass+1}/{N_TTA}")
    predictions = []

    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            audio = load_audio(os.path.join(test_audio_path, row['filename']))
            audio = pad_or_trim(audio)
            mel = audio_to_mel(audio)

            if tta_pass > 0:
                mel = spec_augment(mel)

            mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            probs = torch.softmax(model(mel_tensor), dim=1).cpu().numpy()[0]

            if tta_pass == 0:
                sample_ids.append(row['id'])

            predictions.append(probs)

    all_probs.append(np.array(predictions))

avg_probs = np.mean(all_probs, axis=0)
final_predictions = [IDX2GENRE[np.argmax(p)] for p in avg_probs]

submission = pd.DataFrame({'id': sample_ids, 'genre': final_predictions})
print(f"Total predictions: {len(submission)}")
print(f"Unique genres: {submission['genre'].nunique()}")
print(submission['genre'].value_counts())

submission.to_csv('submission.csv', index=False)
print("Saved to submission.csv")
print(submission.head(10))
