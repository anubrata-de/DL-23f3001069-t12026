# Messy Mashup - Audio Genre Classification

**Name:** 23f3001069  
**Roll No:** 23f3001069  
**Term:** T1-2026  
**Course:** Deep Learning & Generative AI Project

## Problem Statement

Multi-class audio genre classification across 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) using noisy stem mashups. Evaluated on Macro F1 score.

## Models

| Model | Type | Description |
|-------|------|-------------|
| EnhancedCNN | From Scratch | 4 conv blocks with batch norm and attention mechanism |
| EfficientNet-B2 | Pretrained | ImageNet pretrained, fine-tuned for audio mel spectrograms |
| CRNN | From Scratch | CNN frontend + Bidirectional GRU for temporal modeling |

## Project Structure

```
/notebooks       - Kaggle training and inference notebooks
/scripts         - Standalone Python scripts for training and preprocessing
/data            - Sample data, EDA outputs
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Training is done on Kaggle with GPU. The notebook handles:
- Stem mashup generation with augmentation
- ESC-50 noise injection
- Mel spectrogram conversion
- SpecAugment during training
- Test Time Augmentation (TTA) during inference

## Experiment Tracking

All runs tracked on Weights & Biases:  
Project: `23f3001069-t12026`  
Entity: `23f3001069-23f3001069-t12026`

## Results

| Model | Val F1 | Val Accuracy |
|-------|--------|--------------|
| EnhancedCNN | - | - |
| EfficientNet-B2 | - | - |
| CRNN | - | - |

*(Results will be updated after training completes)*

## Kaggle Competition

Competition: Messy Mashup - Jan 2026 DL GenAI Project  
Target Score: 0.80 Macro F1
