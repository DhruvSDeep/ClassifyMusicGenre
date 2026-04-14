# ClassifyMusicGenre

A convolutional neural network that classifies **music genre from audio** by converting recordings into mel spectrogram images and feeding them through a CNN. Designed for the [GTZAN](http://marsyas.info/downloads/datasets.html) dataset, the model distinguishes between 10 genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock.

---

## Highlights

- **Audio → spectrogram → classification pipeline** — converts `.wav` / `.mp3` files to mel spectrogram images, then classifies them with a lightweight CNN.
- **Simple and interpretable architecture** — 2 convolutional layers followed by a single fully connected layer. No complex pooling tricks or deep stacks; easy to understand and extend.
- **Best-model checkpointing** — the training loop tracks validation loss and saves the best weights automatically.
- **Full evaluation suite** — confusion matrix visualization and per-class classification report on the held-out test set.

---

## Model Architecture

```
Input (3-channel mel spectrogram image)
  │
  ├─ Conv2d(3 → 8, 3×3, padding=1) → ReLU → MaxPool(2×2)
  ├─ Conv2d(8 → 16, 3×3, padding=1) → ReLU → MaxPool(2×2)
  │
  ├─ Flatten
  └─ Linear(16 × 72 × 108 → 10)
```

The network is intentionally compact. Additional convolutional layers (32 and 64 channels) are defined in the code for experimentation but are not active in the current configuration.

---

## Project Structure

```
ClassifyMusicGenre/
├── BackUpButFront.ipynb   # Full pipeline: spectrogram generation, training, evaluation
└── README.md
```

At runtime the notebook creates:
- `music_dataset/Data/genres_original_img/` — mel spectrogram images organized by genre folder.
- `bestModel.pth` — saved weights for the best checkpoint.

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch, librosa, scikit-learn, torchmetrics, matplotlib, Pillow

```bash
pip install torch torchvision librosa scikit-learn torchmetrics matplotlib pillow
```

### Dataset

Download the [GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html) and place the audio files under `music_dataset/Data/genres_original/`, with one subfolder per genre:

```
music_dataset/Data/genres_original/
├── blues/
├── classical/
├── country/
├── disco/
├── hiphop/
├── jazz/
├── metal/
├── pop/
├── reggae/
└── rock/
```

### Train & Evaluate

Open and run `BackUpButFront.ipynb` end-to-end. The notebook will:

1. Convert every audio file into a mel spectrogram saved as a borderless JPG, organized by genre.
2. Load the spectrogram images via `torchvision.datasets.ImageFolder`.
3. Split the data 70 / 20 / 10 into train, validation, and test sets.
4. Train for 25 epochs with Adam, saving the best model by validation loss.
5. Report test accuracy, a per-class classification report, and a confusion matrix.

Training hyperparameters:

| Parameter | Value |
|---|---|
| Batch size | 99 |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻³ |
| Loss | Cross-entropy |
| Epochs | 25 |
| Data split | 70% train / 20% valid / 10% test |

---

## How It Works

### Spectrogram Generation

Each audio file is loaded with librosa at the default 22 050 Hz sample rate. A mel spectrogram is computed, converted to decibel scale (referenced to the maximum), and saved as a borderless JPG image with no axes or labels. The resulting images serve as RGB inputs to the CNN via PyTorch's `ImageFolder`.

### Training

The model is trained with standard cross-entropy loss and the Adam optimizer. After each epoch, the validation loss is computed; if it improves, the model weights are saved to `bestModel.pth`. No data augmentation or regularization (dropout, weight decay) is applied in the current configuration, keeping the baseline simple.

### Evaluation

The best checkpoint is loaded and evaluated on the held-out test set. The notebook produces a test accuracy score (via `torchmetrics.Accuracy`), a confusion matrix heatmap, and a full `sklearn` classification report with per-genre precision, recall, and F1 scores.

---

## Possible Improvements

The codebase is structured for easy experimentation. Some directions to explore:

- **Enable the deeper layers** — `conv3` and `conv4` are already defined; uncommenting them and adjusting the flatten dimensions adds depth.
- **Add regularization** — BatchNorm, dropout, or weight decay to reduce overfitting (as explored in the author's [EmotionSpeechRecognitionCNN](https://github.com/DhruvSDeep/EmotionSpeechRecognitionCNN)).
- **Data augmentation** — pitch shifting, time stretching, or spectrogram-level augmentations like SpecAugment.
- **Global pooling** — replacing the fixed flatten with GAP/GMP makes the model resolution-independent.

---

## License

No license specified. Contact the repository owner for usage terms.
