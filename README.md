# Lightweight Image Classifier

This project is a lightweight image classification framework built with TensorFlow and TensorFlow Lite.  
It's designed to train and run entirely on CPU - no GPU or CUDA required - while keeping training time short and inference fast.  

The model uses **transfer learning** (MobileNetV2 by default) and can be fine-tuned on small datasets like flowers, beans, or rock–paper–scissors.  
It exports optimized `.tflite` models for running on low-power devices or edge environments.

---

## Key Features

- Works with **small public datasets** (`tf_flowers`, `beans`, `rock_paper_scissors`)
- Automatically creates **train / validation / test splits**
- Uses **MobileNetV2** for fast, accurate transfer learning
- Exports to **TensorFlow Lite** in multiple quantization formats (`float32`, `dynamic`, `float16`, `int8`)
- Generates **training curves** and **confusion matrix** images automatically
- Fully **CPU-compatible** (no GPU dependencies)
- Optional **OpenRouter API** integration for text-based model insights or documentation

---

## Quick Start

### 1. Environment setup

```
python3 -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 2. Train the model

```
python -m src.image_classifier   --dataset tf_flowers   --epochs 10   --batch-size 32   --export-quantizations dynamic float16   --artifacts-dir artifacts
```

This will:
- Download the dataset
- Train a fine-tuned MobileNetV2 model
- Evaluate it on a test split
- Export `.keras` and `.tflite` versions
- Save plots (`history.png`, `confusion_matrix.png`)

---

## Inference Demo

After training, you can test your model on sample images:

```
python -m src.demo_inference   --model-path artifacts/model_dynamic.tflite   --images path/to/image1.jpg path/to/image2.jpg
```

You'll get top class predictions and confidence scores.


## Example Commands

Train with different datasets or parameters:

```
# Use beans dataset
python -m src.image_classifier --dataset beans --epochs 15 --batch-size 64

# Skip TFLite export
python -m src.image_classifier --dataset tf_flowers --no-export

# Evaluate trained model
python -m src.demo_inference --model-path artifacts/model.keras
```

---

## TFLite Model Options

| Format | Size | Speed | Accuracy | When to Use |
|--------|------|-------|-----------|--------------|
| float32 | ~14 MB | 1x | 100% | Baseline / reference |
| dynamic | ~7 MB | 1.5x | ~99% | General CPU deployment |
| float16 | ~5 MB | 2x | ~99% | ARM devices / mobile |
| int8 | ~3 MB | 4x | ~95% | Edge or IoT devices |

---

## Supported Datasets

| Dataset | Classes | Size | Split Strategy |
|----------|----------|-------|----------------|
| `tf_flowers` | 5 | ~3.6k | Auto 70/15/15 |
| `beans` | 3 | ~1.3k | Native train/val/test |
| `rock_paper_scissors` | 3 | ~2.9k | Auto 80/20 |

---

## OpenRouter (Optional)

If you want to use the OpenRouter API for generating explanations or documentation snippets, set your key:

```
export OPENROUTER_API_KEY="sk-your-key"
```

Example usage:

```python
from src.utils.openrouter_client import OpenRouterClient

client = OpenRouterClient()
explanation = client.explain_prediction("daisy", 0.95, {"daisy": 0.95, "rose": 0.04})
print(explanation)
```

---

## Requirements

- Python 3.10–3.12  
- TensorFlow 2.18.0 (CPU build)  
- NumPy, SciPy, Matplotlib, scikit-learn, TensorFlow Datasets  
- OpenCV-Python-Headless, Requests  

All packages are pinned and tested for CPU-only environments (WSL, Linux, macOS).

---

## Performance Summary

- Training: ~5–10 minutes per epoch on CPU  
- Inference: ~50–100 ms per image  
- Test accuracy (tf_flowers): **~ 89%**  
- Model sizes: `float32` 13 MB → `float16` 4.9 MB → `dynamic` 2.8 MB  

---

## Common Issues

**Dataset download errors**
```
export TFDS_DATA_DIR=/path/to/tfds_data
```

**Out of memory during training**
```
python -m src.image_classifier --batch-size 16
```

**OpenRouter errors**  
Check that your `OPENROUTER_API_KEY` is set and valid. The program will still run without it.
