"""
Demo inference script with visualization of accuracy curves and confusion matrices.

Supports:
- Loading trained models (Keras or TFLite)
- Running inference on test dataset
- Plotting training history (accuracy/loss curves)
- Generating confusion matrices
- Optional prediction explanations via OpenRouter
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from image_classifier import ImageClassifier
from utils.openrouter_client import OpenRouterClient


def load_test_dataset(
    dataset_name: str, img_size: int = 224, batch_size: int = 32
) -> Tuple[tf.data.Dataset, List[str]]:
    """
    Load test dataset from TFDS with robust split handling.

    Mirrors the split strategy from image_classifier.load_dataset():
    - If 'test' split exists, use it
    - Otherwise, use train[85%:]

    Args:
        dataset_name: Name of TFDS dataset
        img_size: Image size
        batch_size: Batch size

    Returns:
        Tuple of (test_ds, class_names)
    """
    print(f"Loading {dataset_name} test dataset...")

    # Get available splits
    builder = tfds.builder(dataset_name)
    builder.download_and_prepare()
    info = builder.info
    splits = set(info.splits.keys())

    # Determine which split to use for testing
    if "test" in splits:
        (test_ds,), info = tfds.load(
            dataset_name,
            split=["test"],
            with_info=True,
            as_supervised=True,
        )
        print(f"[data] Using provided test split")
    else:
        (test_ds,), info = tfds.load(
            dataset_name,
            split=["train[85%:]"],
            with_info=True,
            as_supervised=True,
        )
        print(f"[data] Using train[85%:] as test split")

    class_names = info.features["label"].names

    def preprocess(image, label):
        image = tf.image.resize(image, (img_size, img_size))
        image = image / 255.0
        return image, label

    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return test_ds, class_names


def run_inference(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
    use_openrouter: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on test dataset.

    Args:
        model: Trained Keras model
        test_ds: Test dataset
        class_names: List of class names
        use_openrouter: Whether to use OpenRouter for explanations

    Returns:
        Tuple of (predictions, true_labels, confidences)
    """
    print("Running inference on test set...")

    predictions = []
    true_labels = []
    confidences = []

    openrouter = OpenRouterClient() if use_openrouter else None

    for images, labels in test_ds:
        logits = model(images, training=False)
        probs = tf.nn.softmax(logits).numpy()
        preds = np.argmax(probs, axis=1)

        predictions.extend(preds)
        true_labels.extend(labels.numpy())
        confidences.extend(np.max(probs, axis=1))

        # Optional: Show first few predictions with explanations
        if use_openrouter and len(predictions) <= 3:
            for i, (pred, conf) in enumerate(zip(preds, np.max(probs, axis=1))):
                top_3_idx = np.argsort(probs[i])[-3:][::-1]
                top_3_classes = {
                    class_names[idx]: float(probs[i][idx]) for idx in top_3_idx
                }
                explanation = openrouter.explain_prediction(
                    class_names[pred], float(conf), top_3_classes
                )
                print(f"  Sample {len(predictions) + i}: {explanation}")

    return np.array(predictions), np.array(true_labels), np.array(confidences)


def plot_accuracy_curve(
    history: dict, output_path: Optional[str] = None
) -> None:
    """
    Plot training accuracy and loss curves.

    Args:
        history: Training history dictionary
        output_path: Path to save figure (optional)
    """
    print("Plotting accuracy curves...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history["accuracy"], label="Train Accuracy")
    axes[0].plot(history["val_accuracy"], label="Val Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history["loss"], label="Train Loss")
    axes[1].plot(history["val_loss"], label="Val Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Model Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        print(f"  Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
) -> None:
    """
    Plot confusion matrix.

    Args:
        predictions: Model predictions
        true_labels: Ground truth labels
        class_names: List of class names
        output_path: Path to save figure (optional)
    """
    print("Plotting confusion matrix...")

    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        print(f"  Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    """CLI entry point for demo inference."""
    parser = argparse.ArgumentParser(
        description="Run inference and visualize results"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained model (.keras or .tflite)",
    )
    parser.add_argument(
        "--dataset",
        choices=ImageClassifier.SUPPORTED_DATASETS,
        default="tf_flowers",
        help="Dataset to evaluate on (default: tf_flowers)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image size (default: 224)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to save plots (default: artifacts)",
    )
    parser.add_argument(
        "--use-openrouter",
        action="store_true",
        help="Use OpenRouter for prediction explanations",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plotting (only show metrics)",
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    if args.model_path.endswith(".keras"):
        model = tf.keras.models.load_model(args.model_path)
    elif args.model_path.endswith(".tflite"):
        print("  Note: TFLite models require special handling for inference")
        raise NotImplementedError("TFLite inference not yet implemented")
    else:
        raise ValueError("Model must be .keras or .tflite")

    # Load test dataset
    test_ds, class_names = load_test_dataset(
        args.dataset, args.img_size, args.batch_size
    )

    # Run inference
    predictions, true_labels, confidences = run_inference(
        model, test_ds, class_names, args.use_openrouter
    )

    # Calculate metrics
    accuracy = np.mean(predictions == true_labels)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Mean Confidence: {np.mean(confidences):.4f}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Plot confusion matrix
    if not args.no_plots:
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(predictions, true_labels, class_names, str(cm_path))

    print("\nInference complete!")


if __name__ == "__main__":
    main()

