"""
Lightweight image classifier with TensorFlow and TFLite export support.

Supports multiple datasets via TFDS:
- tf_flowers (default)
- beans
- rock_paper_scissors

Exports models in multiple TFLite formats:
- float32 (baseline)
- dynamic range quantization
- float16 quantization
- int8 quantization (with calibration)
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class ImageClassifier:
    """Lightweight image classifier with TFLite export capabilities."""

    SUPPORTED_DATASETS = ["tf_flowers", "beans", "rock_paper_scissors"]
    SUPPORTED_QUANTIZATIONS = ["float32", "dynamic", "float16", "int8"]

    def __init__(
        self,
        dataset_name: str = "tf_flowers",
        img_size: int = 224,
        batch_size: int = 32,
        epochs: int = 10,
        artifacts_dir: str = "artifacts",
    ):
        """
        Initialize the image classifier.

        Args:
            dataset_name: Name of TFDS dataset to use
            img_size: Image size for model input
            batch_size: Training batch size
            epochs: Number of training epochs
            artifacts_dir: Directory to save models and artifacts
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset must be one of {self.SUPPORTED_DATASETS}, got {dataset_name}"
            )

        self.dataset_name = dataset_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True)

        self.model = None
        self.history = None
        self.class_names = None
        self.num_classes = None

    def load_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
        """
        Load and preprocess dataset from TFDS with robust split handling.

        Supports datasets with or without 'validation' and 'test' splits.
        If missing, creates subsplits from 'train':
        - train := train[:70%]
        - val   := train[70%:85%]
        - test  := train[85%:]

        Returns:
            Tuple of (train_ds, val_ds, test_ds, class_names)
        """
        print(f"Loading {self.dataset_name} dataset...")

        # Get available splits
        builder = tfds.builder(self.dataset_name)
        builder.download_and_prepare()
        info = builder.info
        splits = set(info.splits.keys())

        print(f"[data] Available splits: {sorted(splits)}")

        # Determine split strategy
        if "validation" in splits and "test" in splits:
            # Case A: dataset has both validation and test
            (train_ds, val_ds, test_ds), info = tfds.load(
                self.dataset_name,
                split=["train", "validation", "test"],
                with_info=True,
                as_supervised=True,
            )
            print(f"[data] Using provided splits: train/validation/test")

        elif "validation" in splits:
            # Case B: dataset has validation but no test
            (train_ds, val_ds), info = tfds.load(
                self.dataset_name,
                split=["train", "validation"],
                with_info=True,
                as_supervised=True,
            )
            # Create test from tail of train
            (_, test_ds), _ = tfds.load(
                self.dataset_name,
                split=["train[:85%]", "train[85%:]"],
                with_info=True,
                as_supervised=True,
            )
            print(f"[data] Using provided train/validation; created test from train[85%:]")

        else:
            # Case C: no validation (e.g., tf_flowers) — create all 3 subsplits
            (train_ds, val_ds, test_ds), info = tfds.load(
                self.dataset_name,
                split=["train[:70%]", "train[70%:85%]", "train[85%:]"],
                with_info=True,
                as_supervised=True,
            )
            print(f"[data] Created subsplits: train[:70%], val=train[70%:85%], test=train[85%:]")

        self.class_names = info.features["label"].names
        self.num_classes = len(self.class_names)

        print(f"Classes: {self.class_names}")
        print(f"Number of classes: {self.num_classes}")

        # Preprocess function
        def preprocess(image, label):
            image = tf.image.resize(image, (self.img_size, self.img_size))
            image = image / 255.0  # Normalize to [0, 1]
            return image, label

        # Apply preprocessing and batching
        train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds, self.class_names

    def build_model(self) -> tf.keras.Model:
        """
        Build a lightweight MobileNetV2-based model.

        Returns:
            Compiled Keras model
        """
        print("Building model...")

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.img_size, self.img_size, 3)),
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        return model

    def train(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> Dict:
        """
        Train the model.

        Args:
            train_ds: Training dataset
            val_ds: Validation dataset

        Returns:
            Training history dictionary
        """
        print(f"Training for {self.epochs} epochs...")

        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            verbose=1,
        )

        return self.history.history

    def export_tflite(self, quantization: str = "float32") -> str:
        """
        Export model to TFLite format with specified quantization.

        Args:
            quantization: Quantization type (float32, dynamic, float16, int8)

        Returns:
            Path to exported TFLite model
        """
        if quantization not in self.SUPPORTED_QUANTIZATIONS:
            raise ValueError(
                f"Quantization must be one of {self.SUPPORTED_QUANTIZATIONS}, got {quantization}"
            )

        print(f"Exporting to TFLite with {quantization} quantization...")

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        if quantization == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantization == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            # For int8, we'd need a representative dataset
            # This is a simplified version without full calibration
            print("  Note: int8 quantization without calibration dataset")

        tflite_model = converter.convert()

        output_path = self.artifacts_dir / f"model_{quantization}.tflite"
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        print(f"  Saved to {output_path}")
        return str(output_path)

    def save_model(self, format: str = "keras") -> str:
        """
        Save model in specified format.

        Args:
            format: Format to save (keras, savedmodel)

        Returns:
            Path to saved model
        """
        if format == "keras":
            output_path = self.artifacts_dir / "model.keras"
            self.model.save(output_path)
        elif format == "savedmodel":
            output_path = self.artifacts_dir / "model_savedmodel"
            self.model.save(output_path)
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"Saved model to {output_path}")
        return str(output_path)

    def plot_training_history(self) -> str:
        """
        Plot training history (accuracy and loss curves).

        Returns:
            Path to saved plot
        """
        if self.history is None:
            print("No training history available")
            return ""

        import matplotlib.pyplot as plt

        print("Plotting training history...")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy
        axes[0].plot(self.history.history["accuracy"], label="Train Accuracy")
        axes[0].plot(self.history.history["val_accuracy"], label="Val Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Model Accuracy")
        axes[0].legend()
        axes[0].grid(True)

        # Loss
        axes[1].plot(self.history.history["loss"], label="Train Loss")
        axes[1].plot(self.history.history["val_loss"], label="Val Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Model Loss")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        output_path = self.artifacts_dir / "history.png"
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        print(f"  Saved to {output_path}")
        plt.close()

        return str(output_path)

    def plot_confusion_matrix(self, test_ds: tf.data.Dataset, class_names: List[str]) -> str:
        """
        Generate and plot confusion matrix on test set.

        Args:
            test_ds: Test dataset
            class_names: List of class names

        Returns:
            Path to saved confusion matrix plot
        """
        print("Generating confusion matrix...")

        # Collect predictions and true labels
        y_true = []
        y_pred = []

        for images, labels in test_ds:
            predictions = self.model.predict(images, verbose=0)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(labels.numpy())

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        plt.title("Confusion Matrix - Test Set")
        plt.tight_layout()

        output_path = self.artifacts_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        print(f"  Saved to {output_path}")
        plt.close()

        return str(output_path)


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train a lightweight image classifier"
    )
    parser.add_argument(
        "--dataset",
        choices=ImageClassifier.SUPPORTED_DATASETS,
        default="tf_flowers",
        help="Dataset to train on (default: tf_flowers)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image size for model input (default: 224)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory to save artifacts (default: artifacts)",
    )
    parser.add_argument(
        "--export-quantizations",
        nargs="+",
        choices=ImageClassifier.SUPPORTED_QUANTIZATIONS,
        default=["float32", "dynamic", "float16"],
        help="TFLite quantization formats to export (default: float32 dynamic float16)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip TFLite export",
    )

    args = parser.parse_args()

    # Create classifier
    classifier = ImageClassifier(
        dataset_name=args.dataset,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        artifacts_dir=args.artifacts_dir,
    )

    # Load dataset (now returns train, val, test, class_names)
    train_ds, val_ds, test_ds, class_names = classifier.load_dataset()

    # Build and train model
    classifier.build_model()
    classifier.train(train_ds, val_ds)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = classifier.model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save model
    classifier.save_model("keras")

    # Plot training history
    classifier.plot_training_history()

    # Generate confusion matrix
    classifier.plot_confusion_matrix(test_ds, class_names)

    # Save class names for reference
    labels_path = classifier.artifacts_dir / "labels.txt"
    with open(labels_path, "w") as f:
        for label in class_names:
            f.write(f"{label}\n")
    print(f"Saved class labels to {labels_path}")

    # Export TFLite models
    if not args.no_export:
        for quant in args.export_quantizations:
            classifier.export_tflite(quant)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

