"""
Model Conversion Utility
Converts H5 model to SavedModel format.
Architecture: MobileNetV2 + GlobalAveragePooling2D + Dense(256) + Dense(12)
"""

import tensorflow as tf
from tensorflow.keras import Sequential, layers
import os
import sys

# ── Constants ─────────────────────────────────────────────────────────────────
H5_PATH         = "model/Garbage_Classifier_Final_95.h5"
SAVEDMODEL_PATH = "model/garbage_classifier_saved"
IMG_SIZE        = (128, 128, 3)
NUM_CLASSES     = 12


def build_model() -> tf.keras.Model:
    """Rebuild the exact same architecture used during training."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE,
        include_top=False,
        weights=None  # weights will be loaded from H5
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model


def convert():
    print("=" * 50)
    print("  Garbage Classifier — Model Conversion")
    print("=" * 50)

    # ── Check H5 exists ───────────────────────────────────────────────────────
    if not os.path.exists(H5_PATH):
        print(f"\n❌ Model file not found: {H5_PATH}")
        print("   Make sure the .h5 file is inside the 'model/' folder.")
        return False

    # ── Attempt 1: Direct load ────────────────────────────────────────────────
    print(f"\n[1/3] Loading model from {H5_PATH} ...")
    try:
        model = tf.keras.models.load_model(H5_PATH, compile=False)
        print("      ✅ Model loaded successfully.")

    except Exception as e:
        print(f"      ⚠️  Direct load failed: {e}")
        print("\n[1/3] Attempting architecture rebuild + weight transfer...")

        try:
            model = build_model()
            model.load_weights(H5_PATH, by_name=True, skip_mismatch=True)
            print("      ✅ Weights transferred successfully.")
        except Exception as e2:
            print(f"      ❌ Weight transfer failed: {e2}")
            return False

    # ── Attempt 2: Save as SavedModel ─────────────────────────────────────────
    print(f"\n[2/3] Saving to {SAVEDMODEL_PATH} ...")
    try:
        model.save(SAVEDMODEL_PATH)
        print("      ✅ Saved successfully.")
    except Exception as e:
        print(f"      ❌ Save failed: {e}")
        return False

    # ── Attempt 3: Verify ─────────────────────────────────────────────────────
    print(f"\n[3/3] Verifying converted model ...")
    try:
        test_model = tf.keras.models.load_model(SAVEDMODEL_PATH)
        dummy = tf.zeros([1, 128, 128, 3])
        output = test_model(dummy, training=False)
        assert output.shape == (1, NUM_CLASSES), f"Unexpected output shape: {output.shape}"
        print(f"      ✅ Verified — output shape: {output.shape}")
    except Exception as e:
        print(f"      ❌ Verification failed: {e}")
        return False

    print("\n" + "=" * 50)
    print("  ✅ Conversion complete!")
    print(f"  Use this path in app.py: '{SAVEDMODEL_PATH}'")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = convert()
    sys.exit(0 if success else 1)