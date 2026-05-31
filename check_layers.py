import tensorflow as tf
import numpy as np
import json

base = tf.keras.applications.MobileNetV2(
    input_shape=(128,128,3), 
    include_top=False, 
    weights=None
)

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(12, activation='softmax')
])

for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name} -> weights={len(layer.get_weights())}")
