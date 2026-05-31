import tensorflow as tf
import numpy as np
import json

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.2)
])

base = tf.keras.applications.MobileNetV2(
    input_shape=(128,128,3), 
    include_top=False, 
    weights=None
)

model = tf.keras.Sequential([
    data_augmentation,
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(12, activation='softmax')
])

# Build the model first
model.build((None, 128, 128, 3))

for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name} -> weights={len(layer.get_weights())}")

with open('model/mobilenet_p1.json') as f:
    p1 = json.load(f)
with open('model/mobilenet_p2.json') as f:
    p2 = json.load(f)
mobilenet_weights = [np.array(w) for w in p1 + p2]
model.layers[1].set_weights(mobilenet_weights)
print('MobileNet loaded!')

with open('model/dense_weights.json') as f:
    dense = json.load(f)
model.layers[3].set_weights([np.array(w) for w in dense['dense1']])
model.layers[5].set_weights([np.array(w) for w in dense['dense2']])
print('Dense loaded!')

dummy = np.zeros((1, 128, 128, 3))
out = model(dummy, training=False)
print('Shape:', out.shape)

model.save('model/garbage_classifier_saved')
print('Saved!')
