import tensorflow as tf
import numpy as np
import json

base = tf.keras.applications.MobileNetV2(input_shape=(128,128,3), include_top=False, weights='imagenet')
base.trainable = False

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(12, activation='softmax')
])

with open('model/dense_weights.json', 'r') as f:
    dense_weights = json.load(f)

for layer_data in dense_weights:
    try:
        layer = model.get_layer(layer_data['name'])
        weights = [np.array(w) for w in layer_data['weights']]
        if weights:
            layer.set_weights(weights)
            print('Set:', layer_data['name'])
    except Exception as e:
        print('Skip:', e)

dummy = np.zeros((1, 128, 128, 3))
out = model(dummy, training=False)
print('Shape:', out.shape)
model.save('model/garbage_classifier_saved')
print('Saved!')
