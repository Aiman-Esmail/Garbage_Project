import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('model/garbage_classifier_saved')

classes = ['battery', 'biological', 'brown-glass', 'cardboard', 
           'clothes', 'green-glass', 'metal', 'paper', 
           'plastic', 'shoes', 'trash', 'white-glass']

# اختبر بصورة عشوائية
dummy = np.random.rand(1, 128, 128, 3).astype(np.float32)
pred = model(dummy, training=False)
print('Random image prediction:', classes[np.argmax(pred)])
print('All probs:', [f'{p:.3f}' for p in pred[0]])
