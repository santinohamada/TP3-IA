import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import VGG16, MobileNetV2

print("Rebuilding VGG16 in Keras 2...")
vgg_weights = np.load("vgg_weights.npz")
num_classes = vgg_weights[vgg_weights.files[-1]].shape[0]

base_vgg = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
model_vgg = Sequential([
    base_vgg,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])
model_vgg.set_weights([vgg_weights[k] for k in vgg_weights.files])
model_vgg.save("mejor_modelo_VGG16.keras")

print("Rebuilding MBNet in Keras 2...")
mb_weights = np.load("mbnet_weights.npz")
base_mb = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
model_mb = Sequential([
    base_mb,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])
model_mb.set_weights([mb_weights[k] for k in mb_weights.files])
model_mb.save("mejor_modelo_MBNet.keras")
print("Rebuilt both models purely in Keras 2! No format translation bugs!")
