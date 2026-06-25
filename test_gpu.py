import tensorflow as tf
import numpy as np

print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)
print("Dispositivos:", tf.config.list_physical_devices())
print("GPUs:", tf.config.list_physical_devices("GPU"))

with tf.device("/GPU:0"):
    a = tf.random.normal([3000, 3000])
    b = tf.random.normal([3000, 3000])
    c = tf.matmul(a, b)

print(c.device)
print("Listo! TensorFlow y DirectML funcionando.")