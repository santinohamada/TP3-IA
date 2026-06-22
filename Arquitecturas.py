from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import os

os.makedirs('imagenes', exist_ok=True)

modelos = {
    'modelo_peces_Custom.keras': 'imagenes/arquitectura_custom_cnn.png',
    'modelo_peces_vgg16.keras': 'imagenes/arquitectura_vgg16.png',
    'modelo_peces_MobileNetV2.keras': 'imagenes/arquitectura_mobilenetv2.png',
}

for path, out in modelos.items():
    model = load_model(path)
    plot_model(model, to_file=out, show_shapes=True, show_layer_names=True, dpi=150)
    print(f"Guardado: {out}")
