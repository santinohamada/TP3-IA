import json, zipfile, os, shutil
import tensorflow as tf
from tensorflow import keras
import numpy as np

def patch_keras(filename, fixed_name):
    with zipfile.ZipFile(filename, 'r') as z:
        z.extractall("temp")
    with open("temp/config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)
    def remove_bad_keys(d):
        if isinstance(d, dict):
            d.pop("quantization_config", None)
            d.pop("renorm", None)
            d.pop("renorm_clipping", None)
            d.pop("renorm_momentum", None)
            for k, v in list(d.items()): remove_bad_keys(v)
        elif isinstance(d, list):
            for item in d: remove_bad_keys(item)
    remove_bad_keys(config)
    with open("temp/config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f)
    with zipfile.ZipFile(fixed_name, 'w') as z:
        for root, dirs, files in os.walk("temp"):
            for file in files:
                p = os.path.join(root, file)
                z.write(p, os.path.relpath(p, "temp"))
    shutil.rmtree("temp")

print("Extracting VGG16 weights...")
patch_keras("mejor_modelo_VGG16_keras3.bak", "vgg_fixed.keras")
m = keras.models.load_model("vgg_fixed.keras")
np.savez("vgg_weights.npz", *m.get_weights())
os.remove("vgg_fixed.keras")

print("Extracting MBNet weights...")
patch_keras("mejor_modelo_MBNet_keras3.bak", "mbnet_fixed.keras")
m2 = keras.models.load_model("mbnet_fixed.keras")
np.savez("mbnet_weights.npz", *m2.get_weights())
os.remove("mbnet_fixed.keras")
print("All weights extracted successfully.")
