import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

test_dir = 'Animals_Dataset_Splitted/test'
if not os.path.exists(test_dir):
    print(f"Error: No se encontró {test_dir}")
    exit(1)

model_cm = load_model("mejor_modelo_CM.keras", compile=False)
model_vgg16 = load_model("mejor_modelo_VGG16.keras", compile=False)
model_mbnet = load_model("mejor_modelo_MBNet.keras", compile=False)

for model in [model_cm, model_vgg16, model_mbnet]:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

datagen_cm = ImageDataGenerator(rescale=1./255)
datagen_vgg16 = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
datagen_mbnet = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

test_gen_cm = datagen_cm.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='sparse', shuffle=False)
test_gen_vgg16 = datagen_vgg16.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='sparse', shuffle=False)
test_gen_mbnet = datagen_mbnet.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='sparse', shuffle=False)

class_names = list(test_gen_cm.class_indices.keys())

def get_report(model, gen, name):
    print(f"\nGenerando reporte para {name}...")
    preds = model.predict(gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = gen.classes
    acc = accuracy_score(y_true, y_pred)
    loss = model.evaluate(gen, verbose=0)[0]
    print(f"{name} - Acc: {acc:.4f}, Loss: {loss:.4f}")
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    rep['loss'] = loss
    return rep

rep_cm = get_report(model_cm, test_gen_cm, "Custom")
rep_vgg16 = get_report(model_vgg16, test_gen_vgg16, "VGG16")
rep_mbnet = get_report(model_mbnet, test_gen_mbnet, "MobileNetV2")

with open('reportes.json', 'w') as f:
    json.dump({'CM': rep_cm, 'VGG16': rep_vgg16, 'MBNet': rep_mbnet}, f, indent=2)
print("Reportes guardados en reportes.json")
