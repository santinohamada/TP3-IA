# TP3 - Clasificación de Imágenes con Redes Neuronales Convolucionales

Trabajo Práctico N°3 de Inteligencia Artificial. Se comparan tres arquitecturas de redes neuronales convolucionales (CNN) para clasificar imágenes de animales en 10 categorías distintas.

---

## Tabla de Contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Estructura del Repositorio](#estructura-del-repositorio)
3. [Dataset](#dataset)
4. [Arquitecturas Utilizadas](#arquitecturas-utilizadas)
5. [Resultados en Test](#resultados-en-test)
6. [Glosario de Conceptos Clave](#glosario-de-conceptos-clave)
7. [Requisitos y Setup](#requisitos-y-setup)

---

## Descripción del Proyecto

El objetivo es entrenar tres modelos distintos para clasificar imágenes de animales y comparar su rendimiento:

| Notebook | Modelo | Estrategia |
|---|---|---|
| `TP3_IA-CM.ipynb` | **Custom CNN** | Arquitectura diseñada desde cero |
| `TP3_IA-VGG16.ipynb` | **VGG16** | Transfer Learning (pesos de ImageNet) |
| `TP3_IA-MBNet.ipynb` | **MobileNetV2** | Transfer Learning (pesos de ImageNet) |
| `TP3_IA-Comparacion.ipynb` | **Comparación** | Evaluación y comparación de los 3 modelos |

---

## Estructura del Repositorio

```
TP3-IA/
│
├── Animals_Dataset_Splitted/
│   ├── train/          # 70% de las imágenes por clase
│   ├── validation/     # 15% de las imágenes por clase
│   └── test/           # 15% de las imágenes por clase
│
├── TP3_IA-CM.ipynb             # Entrenamiento modelo Custom CNN
├── TP3_IA-VGG16.ipynb          # Entrenamiento modelo VGG16
├── TP3_IA-MBNet.ipynb          # Entrenamiento modelo MobileNetV2
├── TP3_IA-Comparacion.ipynb    # Comparación de los 3 modelos
│
├── mejor_modelo_CM.keras       # Modelo entrenado (Custom)
├── mejor_modelo_VGG16.keras    # Modelo entrenado (VGG16)
├── mejor_modelo_MBNet.keras    # Modelo entrenado (MobileNetV2)
│
└── README.md
```

---

## Dataset

El dataset **Animals_Dataset_Splitted** contiene imágenes del mundo real organizadas en **10 clases** (Araña, Ardilla, Caballo, Elefante, Gallina, Gato, Mariposa, Oveja, Perro, Vaca). Las imágenes se redimensionan a `224x224` píxeles para que sean compatibles con los tres modelos.

**Distribución:**
- **Train:** 70% de las imágenes
- **Validation:** 15% de las imágenes
- **Test:** 15% de las imágenes

**Desbalance de Clases:**
El dataset presenta un fuerte desbalance (ej. Perros ~4800 imágenes, Elefantes ~1400 imágenes). Para solucionar esto durante el entrenamiento, se implementa la técnica de **Class Weights**, que penaliza matemáticamente los errores en las clases minoritarias.

### Data Augmentation (Solo en entrenamiento)

Para mejorar la generalización del modelo y evitar el overfitting, se aplican transformaciones aleatorias **únicamente al conjunto de entrenamiento**:

| Parámetro | Valor | Descripción |
|---|---|---|
| `rotation_range` | 20° | Rotación aleatoria de hasta ±20 grados |
| `width_shift_range` | 0.2 | Desplazamiento horizontal de hasta 20% |
| `height_shift_range` | 0.2 | Desplazamiento vertical de hasta 20% |
| `zoom_range` | 0.2 | Zoom aleatorio de hasta 20% |
| `horizontal_flip` | True | Espejo horizontal aleatorio |

---

## Arquitecturas Utilizadas

### 1. Custom CNN (`TP3_IA-CM.ipynb`)

Red convolucional diseñada e implementada desde cero.

**Arquitectura Base:**
```
Rescaling(1./255)           → Normalización de píxeles [0,255] → [0,1]
Conv2D(32) → MaxPooling2D()
Conv2D(64) → MaxPooling2D()
Conv2D(128) → MaxPooling2D()
Conv2D(256) → BatchNormalization()
```

**Cabeza de clasificación (Compartida en todos los modelos):**
```
GlobalAveragePooling2D()
Dense(256, ReLU)
Dropout(0.4)
Dense(10, Softmax)           → Salida: 10 clases
```

**Configuración de entrenamiento:**
- **Optimizer:** Adam (learning rate por defecto: `0.001`)
- **Loss:** Sparse Categorical Crossentropy
- **Métrica:** Accuracy
- **EarlyStopping:** `patience=5`, monitorea `val_accuracy`, `restore_best_weights=True`

**✅ Ventajas:** Total control sobre la arquitectura; modelo liviano; aprende representaciones específicas desde cero.
**❌ Desventajas:** Requiere más epochs para aprender; alcanza un límite estructural ante la complejidad del dataset (Accuracy ~71%).

---

### 2. VGG16 con Transfer Learning (`TP3_IA-VGG16.ipynb`)

Se utiliza la red **VGG16**, preentrenada en **ImageNet** (14 millones de imágenes), aplicando Transfer Learning en modo **Feature Extraction** (capas congeladas).

**Arquitectura de VGG16 (base congelada):**
*13 capas convolucionales agrupadas en 5 bloques. Total parámetros base: ~14.7M (no entrenables).*

**Cabeza de clasificación:**
```
GlobalAveragePooling2D()
Dense(256, ReLU)
Dropout(0.3)
Dense(10, Softmax)           → Salida: 10 clases
```

**Configuración de entrenamiento:**
- **Optimizer:** `Adam(learning_rate=1e-4)` (reducido para Transfer Learning)
- **Loss:** Sparse Categorical Crossentropy
- **Preprocesamiento:** `preprocess_input` de VGG16 (centra en 0 usando la media de ImageNet).

**✅ Ventajas:** Las features extraídas de ImageNet son muy ricas; converge rápido y logra la mayor Accuracy (~96%).
**❌ Desventajas:** Modelo muy pesado (~60 MB); muy lento en inferencia; consume mucha memoria RAM/VRAM.

---

### 3. MobileNetV2 con Transfer Learning (`TP3_IA-MBNet.ipynb`)

Se utiliza **MobileNetV2**, arquitectura preentrenada en ImageNet y diseñada para ser extremadamente eficiente en dispositivos con recursos limitados.

**Innovación clave:** Usa **Depthwise Separable Convolutions** (Convoluciones separables), logrando una precisión altísima con una fracción de los parámetros de VGG16 (53 capas profundas).

**Cabeza de clasificación:**
```
GlobalAveragePooling2D()
Dense(256, ReLU)
Dropout(0.3)
Dense(10, Softmax)           → Salida: 10 clases
```

**Configuración de entrenamiento:**
- **Optimizer:** `Adam(learning_rate=1e-4)`
- **Loss:** Sparse Categorical Crossentropy
- **Preprocesamiento:** `preprocess_input` de MobileNetV2 (normaliza a rango `[-1, 1]`).

**✅ Ventajas:** Muy eficiente; rápido en entrenamiento e inferencia; ideal para entornos de producción (Mobile/Edge); logra Accuracy casi idéntico a VGG16 (~95%).
**❌ Desventajas:** La arquitectura interna (Inverted Residuals) es compleja de analizar matemáticamente a mano.

---

## Resultados en Test

| Modelo | Accuracy Test | F1-Score | Observación principal |
|---|---|---|---|
| **VGG16** | **~ 96%** | Excelente | Máxima capacidad de abstracción. Lento y pesado. |
| **MobileNetV2** | **~ 95%** | Excelente | Ganador absoluto en eficiencia (Costo/Beneficio). |
| **Custom CNN** | **~ 71%** | Aceptable | Límite matemático estructural alcanzado al entrenar desde cero. |

> Al compartir la misma "cabeza" clasificadora de 256 neuronas, la enorme diferencia de rendimiento demuestra empíricamente la superioridad absoluta del Transfer Learning frente a redes construidas desde cero para problemas de visión complejos.

---

## Glosario de Conceptos Clave

### Redes Neuronales y Arquitectura
* **CNN:** Red neuronal para procesamiento de imágenes. Usa convolución para extraer características locales (bordes, formas).
* **GlobalAveragePooling2D (GAP):** Alternativa al Flatten. Colapsa el espacio de la imagen a un solo promedio por canal, reduciendo drásticamente los parámetros y previniendo el overfitting.
* **Dropout:** Técnica de regularización que apaga neuronas aleatoriamente durante el entrenamiento, forzando a la red a no depender de un solo camino de decisión.
* **Softmax:** Función final que convierte las salidas crudas en una distribución de probabilidades que suma 100%.

### Entrenamiento y Control (Callbacks)
* **Class Weights:** Multiplicador matemático que castiga más a la red si se equivoca prediciendo una clase minoritaria, solucionando el desbalance de datos.
* **EarlyStopping:** Frena el entrenamiento si la métrica no mejora durante un número X de epochs (patience) y restaura la mejor versión histórica de los pesos.
* **ReduceLROnPlateau:** Si el entrenamiento se estanca, reduce la "velocidad de aprendizaje" para dar pasos matemáticos más pequeños y precisos.
* **Data Augmentation:** Deformación en tiempo real de las imágenes durante el entrenamiento (rotación, espejado) para que la red no memorice los píxeles.

### Métricas
* **Accuracy:** Porcentaje puro de aciertos totales.
* **F1-Score:** Media armónica entre la Precisión (Falsos Positivos) y el Recall (Falsos Negativos).
* **AUC (Área Bajo la Curva ROC):** Mide la capacidad estadística del modelo para separar matemáticamente a las 10 clases de manera perfecta.

---

## Requisitos y Setup

**Dependencias:**
```
tensorflow>=2.10.0
pillow
matplotlib
seaborn
scikit-learn
```

**Para ejecutar los notebooks:**
1. Activar el entorno virtual: `.venv/Scripts/Activate.ps1` (Windows)
2. Abrir los notebooks en orden: primero los de entrenamiento (`CM`, `VGG16`, `MBNet`), luego el de comparación.
3. Los modelos `.keras` se guardan automáticamente tras ejecutar los callbacks de Checkpoint.
