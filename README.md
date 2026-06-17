# TP3 - Clasificación de Imágenes con Redes Neuronales Convolucionales

Trabajo Práctico N°3 de Inteligencia Artificial. Se comparan tres arquitecturas de redes neuronales convolucionales (CNN) para clasificar imágenes de peces en 5 categorías distintas.

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

El objetivo es entrenar tres modelos distintos para clasificar imágenes de peces y comparar su rendimiento:

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
├── Fish_Dataset_Splitted/
│   ├── train/          # 4000 imágenes (800 por clase)
│   ├── validation/     # 500 imágenes (100 por clase)
│   └── test/           # 500 imágenes (100 por clase)
│
├── TP3_IA-CM.ipynb             # Entrenamiento modelo Custom CNN
├── TP3_IA-VGG16.ipynb          # Entrenamiento modelo VGG16
├── TP3_IA-MBNet.ipynb          # Entrenamiento modelo MobileNetV2
├── TP3_IA-Comparacion.ipynb    # Comparación de los 3 modelos
│
├── modelo_peces_Custom.keras       # Modelo entrenado (Custom)
├── modelo_peces_vgg16.keras        # Modelo entrenado (VGG16)
├── modelo_peces_MobileNetV2.keras  # Modelo entrenado (MobileNetV2)
│
└── README.md
```

---

## Dataset

El dataset **Fish Dataset Splitted** contiene imágenes de peces organizadas en **5 clases**. Las imágenes se redimensionan a `224x224` píxeles para que sean compatibles con los tres modelos.

**Distribución:**
- **Train:** 4000 imágenes (800 por clase)
- **Validation:** 500 imágenes (100 por clase)
- **Test:** 500 imágenes (100 por clase)

### Data Augmentation (Solo en entrenamiento)

Para mejorar la generalización del modelo y reducir el overfitting, se aplican transformaciones aleatorias **únicamente al conjunto de entrenamiento**:

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

**Arquitectura:**
```
Rescaling(1./255)           → Normalización de píxeles [0,255] → [0,1]
Conv2D(32, 3, "same", ReLU)
MaxPooling2D()
Conv2D(64, 3, "same", ReLU)
MaxPooling2D()
Conv2D(128, 3, "same", ReLU)
MaxPooling2D()
Conv2D(256, 3, "same", ReLU)
GlobalAveragePooling2D()
Dense(128, ReLU)
Dropout(0.4)
Dense(5, Softmax)           → Salida: 5 clases
```

**Configuración de entrenamiento:**
- **Optimizer:** Adam (learning rate por defecto: `0.001`)
- **Loss:** Sparse Categorical Crossentropy
- **Métrica:** Accuracy
- **Epochs máximos:** 50
- **EarlyStopping:** `patience=5`, monitorea `val_accuracy`, `restore_best_weights=True`
- **Preprocesamiento:** La capa `Rescaling(1./255)` está integrada dentro del modelo.

**✅ Ventajas:**
- Total control sobre la arquitectura: se puede ajustar cada capa según el problema.
- Modelo liviano en comparación con VGG16 (menos parámetros).
- Aprende representaciones específicas para el dataset de peces desde cero.
- La normalización integrada como capa simplifica el pipeline de inferencia.

**❌ Desventajas:**
- Requiere más epochs y datos para aprender buenas representaciones.
- Es difícil superar el rendimiento de modelos preentrenados en datasets pequeños-medianos.
- Diseñar una buena arquitectura desde cero requiere mucha experimentación.

---

### 2. VGG16 con Transfer Learning (`TP3_IA-VGG16.ipynb`)

Se utiliza la red **VGG16**, preentrenada en **ImageNet** (14 millones de imágenes, 1000 clases), aplicando Transfer Learning en modo **Feature Extraction** (todas las capas congeladas).

**Arquitectura de VGG16 (base congelada):**
```
Input(224, 224, 3)
Block 1: Conv2D(64) → Conv2D(64) → MaxPooling2D
Block 2: Conv2D(128) → Conv2D(128) → MaxPooling2D
Block 3: Conv2D(256) → Conv2D(256) → Conv2D(256) → MaxPooling2D
Block 4: Conv2D(512) → Conv2D(512) → Conv2D(512) → MaxPooling2D
Block 5: Conv2D(512) → Conv2D(512) → Conv2D(512) → MaxPooling2D
```
*Total parámetros base: ~14.7M (todos no entrenables)*

**Cabeza de clasificación añadida:**
```
GlobalAveragePooling2D()
Dense(256, ReLU)
Dropout(0.3)
Dense(5, Softmax)           → Salida: 5 clases
```

**Configuración de entrenamiento:**
- **Optimizer:** `Adam(learning_rate=1e-4)` (learning rate reducido para Transfer Learning)
- **Loss:** Sparse Categorical Crossentropy
- **Métrica:** Accuracy
- **Epochs máximos:** 50
- **EarlyStopping:** `patience=5`, monitorea `val_accuracy`, `restore_best_weights=True`
- **Preprocesamiento:** `preprocess_input` de VGG16 (centra en 0 usando la media de ImageNet, **NO** normaliza a [0,1])

**✅ Ventajas:**
- Las features extraídas de ImageNet son muy ricas y transferibles.
- Converge rápido: logra alta accuracy en pocas epochs.
- Arquitectura simple y bien estudiada.
- Ideal cuando se tiene poco dato de entrenamiento.

**❌ Desventajas:**
- Modelo muy pesado (~60 MB en disco, ~14.7M parámetros solo en la base).
- Lento en inferencia y entrenamiento comparado con MobileNetV2.
- Diseñado para imágenes de 224x224, no es flexible en tamaños de entrada.
- Consume mucha memoria RAM/VRAM.
- No es apto para deployment en dispositivos móviles o edge.

---

### 3. MobileNetV2 con Transfer Learning (`TP3_IA-MBNet.ipynb`)

Se utiliza **MobileNetV2**, arquitectura diseñada para ser eficiente en dispositivos con recursos limitados (móviles, embebidos). Preentrenada en **ImageNet**, aplicando Transfer Learning en modo **Feature Extraction**.

**Innovación clave de MobileNetV2:** Usa **Depthwise Separable Convolutions** e **Inverted Residual Blocks** con conexiones residuales. Esto permite lograr una precisión comparable a arquitecturas más grandes con una fracción de los parámetros y operaciones.

**Cabeza de clasificación añadida:**
```
GlobalAveragePooling2D()
Dense(128, ReLU)
Dropout(0.3)
Dense(5, Softmax)           → Salida: 5 clases
```

**Configuración de entrenamiento:**
- **Optimizer:** `Adam(learning_rate=1e-4)`
- **Loss:** Sparse Categorical Crossentropy
- **Métrica:** Accuracy
- **Epochs máximos:** 50
- **EarlyStopping:** `patience=5`, monitorea `val_accuracy`, `restore_best_weights=True`
- **Preprocesamiento:** `preprocess_input` de MobileNetV2 (normaliza a rango `[-1, 1]`)

**✅ Ventajas:**
- Muy eficiente: alta accuracy con pocos parámetros (~3.4M).
- Rápido en entrenamiento e inferencia.
- Diseñado para dispositivos con recursos limitados (móviles, IoT).
- Excelente balance entre tamaño y rendimiento.
- Las Depthwise Separable Convolutions reducen drásticamente el costo computacional.

**❌ Desventajas:**
- Puede ser menos preciso que modelos más grandes en datasets muy complejos o con alta variabilidad.
- La arquitectura con Inverted Residuals es más compleja de entender e implementar desde cero.
- Para datasets muy grandes y complejos, modelos como VGG16 o ResNet pueden superarlo.

---

## Resultados en Test

| Modelo | Accuracy | Loss |
|---|---|---|
| **MobileNetV2** | **1.0000** | 0.0213 |
| Custom CNN | 0.9980 | 0.0159 |
| VGG16 | 0.9940 | 0.0125 |

> Los tres modelos alcanzaron un rendimiento excelente sobre el conjunto de test. MobileNetV2 obtuvo el mayor accuracy con el menor costo computacional, lo que lo convierte en el ganador general de este experimento.

---

## Glosario de Conceptos Clave

### Redes Neuronales y Arquitectura

**CNN (Convolutional Neural Network)**
Red neuronal diseñada específicamente para procesar datos con estructura de grilla (imágenes). Usa operaciones de convolución para extraer características locales (bordes, texturas, formas) de manera jerárquica.

**Capa Convolucional (Conv2D)**
Aplica filtros (kernels) sobre la imagen de entrada deslizándose por toda la grilla. Cada filtro aprende a detectar una característica particular (bordes horizontales, esquinas, etc.). Los parámetros `filters` y `kernel_size` definen cuántos y qué tan grandes son esos filtros.

**MaxPooling2D**
Operación de reducción espacial. Toma el valor máximo de cada región de la imagen, reduciendo su tamaño a la mitad. Logra dos objetivos: reduce el costo computacional y añade cierta invarianza a traslaciones pequeñas.

**GlobalAveragePooling2D**
Alternativa más moderna al `Flatten`. En lugar de "aplanar" todos los valores, calcula el promedio de cada mapa de características completo. Reduce drásticamente el número de parámetros y actúa como regularizador implícito.

**Dense (Fully Connected)**
Capa clásica de red neuronal donde cada neurona está conectada a todas las neuronas de la capa anterior. Se usa al final de la red para hacer la clasificación final.

**Softmax**
Función de activación de la capa de salida en problemas de clasificación multiclase. Convierte un vector de valores reales en probabilidades que suman 1. La clase predicha es la de mayor probabilidad.

**ReLU (Rectified Linear Unit)**
Función de activación más usada en capas ocultas. Definida como `f(x) = max(0, x)`. Introduce no-linealidad, es computacionalmente eficiente y ayuda a mitigar el problema de los gradientes que desaparecen.

**Dropout**
Técnica de regularización que apaga neuronas aleatoriamente durante el entrenamiento con una probabilidad `p`. Evita que la red memorice el dataset (overfitting) y fuerza a que múltiples caminos aprendan la misma información. En este proyecto se usa `Dropout(0.4)` en la Custom CNN y `Dropout(0.3)` en los modelos de Transfer Learning.

**BatchNormalization**
Normaliza las activaciones de una capa por mini-batch durante el entrenamiento. Estabiliza el proceso de aprendizaje, permite usar learning rates más altos y actúa como regularizador. Es un componente central de MobileNetV2.

---

### Entrenamiento y Optimización

**Epoch**
Una pasada completa sobre todo el dataset de entrenamiento. Si el dataset tiene 4000 imágenes y el batch size es 32, un epoch equivale a 125 iteraciones (pasos de gradiente). En este proyecto se configuran hasta 50 epochs, pero el EarlyStopping puede detenerlo antes.

**Batch / Mini-batch**
Subconjunto del dataset que se procesa en cada iteración antes de actualizar los pesos. En este proyecto `BATCH_SIZE = 32`. Usar mini-batches (en lugar de una sola imagen o todo el dataset) es el balance óptimo entre velocidad y estabilidad del gradiente.

**Gradient Descent**
Algoritmo base de optimización. Calcula el gradiente de la función de pérdida respecto a cada parámetro y los actualiza en la dirección que minimiza esa pérdida. El **learning rate** controla qué tan grande es ese paso.

**Adam (Adaptive Moment Estimation)**
Optimizador avanzado que combina dos ideas: **Momentum** (acumula gradientes pasados para ganar velocidad) y **RMSProp** (adapta el learning rate por parámetro). Es el optimizador por defecto en deep learning y generalmente converge más rápido que el Gradient Descent clásico.

**Learning Rate (Tasa de aprendizaje)**
Hiperparámetro que controla el tamaño del paso en cada actualización de pesos. Un valor muy alto hace que el entrenamiento diverja (los pesos oscilan sin converger). Un valor muy bajo hace que el entrenamiento sea lento o quede atrapado en mínimos locales.
- **Custom CNN:** Learning rate por defecto de Adam = `0.001`
- **VGG16 / MobileNetV2:** `learning_rate=1e-4` (más bajo, para no destruir las features preaprendidas)

**Loss Function / Función de Pérdida**
Mide qué tan equivocado está el modelo en sus predicciones. El objetivo del entrenamiento es minimizarla. En este proyecto se usa **Sparse Categorical Crossentropy**, ideal para clasificación multiclase cuando las etiquetas son enteros (0, 1, 2...) en lugar de one-hot encoding.

**EarlyStopping**
Callback que detiene el entrenamiento automáticamente cuando una métrica de validación deja de mejorar durante `patience` epochs consecutivos. En este proyecto:
- `monitor="val_accuracy"` → observa la accuracy en validación
- `mode="max"` → busca el máximo (mayor accuracy = mejor)
- `patience=5` → espera 5 epochs sin mejora antes de parar
- `restore_best_weights=True` → al finalizar, restaura los pesos del mejor epoch

**Overfitting**
El modelo "memoriza" el dataset de entrenamiento y pierde capacidad de generalizar a datos nuevos. Se manifiesta cuando la `accuracy` de entrenamiento sube pero la de validación se estanca o baja. Se combate con Dropout, Data Augmentation y EarlyStopping.

**Underfitting**
El modelo es demasiado simple o fue entrenado poco tiempo, y no aprende ni los patrones del entrenamiento. Se manifiesta con accuracy baja tanto en train como en validación.

---

### Transfer Learning

**Transfer Learning**
Técnica donde se reutiliza un modelo preentrenado en un problema grande (ImageNet: 14M imágenes, 1000 clases) como punto de partida para un problema más pequeño y específico. El conocimiento aprendido (detectar bordes, texturas, formas) es transferible entre dominios visuales.

**Feature Extraction**
Modalidad de Transfer Learning donde se congelan TODOS los pesos del modelo base y solo se entrenan las nuevas capas añadidas encima. Es la estrategia usada en este proyecto (`NUM_TRAINABLE_LAYERS = 0`).

**Fine-tuning**
Modalidad de Transfer Learning donde, después de la fase de Feature Extraction, se "descongela" una parte de las últimas capas del modelo base y se re-entrena todo el conjunto con un learning rate muy bajo. No fue aplicado en este TP pero está preparado en el código (`NUM_TRAINABLE_LAYERS > 0`).

**ImageNet**
Dataset con más de 14 millones de imágenes etiquetadas en 1000 categorías. Es el benchmark estándar para comparar modelos de visión por computadora. Los modelos preentrenados en ImageNet aprenden representaciones visuales muy ricas y generalizables.

**`preprocess_input`**
Función obligatoria cuando se usan modelos preentrenados. Cada arquitectura fue entrenada con una normalización específica de sus imágenes, y hay que reproducir exactamente la misma al hacer inferencia o fine-tuning:
- **VGG16:** Centra los píxeles restando la media de ImageNet por canal (RGB). Resultado: valores alrededor de [-100, 150].
- **MobileNetV2:** Normaliza a rango `[-1, 1]` dividiendo y desplazando.
- **Custom CNN:** Usa `Rescaling(1./255)` integrada en la arquitectura, convirtiendo píxeles [0, 255] a [0, 1].

---

### Evaluación

**Accuracy**
Porcentaje de predicciones correctas sobre el total. `Accuracy = (Correctas / Total) * 100`. Es una métrica simple y fácil de interpretar, pero puede ser engañosa en datasets desbalanceados.

**Confusion Matrix (Matriz de Confusión)**
Tabla que muestra, para cada clase real, cuántas veces el modelo la predijo como cada clase posible. Permite identificar exactamente qué clases confunde el modelo entre sí.

**Val Accuracy / Validation Accuracy**
Accuracy medida sobre el conjunto de validación al final de cada epoch. Es el indicador real de qué tan bien generaliza el modelo, ya que estos datos no se usaron para actualizar los pesos.

**Test Accuracy**
Accuracy medida sobre el conjunto de test, que es completamente independiente del entrenamiento y la validación. Representa el rendimiento real del modelo en producción.

**`class_mode="sparse"`**
Parámetro del `ImageDataGenerator` que indica que las etiquetas de las clases son enteros (0, 1, 2, 3, 4) en lugar de vectores one-hot. Compatible con la función de pérdida `sparse_categorical_crossentropy`.

---

### Arquitecturas Específicas

**VGG16**
Arquitectura propuesta por el Visual Geometry Group (Oxford) en 2014. Su principio de diseño es la uniformidad: usa solo filtros 3x3 apilados. Tiene 16 capas con pesos (13 conv + 3 FC). Es profunda pero simple de entender.

**MobileNetV2**
Arquitectura de Google diseñada para eficiencia. Sus dos innovaciones clave son:
1. **Depthwise Separable Convolutions:** Separa la convolución espacial (por canal) de la combinación de canales, reduciendo los FLOPs (operaciones de punto flotante) drásticamente.
2. **Inverted Residual Blocks:** Expande el número de canales, aplica la convolución depthwise, y vuelve a comprimir. Las conexiones residuales se aplican cuando la dimensión de entrada y salida son iguales.

**`include_top=False`**
Parámetro al cargar VGG16 o MobileNetV2 que indica que se excluyen las capas densas originales de clasificación (las que clasifican en las 1000 clases de ImageNet). Se conserva solo el "cuerpo" extractor de features.

**`layer.trainable = False`**
Congela los pesos de una capa para que no se actualicen durante el entrenamiento. En Feature Extraction, se congela todo el modelo base y solo se entrenan las capas nuevas que añadimos encima.

---

## Requisitos y Setup

**Dependencias:**
```
tensorflow>=2.21.0
pillow
matplotlib
scipy
scikit-learn
```

**Instalación:**
```bash
pip install tensorflow pillow matplotlib scipy scikit-learn
```

**Versión de Python:** 3.12+

**Para ejecutar los notebooks:**
1. Activar el entorno virtual: `.venv/Scripts/Activate.ps1` (Windows)
2. Abrir los notebooks en orden: primero los de entrenamiento (`CM`, `VGG16`, `MBNet`), luego el de comparación.
3. Los modelos `.keras` se guardan automáticamente en el directorio raíz.
