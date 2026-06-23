# Anexo Técnico: Glosario, Conceptos y Justificaciones

Este documento está diseñado para explicar "en criollo" (de forma sencilla, didáctica y con analogías) cómo funciona cada engranaje, función y concepto matemático de este proyecto de Inteligencia Artificial aplicado a la clasificación de peces.

---

## 1. El Ciclo de Vida de los Datos: La Escuela de la IA

Antes de entrenar una red, tenemos que separar nuestras imágenes. En IA, dividimos los datos simulando un ciclo escolar:

- **Dataset**: Es la biblioteca completa de imágenes que recolectamos.
- **Entrenamiento (Train)**: Son los **"libros de estudio"**. La red los lee una y otra vez todos los días para aprender cómo es un pez.
- **Validación (Validation)**: Son las **"pruebas sorpresa semanales"**. Se toman al final de cada época con fotos que no estaban en el libro. Sirven para que el profesor (nosotros) se fije si la red está aprendiendo de verdad o solo está repitiendo de memoria.
- **Prueba (Test)**: Es el **"examen final definitivo"**. Se toma al final del año escolar usando fotos que la red **jamás vio en su vida**. Es la única métrica real y honesta de qué tan inteligente quedó el modelo.

### Funciones del Código:
- **`flow_from_directory`**: Es una función de Keras que te salva la vida. Vos solo le apuntás a la carpeta principal (ej: `train/`), y la función entra sola, mira las subcarpetas (ej: `Pez_A/`, `Pez_B/`), cuenta las fotos, y automáticamente le pone la "etiqueta" correcta a cada foto sin que tengas que programar nada.

### `ImageDataGenerator` y el Aumento de Datos
- **`ImageDataGenerator`**: Imaginate que tenés fotos pesadísimas. Si las cargás todas juntas, la computadora colapsa (se queda sin RAM). Esta función es como un **"mozo de restaurante"**: va a la cocina (el disco duro), busca una bandeja con 32 fotos (un lote o *batch*), se las sirve a la red neuronal, y luego vuelve por otras 32. Así, la memoria siempre está libre.
- **Data Augmentation (Aumento de Datos)**: Es el gran truco anti-memoria.
  - *¿Cómo funciona?* El "mozo" agarra una foto original. **"Al vuelo" (en tiempo real)**, justo antes de dársela a la red, le aplica una alteración matemática aleatoria: la gira, le hace zoom o la espeja.
  - *¿Se aplica una sola vez?* No, **se aplica de forma aleatoria en cada Época**. Es decir, si entrenás 50 épocas, la red verá 50 versiones ligeramente distintas de esa foto.
  - *Justificación (Overfitting)*: Esto evita el **Sobreajuste**.

---

## 2. Conceptos Clave del Modelo

- **Parámetros del modelo (Weights)**: Imaginate una radio antigua enorme con millones de perillas de sintonía milimétricas. Al principio, todas las perillas están puestas al azar y solo se escucha ruido estático. En cada época, la red mueve sutilmente miles de perillas a la vez para "sintonizar" mejor la señal visual del pez correcto.
- **Overfitting (Sobreajuste)**: Es literalmente **"estudiar de memoria"**. Pasa cuando el modelo tiene tantos parámetros que se memoriza la mancha de agua o el fondo exacto de la foto del "libro de texto" en vez de entender la forma del pez. Si el modelo sufre de overfitting, en las pruebas semanales saca un 10 perfecto, pero en el examen final (donde la luz o el fondo cambia un poquito) saca un 2 catastrófico. El *Data Augmentation* y el *Dropout* son las curas para esto.

---

## 3. Componentes de la Arquitectura CNN

Una Red Neuronal Convolucional (CNN) trata de imitar el ojo humano: empieza detectando cosas simples (líneas) y termina detectando cosas complejas (un pez). Para armar esto usamos funciones estructurales:
- **`Sequential` / `Model`**: Es el andamio o envase vacío donde vamos apilando nuestras capas una encima de otra, como si hiciéramos un sándwich.

### Las Capas:
- **`Conv2D` (Convolución) y Kernels**: Imaginate que un **kernel** (filtro) es una pequeña lupa de 3x3 que va escaneando la foto. Una lupa busca bordes horizontales, otra texturas de escamas. Capa tras capa, estas lupas se vuelven más inteligentes para buscar aletas o colas.
- **`padding='same'`**: Cuando la lupa escanea los bordes exactos, la imagen suele achicarse. Esta función le pone un marco falso de píxeles negros a la foto original para no perder información de las orillas.
- **`MaxPooling2D`**: Imaginate que **entornás los ojos** para ver una imagen borrosa pero captando lo más brillante e importante. Agarra cuadraditos de la imagen y se queda solo con el número más alto. Achica la imagen a la mitad, acelera todo y ayuda a encontrar el pez aunque esté movido hacia un costado (invariancia espacial).
- **`Flatten` vs `GlobalAveragePooling2D`**: 
  - Al final de las lupas, la red tiene que tomar una decisión con matrices 2D, pero necesita una lista plana de números.
  - `Flatten` aplasta toda la matriz a lo bruto en una fila kilométrica.
  - `GlobalAveragePooling2D` es más inteligente: calcula un promedio matemático de cada filtro. Achica los datos de forma masiva y hace que la red final sea un peso pluma rapidísimo.
- **`Dense`**: La capa de red clásica. Es el "cerebro lógico" final. Analiza las pistas de las lupas y decide la especie.
- **`Dropout`**: Es un concepto de supervivencia brutal. En cada época, le decimos al 30% de las neuronas que **"se vayan a dormir"**. En un grupo, si uno hace todo el trabajo, el resto no aprende. Como acá algunas neuronas faltan al azar, las despiertas están obligadas a aprender bien y a no depender de sus compañeras. Resultado: un equipo súper robusto e independiente.

---

## 4. Activaciones, Entrenamiento y Optimización

Las **funciones de activación** son interruptores matemáticos que deciden qué neurona se enciende.
- **`ReLU`**: El interruptor estrella del medio. Si el número es negativo, lo apaga a cero. Si es positivo, lo deja pasar. Esto permite que la IA aprenda contornos y curvas, no solo líneas rectas.
- **`Softmax`**: Exclusivo para la última capa. Agarra puntajes raros y los convierte en porcentajes ordenados perfectos. Te dice: *"Estoy 95% segura de que es un Pez A, y 5% de que es un Pez B"*.

### Ensamblando y Entrenando (`compile` y `fit`)
- **`compile`**: Es preparar la receta. Acá unimos el optimizador y la función de pérdida antes de empezar.
- **`fit`**: Es la orden de *"Empieza el año escolar"*. Le dice a la IA que empiece a leer el dataset.
- **Optimizador (`Adam`)**: Es la brújula. Imaginate al modelo con los ojos vendados en una montaña intentando bajar al valle (el error cero). Adam mueve las "perillas de sintonía" de manera inteligente: da pasos largos si la bajada es obvia, y pasos milimétricos si ya casi llega al piso.
- **Función de Pérdida (`Sparse_Categorical_Crossentropy`)**: Es el "castigo matemático". Mide qué tan feo se equivocó la IA. Se usa `Sparse` porque es más eficiente cuando las etiquetas son números simples (0, 1, 2) en vez de formatos gigantes.

### Los Vigilantes (`Callbacks`)
Son funciones que vigilan a la IA mientras vos tomás café.
- **`EarlyStopping`**: Si pasan 5 épocas y la red no mejora su nota en las "pruebas sorpresa", este vigilante frena todo y te devuelve la mejor versión. Evita que la IA empiece a estudiar de memoria (overfitting).
- **`ReduceLROnPlateau`**: Si el modelo está atascado, le dice a la brújula Adam: *"Achicá el tamaño de tus pasos, vayamos en puntas de pie a ver si encontramos el fondo"*.

---

## 5. Arquitecturas Modernas: El Arte de Tomar Prestado

1. **Feature Extraction y Transfer Learning**: Entrenar una red desde cero para que entienda qué es una curva o una sombra lleva meses. Usar VGG16 o MobileNetV2 es usar el **"cerebro prestado"** de los científicos de Google y Oxford (que ya vieron 14 millones de fotos). Usamos sus **ojos súper sofisticados** que ya entienden la realidad; solo le enchufamos una nueva "boca" para que diga nombres de 5 peces.
2. **Congelamiento de Capas (`trainable = False`)**: Al descargar ese cerebro experto, le ponemos un **candado de seguridad** a casi todo. Si lo dejáramos desbloqueado, los errores garrafales que comete nuestra red nueva en las primeras épocas viajarían hacia atrás, destruyendo para siempre la sabiduría milenaria que Google tardó meses en enseñarle.
3. **Convoluciones Separables por Profundidad (El secreto de MobileNetV2)**: ¿Por qué VGG16 es gigantesca y MobileNetV2 es diminuta y veloz? Imaginate pintar una pared entera con un pincel chiquito: demorás semanas (así funciona la convolución pesada de VGG16, procesando formas y canales de colores todos al mismo tiempo). 
   - MobileNetV2 separó el trabajo inteligentemente: primero pasa un **rodillo gigante** sin pintura solo para detectar la forma espacial del pez, y luego en un segundo paso súper liviano, pasa un **pincel rápido** solo para mezclar los canales de colores (RGB). Al separar el esfuerzo en dos, es ridículamente más rápida y eficiente.

---

## 6. Funciones de Evaluación Final (`evaluate` y Métricas)

- **`evaluate`**: Es la función que toma el examen final definitivo. 
- **`predict`**: Es cuando ya se graduó y le damos una foto cualquiera sin respuesta para ver qué nos dice.

### Midiendo el Éxito (Classification Report)
- **Accuracy (Exactitud)**: De cada 100 fotos que rindió, ¿cuántas acertó en total?.
- **Precision (Precisión)**: Mide a los "Falsos Positivos" (mentirosos). De todas las veces que la IA gritó *"¡Pez A!"*, ¿cuántas era cierto?
- **Recall (Sensibilidad)**: Mide a los "Falsos Negativos" (distraídos). De todos los Pez A que estaban nadando en la foto, ¿cuántos logró encontrar la IA?
- **F1-Score**: El promedio armónico que te castiga si descuidás el Precision para favorecer el Recall.
- **Support**: Es simplemente la cantidad de "preguntas en el examen final" de cada especie (ej: 100 fotos de Pez A).
- **Macro Average**: Es el promedio "democrático". Le da exactamente el mismo valor a todas las especies. Si le fue horrible en el Tiburón pero genial en la Mojarrita, el Macro los promedia al 50%.
- **Weighted Average**: Es el promedio "por volumen de alumnos". Si había 10 fotos de Tiburón y 900 de Mojarrita, la nota de la Mojarrita pesa 90 veces más en el promedio final.

### Curva ROC, AUC y One-vs-Rest
- **Curva ROC**: Grafica el comportamiento del modelo si le cambiamos el umbral de duda (qué tan seguro tiene que estar para clasificar algo).
- **AUC (Área Bajo la Curva)**: Es el puntaje de la **claridad absoluta o la capacidad de no dudar**. Un AUC perfecto de 1.0 significa que la IA no se confunde jamás; separa al Pez A del Pez B trazando una línea mágica e irrompible entre ellos.
- **One-vs-Rest (Uno contra la Pandilla)**: La matemática del AUC y ROC fue inventada en la Segunda Guerra Mundial para detectar "Radar o Pájaro" (2 opciones). Como tenemos 5 peces, para poder usar esta matemática hermosa hacemos un "Torneo solitario": agarramos al Pez A y le preguntamos al modelo: *"¿Esto es el Pez A o es cualquiera del resto combinados?"*. Lo medimos, lo anotamos, y pasamos al Pez B contra la pandilla. Así graficamos la curva para cada pez por separado.
