### **Ejercicios 9 CC0C2**

### 1. Cuaderno: **Atención y codificación posicional**

#### Ejercicio 1: Diccionario robusto con `UNK`

1. En la parte donde construyes el vocabulario y la función `translate(sentence)`:

   * Agrega un token especial `<unk>` al vocabulario.
   * Ajusta la codificación (one-hot o índices) para que cualquier palabra que **no** esté en el diccionario vaya a `<unk>`.
2. Prueba con una frase que mezcle:

   * Palabras conocidas.
   * Una o dos palabras inventadas.
3. : Resultados:

   * Muestra la salida de `translate(...)`.
   * Escribe en 3-4 líneas qué ventaja tiene usar `<unk>` frente a lanzar un error.

#### Ejercicio 2: Softmax y "temperatura"

1. Localiza la parte donde se calculan similitudes y se aplica `softmax` (los scores de atención).

2. Envuelve el softmax en una función:

   ```python
   def softmax_con_temperatura(scores, tau=1.0):
       return torch.softmax(scores / tau, dim=-1)
   ```

3. Presentación:

   * Compara las distribuciones de atención para `tau = 0.5`, `tau = 1.0` y `tau = 2.0`.
   * Imprime los vectores de atención para un ejemplo concreto.

4. Preguntas cortas:

   * ¿Con qué `tau` la atención se vuelve más "concentrada"?
   * ¿Con qué `tau` la atención es más "difusa"?

#### Ejercicio 3: Self-attention manual en un caso pequeño

1. En la sección donde se define la clase `Head` o se ilustran matrices `Q, K, V`:

   * Fija una secuencia muy pequeña, por ejemplo 3 tokens con `n_embd = 2`.
   * Define a mano (en código) `Q`, `K` y `V` como pequeños tensores de tamaño `(3, 2)`.
2. Resultados:

   * Calcula en código `scores = Q @ K.T / sqrt(d_k)`.
   * Aplica softmax y obtenga los pesos de atención.
   * Calcula la salida `att = softmax(scores) @ V`.
3. Luego, escribe en 2-3 líneas:

   * ¿Qué token termina "prestando" más información a los otros?
   * ¿Qué pasaría si uno de los vectores de `K` fuera casi todo ceros?

#### Ejercicio 4: Inspeccionar codificación posicional

1. En la sección de **codificación posicional**, se pide:

   * Generar las posiciones `pos = 0..20` para un `d_model` pequeño (por ejemplo 8).
   * Imprimir las primeras 5 filas de la matriz de codificación posicional.
2. Se pide un pequeño gráfico (aunque sea básico) que al menos compare:

   * Las posiciones 0, 1, 2 en la primera dimensión.
3. Pregunta:

   * ¿Por qué las funciones seno y coseno ayudan a distinguir posiciones lejanas?
   * ¿Qué pasaría si usáramos solo un contador 0,1,2,... en vez de estos patrones?

### 2. Cuaderno: **Transformer para clasificación**

#### Ejercicio 5: Longitud de secuencia y padding

1. En la sección de **cero padding…datos**, debes:

   * Imprimir la longitud (número de tokens) de 3 oraciones antes y después de aplicar padding.
2. Después:

   * Que compruebe que todas las secuencias en un batch tienen la misma longitud.
3. Preguntas:

   * ¿Por qué es necesario el padding cuando usamos `DataLoader` y Transformers?
   * ¿Qué podría salir mal si no "ignoramos" los `PAD` en la atención?

*(Si ya tienen máscara de padding implementada, se pide que impriman la máscara para un batch y marquen cuáles posiciones son `PAD`.)*


#### Ejercicio 6: Cambiar la dimensión de `d_`


1. Identifica los lugares donde se define `d_` o `emb_size`.
2. Se pide que:

   * Ejecutes el o con un `d_` pequeño (por ejemplo 16).
   * Luego con un `d_` mayor (por ejemplo 64).
3. Que registres:

   * Exactamente cuántos parámetros tiene el modelo en cada caso (usando `sum(p.numel() for p in modelo.parameters())`).
4. Preguntas cortas:

   * ¿Cómo cambia el tiempo de entrenamiento por época?
   * ¿Qué intuición tienen sobre el trade-off entre tamaño del modelo y capacidad de generalización?

#### Ejercicio 7: Probar overfitting en un subconjunto


1. Toma solo, por ejemplo, 200 muestras del conjunto de entrenamiento.
2. Debes realizar:

   * Entrenar el modelo **solo** en esas 200 muestras.
   * Observar la pérdida de entrenamiento y la exactitud sobre esas mismas muestras.
3. Preguntas:

   * ¿Es capaz el modelo de llegar a >98% de accuracy en ese mini-dataset?
   * ¿Qué significa eso en términos de overfitting?
   * ¿Cómo se compara esa accuracy con la del conjunto de validación/test?

### 3. Cuaderno: **Transformer para traducción**

#### Ejercicio 8: Explorar la codificación posicional en traducción


1. En la clase `PositionalEncoding` de este cuaderno:

   * Se pide que definan una frase simple en el idioma origen, por ejemplo: `"I like cats"`.
   * Obtener sus embeddings + codificación posicional (después de sumar `embedding + positional_encoding`).
2. Que impriman:

   * El vector resultante para la primera palabra y para la última.
3. Pregunta:

   * Si permutamos las palabras (por ejemplo `"cats like I"`), ¿qué cambia exactamente en los vectores de entrada al Transformer?


#### Ejercicio 9: Decodificación paso a paso (greedy)

1. Usando la parte de **Inferencia** del Transformer de traducción:

   * Se pide implementar una función `greedy_decode_step_by_step` que:

     * Tome la frase origen.
     * Vaya generando un token a la vez hasta llegar a `<eos>` o un máximo de longitud.
2. Debes:

   * Imprimir la secuencia parcial en cada paso (por ejemplo: `<bos>`, `<bos> I`, `<bos> I like`, etc.,  en el idioma destino).
3. Pregunta:

   * ¿Por qué el decoder solo puede ver los tokens previos y no los futuros?
   * ¿En qué se diferencia esto del entrenamiento con *teacher forcing*?

#### Ejercicio 10: Máscara de padding en el encoder


1. Localiza el código donde se construye la máscara de padding (`src_key_padding_mask`, `tgt_key_padding_mask`, etc.).
2.  Se pide:

   * Imprimir la máscara para un batch concreto (por ejemplo, las primeras 4 oraciones).
3. Preguntas:

   * ¿Qué valores tiene la máscara en las posiciones de `PAD`? (True/False o 0/1).
   * ¿Qué pasaría si **no** pasáramos esta máscara al encoder? Describe un posible error o comportamiento raro.

#### Ejercicio 11 (opcional): Inspeccionar pesos de atención

1. Usando el Transformer de traducción:

   * Escoge una frase simple y corre una traducción.
   * Modifica el código para recuperar los pesos de atención de **una capa** del encoder o del decoder (por ejemplo usando `nn.MultiheadAttention` con `need_weights=True`).
2. Se pide:

   * Imprimir los pesos de atención para una cabecera (por ejemplo, forma `[longitud, longitud]`).
   * Que marquen manualmente un token origen y vean a qué otros tokens presta más atención.
3. Pregunta:

   * ¿Coinciden esos pesos con la intuición lingüística (por ejemplo, sujeto <-> verbo, verbo <-> objeto)?

