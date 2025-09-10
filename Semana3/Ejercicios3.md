### Ejercicios 3 CC0C2

### **Ejercicios teóricos**

> Sin código. Para discusión en clase y resolución en grupos.

#### Ejercicio 1: Mapa del pipeline de texto -> LLM/FM

Dibuja (en papel o pizarra) un diagrama que conecte: normalización (Unicode/case), lematización/stemming, tokenización (WordPiece/BPE/Unigram/BBPE), inserción de tokens especiales, *padding* y máscaras de atención. 
Anota en cada flecha qué señales podrían perderse si se aplica esa etapa de forma agresiva.

#### Ejercicio 2: Política de normalización por dominio

Elige un dominio (legal, clínico, redes sociales, logs). Propón 8-10 reglas concretas (minúsculas, NFKC, dígitos, unidades, URLs/emojis).  
Para cada regla: beneficio, riesgo y ejemplo antes/después. Define 5 "casos límite" que tu política debe preservar.

#### Ejercicio 3:  Modelos *cased* vs *uncased*

Para NER multilingüe, contrasta usar un checkpoint *cased* vs *uncased*. Lista 5 ventajas y 5 desventajas operativas (coste en tokens, sensibilidad a acrónimos, impacto en mayúsculas iniciales en español). 
Concluye con una recomendación por dominio.

#### Ejercicio 4:  Selección de tokenizador por escenario

Para cada escenario, elige **uno**: WordPiece, BPE por caracteres, **Byte-Level BPE**, Unigram/SentencePiece. Justifica en 3-4 líneas.

* Chats con emojis, código y mezclas es/en.
* Corpus jurídico en español con citas y tablas.
* Logs técnicos de microservicios.
* Noticias multilingües con nombres propios y topónimos.


#### Ejercicio 5:  BPE "en la cabeza"

Con el mini-corpus `low lowest lower lowering new newer`, identifica la **primera** fusión razonable y explica por qué. 
Describe cómo cambiaría tu elección si trabajas a **nivel de bytes** (UTF-8) en vez de caracteres.

#### Ejercicio 6:  BPC vs BPB, qué medir y cuándo

Asigna **BPC** o **BPB** como métrica preferente y explica en 2-3 líneas:

* Texto monolingüe limpio;
* Mezcla de scripts + emojis;
* CSV con campos técnicos.
  Indica cómo se relaciona con *tokens/caracter* para un tokenizador dado.

#### Ejercicio 7: Unicode y acentos

Compara NFD vs NFKC para cadenas con acentos y símbolos compuestos (ej.: "café", "①", "Ω"). Explica cuándo **no** conviene eliminar diacríticos si tu tarea es NER en español.


#### Ejercicio 8:  Lematización vs *stemming* en lenguas flexivas

Para español y quechua (o una lengua aglutinante de tu elección), argumenta si aplicarías lematización, *stemming* o ninguna antes de un LLM. 
Cita efectos en cobertura de subpalabras y en *prompt budgeting*.


#### Ejercicio 9:  Elegir la distancia adecuada

Asocia la métrica más apropiada y justifica: Levenshtein, Damerau, Jaro-Winkler, "sin sustitución".

* "abc" <-> "acb";
* "color" <-> "colour";
* "server\_01" <-> "server-01";
* "OAuth2.0" <-> "0Auth2O".


#### Ejercicio 10:  Invariantes y QA del tokenizador

Escribe 8 invariantes (determinismo, preservación de marcadores, límites en variación de tokens/char, idempotencia de normalización). 
Diseña un *canary set* de 150 entradas con cobertura de casos duros.

#### Ejercicio 11:  Unigram vs BPE

Explica diferencias de entrenamiento (criterio probabilístico vs fusiones por frecuencia) y cuándo preferirías Unigram/SentencePiece frente a BPE/BBPE en un FM multilingüe.

### **Ejercicios de codificación**

> Basados en los cuadernos: **Normalización/Lematización/Distancias**, **Tokenización**, **BPE/BPC/BPB**.

#### Ejercicio 12:  Pipeline de normalización reproducible

**Tareas:**

* Diseña funciones para: *case folding*, normalización Unicode (NFKC), compactación de espacios, saneo de caracteres de control, y un "diccionario de sustituciones" (ej.: variantes de comillas, unidades, acrónimos).
* Aplica el pipeline a un conjunto de textos heterogéneo (idiomas mixtos, emojis, URLs).
* Reporta: porcentaje de caracteres modificados, tipos de sustituciones, ejemplos antes/después.
  **Criterios de éxito:** idempotencia (aplicar dos veces no cambia el resultado), métricas agregadas y 10 casos límite documentados.

#### Ejercicio 13:  Lematización vs *stemming* (wrappers multilengua)

**Tareas:**

* Implementa un "wrapper" que soporte español e inglés (elige bibliotecas adecuadas para cada idioma).
* Ejecuta sobre un *mini-corpus* etiquetado con POS y calcula reducción del vocabulario (número de formas -> lemas/stems).
* Reporta errores típicos y cuándo el *stemming* rompe palabras válidas.
  **Criterios de éxito:** tabla con tasas de reducción, análisis por categoría gramatical y 8 ejemplos comentados.


#### Ejercicio 14:  Distancias de edición con *backtrace* (Levenshtein, Damerau, sin sustitución)

**Tareas:**

* Implementa las tres distancias con matrices de DP y *backtrace*.
* Evalúa pares representativos ("abc<->acb", "color<->colour", "login<->logn", IDs con guiones).
* Compara tiempos y memoria en longitudes crecientes.
  **Criterios de éxito:** tabla "par->distancia/operaciones", complejidad empírica (gráfico o resumen) y discusión de la métrica adecuada por caso.

#### Ejercicio 15:  Alineación tipo Viterbi en log-probabilidades

**Tareas:**

* Define parámetros en log-espacio y realiza el *backtrace* del camino de máxima probabilidad.
* Sensibiliza el resultado variando los pesos (tres configuraciones).
* Aplica a títulos con *typos* y a secuencias técnico-semánticas (p. ej., "server\_01" vs "server-01").
  **Criterios de éxito:** matriz de puntajes, camino alineado y análisis de estabilidad ante cambios de hiperparámetros.


#### Ejercicio 16:  Tokenización clásica con `torchtext`: vocabulario, especiales y *padding*

**Tareas:**

* Crea vocabulario desde iterador de tokens con especiales `<unk>`, `<bos>`, `<eos>`, `<pad>` y configura índice por defecto para OOV.
* Genera lotes con *padding* uniforme y máscara de atención.
* Mide: tokens/char, proporción de *padding* por lote y distribución de longitudes.
  **Criterios de éxito:** tabla de métricas por lote y verificación de que OOV mapea a `<unk>`.


#### Ejercicio 17:  Tokenizadores de *checkpoints* (BERT/WordPiece vs SentencePiece/Unigram)

**Tareas:**

* Usa tokenizadores "nativos" de dos modelos distintos sobre el mismo corpus multilingüe.
* Reporta: tokens/char por idioma, top-20 subpalabras más frecuentes, latencia media de *encode/decode*.
* Analiza impacto en *prompt budgeting*.
  **Criterios de éxito:** tablas comparativas y breve recomendación técnica por dominio.


#### Ejercicio 18:  BPE didáctico desde cero (caracteres)

**Tareas:**

* Entrena BPE sobre un corpus pequeño partiendo de caracteres.
* Registra las *k* fusiones más frecuentes (k=20 sugerido) y tokeniza palabras nuevas.
* Mide reducción de longitud media en tokens vs baseline por caracteres.
  **Criterios de éxito:** lista ordenada de *merges*, tabla "palabra->tokens" y gráfico/tabla de reducción (%) por iteración.

#### Ejercicio 19:  Byte-Level BPE (BBPE) vs BPE por caracteres

**Tareas:**

* Adapta tu BPE para operar en bytes y compáralo con la versión por caracteres usando textos con emojis, scripts mixtos y ruido de *scraping*.
* Reporta: OOV (esperado 0 en BBPE), tokens/char y fallos de normalización.
  **Criterios de éxito:** tabla comparativa y análisis de casos donde BBPE evita pérdidas.


#### Ejercicio 20:  Métricas BPC/BPB y su lectura operativa

**Tareas:**

* Calcula BPC y BPB en un *corpus de prueba* con un estimador simple (p. ej., modelo ligero o proxy de compresión) y relaciona con tokens/char de dos tokenizadores.
* Segmenta por dominio (redes, legal, logs) y compara.
  **Criterios de éxito:** tabla por dominio (BPC, BPB, tokens/char) y explicación de correlaciones observadas.



#### Entrega sugerida

* Un archivo Markdown con: diseño, decisiones, tablas/figuras y análisis por ejercicio.
* Carpeta con datos de prueba y *logs* de trazabilidad (anonimizados).
