### Tokenización para FM y LLM, BPE y variantes, normalización, lematización, segmentación, distancias de edición y Viterbi


#### 1) Por qué la tokenización es el "puente" entre texto y tensores

En modelos fundacionales (FM) y modelos de lenguaje grandes (LLM), la tokenización transforma texto crudo en unidades discretas que el modelo puede mapear  a identificadores e incrustaciones. 
Esta capa es más que un trámite: condiciona el largo efectivo de secuencia, el costo de entrenamiento e inferencia, la robustez ante dominios variados y la cobertura frente a idiomas, emojis y símbolos técnicos. 

Una buena política de tokenización equilibra tres tensiones prácticas: cobertura abierta para evitar fuera de vocabulario, compacidad para no disparar el cómputo por secuencias larguísimas y estabilidad para mantener consistencia entre entrenamiento e inferencia.

La tokenización moderna suele combinar dos etapas. Primero, una pre-tokenización *top-down* que protege patrones enteros con reglas, por ejemplo, direcciones de correo, URLs, cantidades monetarias o abreviaturas. 
Después, una subtokenización *bottom-up* que descompone cada fragmento en subunidades óptimas, casi siempre con variantes de BPE u otros métodos  de subpalabras. 

Esta combinación captura regularidades lingüísticas sin renunciar a una cobertura universal.

#### 2) Top-down vs bottom-up: cómo se complementan

Top-down significa partir desde la estructura del texto para preservar "trozos" que deben permanecer íntegros. 
En un enunciado como "That U.S.A. poster-print costs $12.40... Email me@example.com!", una estrategia *top-down* extrae tokens como U.S.A., poster-print, $12.40 y me@example.com para evitar que la etapa posterior rompa entidades útiles. 

Esto reduce errores comunes como separar el signo de dólar de la cantidad o descomponer una abreviatura en letras sueltas. 

En LLM, esta capa ayuda a estabilizar entradas con signos de puntuación, números y formatos técnicos, lo que mejora la coherencia semántica que más tarde el modelo verá como secuencias de IDs.

*Bottom-up*, en cambio, parte de unidades pequeñas y aprende a fusionarlas si son frecuentes.
Aquí entra **BPE (Byte-pair encoding)**: un algoritmo de fusiones que induce un vocabulario de subpalabras a partir de un corpus. En producción, el ranking de fusiones se aplica de forma codiciosa a cadenas nuevas. 
Esta idea captura morfemas comunes y repeticiones ortográficas, disminuye la tasa de tokens por palabra habitual y mantiene una vía de escape para palabras raras, nombres propios o neologismos que se descomponen en piezas más cortas sin quedar fuera de vocabulario.

La clave práctica es que *top-down* y *bottom-up* no compiten, sino que se encadenan: primero se preservan unidades críticas con reglas, luego se subtokeniza  con BPE o un método afín. 

Esta arquitectura produce entradas más estables, con menos sorpresas para el modelo y métricas de eficiencia mejores.

#### 3) BPE clásico: qué es y cómo funciona

**Byte-Pair Encoding (BPE)** es un algoritmo bottom-up que aprende fusiones de pares adyacentes. El flujo clásico es sencillo y efectivo:

- Representar cada palabra como una secuencia de símbolos iniciales, típicamente caracteres.
- Contar frecuencias de pares adyacentes en todo el corpus.
- Fusionar el par más frecuente creando un nuevo símbolo de subpalabra.
- Reemplazar y repetir hasta alcanzar un número objetivo de fusiones, que define el tamaño del vocabulario.
- En inferencia, aplicar el ranking de fusiones de forma codiciosa sobre texto nuevo.

Este ciclo compacta secuencias repetidas y morfemas. En un ejemplo canónico con el corpus "low, lowest, new, wider, lowering", las primeras fusiones 
tienden a crear unidades como "lo" y "low", y más adelante "er" o "est", que permiten tokenizar "lower" en "low" + "er" y "lowest" en algo  como "low" + "we" + "st", según frecuencias y trayectoria de fusiones. 

El resultado es un vocabulario de subpalabras que reduce el número promedio de tokens por palabra sin perder la posibilidad de descomponer términos raros.

En español, BPE puede capturar piezas como "ción" o "mente" si el corpus las respalda. Cuando una secuencia con acentos no aparece de forma consistente, el algoritmo cae a unidades más pequeñas y sigue cubriendo el texto de manera determinista.
Marcadores como un prefijo especial para espacio o un delimitador de fin de palabra ayudan a mantener límites coherentes entre palabras.

**Ventajas prácticas del BPE clásico:**

- Eficiencia: menos tokens por palabra típica, lo que permite más contexto con el mismo presupuesto.
- Robustez razonable: palabras desconocidas se descomponen en trozos vistos, evitando el símbolo desconocido.
- Simplicidad y velocidad: el ranking de fusiones es estático y su aplicación codiciosa es rápida.

**Limitaciones:**

- Si la base son caracteres Unicode, ciertos símbolos poco frecuentes o combinaciones de bytes para emojis pueden quedar mal segmentados o escapar a la cobertura esperada.
- Idiomas con escritura compleja o cadenas multilenguaje pueden sufrir degradación, sobre todo si el corpus de entrenamiento no los representó bien.

Estas limitaciones motivan variantes más robustas.

#### 4) Byte-level BPE: cobertura universal desde los bytes

Byte-level BPE parte de los bytes crudos del texto en lugar de caracteres Unicode. Dado que cualquier cadena puede codificarse en bytes, esta variante ofrece cobertura universal: no existen fuera de vocabulario, y emojis o secuencias de distintos alfabetos entran de manera natural. 

Con un ejemplo como "mañana 😊", los bytes UTF-8 de la *ñ* y del emoji se tratan como símbolos iniciales. 
Sin fusiones, la tokenización emite cada byte. Con fusiones aprendidas, puede colapsar pares o grupos frecuentes, por ejemplo, los dos bytes quecodifican *ñ* o los cuatro del emoji, y mantener un prefijo especial para el espacio que estabiliza los límites de palabra. 

Esta estrategia garantiza que toda entrada sea tokenizable y que los trozos más comunes de bytes se compacten, logrando un buen balance entre cobertura y longitud de secuencia.

**Limitaciones y su impacto práctico**

Aunque byte-level BPE garantiza cobertura universal, su menor alineación con morfemas complica tareas de análisis humano y depuración. 
En auditorías de salida, rastrear por qué un modelo generó una forma concreta es más difícil cuando los tokens son bytes o grupos de bytes que no se corresponden con unidades lingüísticas interpretables.

Esto afecta las inspecciones de *saliency* (técnicas que muestran qué partes de la entrada influyen más en la salida), la atribución de errores y 
las *safety reviews* (revisiones de seguridad y cumplimiento, por ejemplo para detectar toxicidad o información personal identificable), porque 
los filtros y las reglas suelen operar a nivel de palabra o de subpalabra con sentido lingüístico, en lugar de tokens de bytes.
 
También puede enmascarar sesgos morfológicos (prefijos/sufijos) y dificultar diagnósticos de inyección de prompt o jailbreaks que se esconden en secuencias de símbolos. 

Recomendación: instrumentar auditorías periódicas de subpalabras por dominio (listas de fusiones más frecuentes, proporción de tokens de un solo byte,...etc), definir listas blancas/negras de grafemas críticos y mantener corpora de prueba por dominio para validar que las fusiones preservan patrones 
relevantes sin degradar la interpretabilidad.

**Implementación de auditorías para byte-level BPE**

Para mitigar las limitaciones de byte-level BPE, las auditorías deben ser sistemáticas y enfocadas en métricas prácticas:

- Análisis de fusiones frecuentes: Generar listas de las subpalabras más comunes por dominio ( redes sociales, textos legales,... etc) y evaluar si reflejan patrones lingüísticos relevantes.
- Proporción de tokens de un solo byte: Medir la fracción de tokens que caen a bytes individuales (indicador de baja compresión).
- Drift por símbolo: Monitorear la aparición de nuevos símbolos (emojis, caracteres técnicos, ...etc) en inferencia frente al corpus de entrenamiento.
- Listas blancas/negras: Definir grafemas críticos que deban preservarse o evitarse en fusiones; protegerlos con reglas de pre-tokenización.
- Corpora de prueba: Mantener conjuntos de datos por dominio para evaluar la calidad de las fusiones.
- Herramientas: Usar bibliotecas como Hugging Face Tokenizers para generar reportes de tokenización y analizar métricas clave.

En la práctica, byte-level BPE brilla en dominios heterogéneos: logs, redes sociales, mezclas de idiomas, cadenas con símbolos de moneda o unidades técnicas.
A cambio, los tokens intermedios pueden no alinearse con morfemas lingüísticos, lo que dificulta la interpretabilidad en tareas de análisis humano o debugging.
Además, la calidad de las fusiones depende de un corpus representativo; si ciertos dominios o alfabetos están subrepresentados, las subpalabras pueden ser menos óptimas, aumentando la longitud de secuencia.

#### 5) BPE con byte-fallback: un híbrido pragmático

BPE con byte-fallback combina lo mejor de dos mundos. El vocabulario principal se entrena como en BPE clásico sobre unidades Unicode 
o subpalabras más "lingüísticas". Solo cuando aparece un tramo que no existe en el vocabulario, el tokenizador cae a bytes para ese fragmento específico. 

Así, "pago ₿100" puede tokenizar como subpalabras para "pago" y "100", y emitir bytes únicamente para el símbolo de Bitcoin si ese carácter particular no existe en el vocabulario. 

Esto preserva la compacidad y la interpretabilidad de subpalabras comunes, a la vez que asegura cobertura universal para raros extremos.

En FM y LLM de producción, byte-fallback resulta especialmente útil: se entrena y perfila como un BPE convencional, no se dispara el tamaño del vocabulario, pero se garantizan "salvavidas" ante novedades del mundo real sin necesidad de volver a entrenar el tokenizador.

#### 6) Ejemplos comparativos integrados

Consideremos el texto: "That U.S.A. poster-print costs $12.40... mañana ₿ y 😊".

- Con pre-tokenización top-down, se preservan unidades como U.S.A., poster-print y $12.40.
- Con BPE clásico, cada fragmento se subtokeniza según sus fusiones aprendidas. Símbolos raros como "₿" o el emoji pueden no tener entrada directa y descomponerse en piezas pequeñas o quedar como desconocidos si el esquema no es byte-aware.
- Con byte-level BPE, todo es tokenizable desde el inicio porque parte de bytes y fusiona lo frecuente.
- Con byte-fallback, se utilizan subpalabras "lingüísticas (alineadas con la escritura natural)" y se emiten bytes solo para el símbolo poco común o el emoji, manteniendo el resto en piezas más interpretables.


#### 7) Normalización: coherencia antes de segmentar

La normalización estabiliza el texto antes de cualquier segmentación. Suele incluir *case folding* para reducir variaciones por mayúsculas, normalización *Unicode* (por ejemplo, *NFKC*) para unificar representaciones canónicas, compactación de espacios y sustituciones de signos equivalentes. 

En dominios técnicos conviene estandarizar formatos numéricos y unidades; en redes sociales, definir políticas para emojis, elongaciones o repeticiones de caracteres.

Un punto crítico es no destruir información semántica que el modelo necesita. Por ejemplo, pasar todo a minúsculas puede ser deseable en un  clasificador pequeño, pero en LLM modernos a menudo se conserva el caso porque influye en significado y estilo. 
Con *Unicode*, elegir una forma de normalización estable evita duplicaciones invisibles de tokens. 

La normalización debe ser consistente entre entrenamiento e inferencia y debe documentarse como parte del contrato del tokenizador.El dominio manda. En legal o clínico es común respetar tildes, símbolos de cita y separadores de secciones. 
En logs y monitoreo quizá se prefiera preservar hashes y direcciones, pero colapsar repeticiones triviales de espacios. 

En general, conviene hacer lo mínimo necesario para estabilizar, posponer transformaciones destructivas y medir su impacto en longitud de secuencia y calidad.

#### 8) Lematización: cuándo aporta valor

La lematización mapea formas flexionadas a su lema canónico, por ejemplo, "canté", "cantaba", "cantaré" -> "cantar".  A diferencia del **stemming**, evita recortes agresivos y mantiene palabras válidas. 

En pipelines clásicos de NLP mejora la generalización de modelos n-grama o lineales, y en recuperación de información unifica variantes. 

En LLM grandes no es requisito de entrenamiento, porque el modelo aprende regularidades morfológicas sobre subpalabras. 
Sin embargo, la lematización puede ser valiosa para limpieza y análisis de datasets, búsqueda semántica, indexación o para construir señales de  evaluación que comparan predicciones a nivel de lema. 

También ayuda a reducir **sparsity** cuando se entrenan cabezales especializados (clasificador lineal, CRF, MLP) con pocos datos: al colapsar 
variantes morfológicas en un solo lema, concentras ejemplos en menos tipos, aumentas la densidad por característica, estabilizas las estimaciones de pesos y mejoras calibración y eficiencia muestral. 
Incluso usando embeddings, agrupar por lema reduce la varianza entre formas casi equivalentes y hace más estables las representaciones agregadas.

#### 9) Segmentación de palabras y subpalabras

La segmentación de palabras es trivial en idiomas con espacios, pero crucial en lenguas como chino o tailandés. 
En el enfoque *top-down*, se usan diccionarios y reglas, en el probabilístico, se recurre a **Viterbi** con un modelo de lenguaje de caracteres o palabras para encontrar la segmentación más probable. 

La segmentación de subpalabras con BPE es distinta: no usa **Viterbi**, sino un algoritmo determinista de fusiones. Aun así, pensar en términos probabilísticos 
es útil al fijar políticas de pre-tokenización o al comparar diseños de vocabulario, porque el objetivo final sigue siendo maximizar la probabilidad del texto con el menor costo de tokens. 

En LLM, la combinación de un marcador especial para el espacio y fusiones frecuentes permite que los límites de palabras sean estables. En dominios con mezcla de alfabetos y emojis, variantes byte-aware evitan que el tokenizador "rompa" caracteres compuestos y reducen sorpresas.

#### 10) Distancias de edición: Levenshtein y familia

Las distancias de edición miden cuánta transformación necesita una cadena para convertirse en otra. **Levenshtein** contabiliza inserciones, eliminaciones y sustituciones con costo uno. 

**Damerau-Levenshtein** añade transposiciones adyacentes, y **Jaro-Winkler** pondera prefijos compartidos, útil para nombres propios.

En FM y LLM estas distancias son útiles en varias capas:

- Calidad de datos: detectar duplicados casi idénticos, consolidar variantes ortográficas, auditar drift en capturas de texto.
- Normalización y limpieza: sugerir correcciones de ruido en OCR o tecleo, con umbrales prudentes.
- Métricas de evaluación: a nivel de caracteres para WER en ASR u OCR, o como señal auxiliar cuando el objetivo no es estrictamente semántico.
- Alineación y deduplicación: filtros de similitud reducen repeticiones que sesgan el entrenamiento.

Recomendación: no usarlas de forma ciega en producción, sino como filtros asistidos en pipelines de ingeniería de datos.

En evaluación, complementan métricas semánticas y ayudan a diagnosticar si un descenso de calidad se debe a errores ortográficos o a desalineación conceptual.

#### 11) Viterbi: decodificación óptima con modelos de estados

El algoritmo de **Viterbi** encuentra la secuencia de estados más probable en un modelo oculto dado un conjunto de observaciones. 
En NLP clásico se usa para etiquetado gramatical, segmentación en idiomas sin espacios y decodificación de CRF para tareas como NER. 

En contextos de FM y LLM aparece como herramienta para:

- Generar etiquetas débiles a gran escala (por ejemplo, un etiquetador de entidades basado en CRF con Viterbi).
- Segmentación previa en idiomas sin espacios, donde un tokenizador por subpalabras se beneficia de límites de palabra razonables.
- Post-proceso de predicciones token-level, imponiendo restricciones globales (esquemas BIO válidos o validaciones de formato,... etc).

Con Viterbi se garantiza consistencia global en secuencias. En la práctica, conviene perfilarlo respecto al tamaño de secuencia y al inventario de estados, yaque la complejidad crece con ambos.

#### 12) Métricas para evaluar la tokenización en FM y LLM

Una política de tokenización se valora por sus métricas operativas:

- Radio de compresión: número promedio de tokens por carácter o por palabra.
- Distribución de longitudes: colas pesadas pueden indicar reglas deficientes o vocabularios poco ajustados al dominio.
- Consistencia entrenamiento-inferencia: divergencias en normalización o reglas previas se multiplican en errores.
- Cobertura de símbolos: cuántos caracteres/bytes comunes quedan resueltos como subpalabras útiles y cuántas veces se cae a rutas de escape.
- Estabilidad entre dominios: validar en redes sociales, legal, clínico, logs.

En la práctica, tamaños de vocabulario entre 32k y 100k suelen equilibrar compacidad y costo de embeddings/softmax. 
Conviene incorporar marcadores de palabra y serializar el ranking de fusiones en estructuras veloces (tries, autómatas) para acelerar la inferencia.

#### 13) Recomendaciones de diseño por variante de BPE

- BPE clásico: adecuado si el corpus es lingüísticamente homogéneo y se desea alineación con morfemas; reforzar con pre-tokenización sólida.
- Byte-level BPE: cuando se necesita cobertura universal y se enfrentan emojis, signos técnicos o mezcla de alfabetos, auditar que no emerjan subpalabras inútiles por dominio.
- BPE con byte-fallback: solución híbrida generalista; mantiene interpretabilidad y cobertura sin reentrenar el tokenizador.

En los tres casos, perfilar el trade-off entre tamaño de vocabulario y longitud de secuencia, y observar el impacto en throughput y latencia. 
La sensibilidad a la normalización es real: pequeñas decisiones sobre Unicode/espacios pueden mover varios puntos porcentuales en tokens por mil caracteres.

#### 14) Integración con normalización, lematización y distancias

Una política integral para FM y LLM orquesta estas piezas:

1. Normalización mínima pero estable.
2. Pre-tokenización top-down para proteger formatos especiales.
3. Subtokenización con BPE o afín (clásico, byte-level o byte-fallback).
4. Medición de compresión y estabilidad.
5. Lematización selectiva si aporta a tareas aguas arriba.
6. Distancias de edición para QA y deduplicación.
7. Viterbi u otra decodificación global para coherencia secuencial.

Esto reduce sorpresas en inferencia, estabiliza costos y mejora la calidad percibida en dominios con ruido.

#### 15) Ejemplo narrativo de extremo a extremo

Texto de prueba: "That U.S.A. poster-print costs $12.40... mañana ₿ y 😊".

1. Normalización: forma Unicode estable (p. ej., NFKC).
2. Pre-tokenización top-down: detectar U.S.A., poster-print, $12.40; tratar el emoji como carácter indivisible.
3. Subtokenización:
   -  BPE clásico: "poster-print" en subpalabras morfológicas; "mañana" puede partirse si fue infrecuente; "₿" puede quedar fuera si el esquema no es byte-aware.
   -  Byte-level BPE: cada símbolo representable por bytes; fusiones frecuentes compactan "ñ" y el emoji; secuencia robusta.
   -  Byte-fallback: palabras comunes en subpalabras interpretables; "₿" y el emoji se emiten como bytes.


| Método | Salida de tokens | Nº de tokens | Comentarios sobre interpretabilidad |
|---|---|---|---|
| BPE clásico | That, U.S.A., poster, -, print, costs, $12.40, ..., ma, ña, na, ₿, y, 😊 | 14 | Alta para texto común; baja para símbolos raros (₿, 😊). |
| Byte-level BPE | T, h, a, t,  , U, ., S, ., A, .,  , p, o, s, t, e, r, -, p, r, i, n, t,  , c, o, s, t, s,  , $, 1, 2, ., 4, 0,  , ...,  , m, a, ñ, a, n, a,  , ₿,  , y,  , 😊 | 50 | Cobertura total; interpretabilidad baja a nivel morfológico. |
| Byte-fallback | That, U.S.A., poster-print, costs, $12.40, ..., mañana, byte:₿, y, byte:😊 | 10 | Alta; bytes solo para símbolos fuera del vocabulario (₿, 😊). |

#### 16) Consideraciones operativas para producción

- Serializar el ranking de fusiones y usar estructuras eficientes (tries, autómatas) para acelerar la aplicación codiciosa.
- Documentar normalización y pre-tokenización como parte del contrato de entrada del modelo.
- Versionar vocabulario y fusiones; pequeños cambios pueden afectar métricas.
- Monitorear la distribución de símbolos y la proporción de caídas a bytes en byte-fallback.
- Auditar con distancias de edición para evitar duplicados o near-duplicates que sesguen la distribución.
- Dimensionar vocabulario al presupuesto de memoria y throughput; 32k–100k suele ser un rango útil.

#### 17) Conexión con FM y LLM actuales y consideraciones multilingües

Los FM y LLM recientes operan sobre datos ruidosos y multiformato; por ello se popularizan variantes byte-aware y caídas a bytes. La pre-tokenización top-down sigue siendo útil para proteger entidades y formatos. 

Aunque la lematización no es requisito de entrenamiento, mejora ciertas tareas de búsqueda y análisis.

En modelos multilingües, un vocabulario sesgado hacia un idioma dominante puede generar subpalabras ineficientes para lenguas minoritarias, elevando longitud de secuencia y costo. 

Mitigaciones:

- Corpos balanceados al entrenar el tokenizador.
- Pre-tokenización específica por idioma donde aplique (por ejemplo: segmentación para chino).
- Métricas de compresión por idioma para detectar desequilibrios.
- Considerar tokenizadores híbridos (BPE clásico + byte-level para coberturas específicas).

#### 18) Herramientas y bibliotecas para implementación

- [SentencePiece](https://github.com/google/sentencepiece): BPE y otras técnicas de subtokenización con opciones byte-aware.
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index):  BPE, byte-level BPE y byte-fallback; personalización de normalización y pre-tokenización.
- [Tiktoken](https://github.com/openai/tiktoken): orientado a grandes modelos e inferencia eficiente con texto heterogéneo.

