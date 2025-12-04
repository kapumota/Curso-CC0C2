### Examen Final de CC0C2

#### 0. Formato común para todos los proyectos

* **Trabajo:** **individual**.
* **Release del enunciado:** **4 de diciembre**.
* **Entrega 1 (E1):** **13 de diciembre**

  * Código funcional (versión *baseline + α*).
  * Informe corto (2-3 páginas).
  * **Video corto de ejecución (5-8 min, sin exposición formal).**
* **Entrega 2 (E2, final):** **20 de diciembre**

  * Código final + experimentos adicionales.
  * Informe extendido/versión final (6-10 páginas).
  * Video de ejecución + **exposición oral/defensa del proyecto**.

> E1 y E2 forman parte del examen final, junto con la exposición oral del 20 de diciembre. La forma exacta de cálculo de la nota está detallada al final de este documento.

### 1. Herramientas y modelos recomendados

**Todo debe funcionar en CPU**, usando la laptop/PC personal del estudiante.

* Lenguajes y entorno:

  * Python.
  * Jupyter Notebook/VS Code.
* Librerías recomendadas:

  * `transformers`, `datasets`, `peft`, `accelerate`.
  * `trl` (opcional).
  * `bitsandbytes` (solo si la máquina lo soporta).
* Para RAG y agentes:

  * `langchain` o `llama-index`.
  * Base vectorial: `chromadb` o `faiss`.
* Modelos pequeños (ejemplos sugeridos):

  * `distilgpt2`, `gpt2`.
  * `google/flan-t5-small`.
  * `tiiuae/falcon-rw-1b` (preferible en versión cuantizada).
  * Modelos de embeddings: `sentence-transformers/all-MiniLM-*`, etc.

> Se permite usar otros modelos pequeños siempre que se justifique en el informe (limitaciones de hardware, idioma, etc.).

### 2. Opciones de proyecto (R1-R8)

Cada estudiante elige **un solo proyecto** entre R1 y R8.

#### R1 - *"Mini-Instructor CPU"*: Fine-tuning eficiente + inferencia optimizada

**Idea:**
Construir un mini modelo "instructor" para una tarea concreta (por ejemplo, explicar conceptos de la universidad, responder dudas sobre un curso, o resumir artículos cortos) usando **fine-tuning eficiente en parámetros + cuantización + KV caching**, y medir el impacto en **calidad vs latencia**.

#### Alcance mínimo

* Elegir un modelo base pequeño (por ejemplo, `flan-t5-small` o `distilgpt2`) y un dataset de **instruction-tuning** pequeño (mezcla de datasets públicos + algunos ejemplos propios).
* Aplicar **PEFT (LoRA)**, idealmente con 4-bit u 8-bit (trabajo con precisión reducida).
* Experimentar con:

  * **Tamaño de batch**.
  * Algún **parámetro de regularización** (weight decay, dropout).
  * Opciones de **optimización de memoria** (gradient checkpointing, offloading, etc., si aplica).
* Comparar **inferencia baseline** vs:

  * KV caching activado.
  * Cuantización (por ejemplo, int8) vs 16/32-bit.
  * Opcional: probar al menos una técnica de **decodificación especulativa o paralela** (aunque sea usando una librería o un prototipo simple).

#### Entrega 1 (13/12)

* Pipeline completo de **fine-tuning con PEFT** en CPU (aunque sea con pocas iteraciones/épocas).
* Comparación básica:

  * Modelo sin fine-tuning vs modelo ajustado en 2-3 ejemplos de test.
  * Tabla con **latencia de inferencia** (ms o s) para varias configuraciones:

    * sin KV cache/con KV cache,
    * con y sin cuantización.
* Video mostrando:

  * Cómo se ejecuta el script/notebook de entrenamiento.
  * Cómo se prueba el modelo con algunos *prompts*.

#### Entrega 2 (20/12)

* Experimentos más sistemáticos:

  * Variar **batch size** y documentar impacto en tiempo y uso de memoria.
  * Probar al menos **dos niveles de precisión/cuántización** (por ejemplo, float16 vs int8).
  * Analizar algún **trade-off calidad/latencia** (por ejemplo, BLEU/ROUGE/puntuación humana simple vs tiempo de respuesta).
* Informe final:

  * Explicación clara de:

    * **Fine-tuning**.
    * **Fine-tuning eficiente en parámetros (PEFT/LoRA)**.
    * **Precisión reducida**.
    * **KV caching**.
    * Opcional: **early-exit** o técnicas afines si las implementa.
  * Gráficas/tablas simples con resultados.

### R2 - *"Continual Student"*: Preentrenamiento continuo + Replay

**Idea:**
Simular un escenario de **preentrenamiento continuo** sobre un dominio local (por ejemplo, documentos de la facultad, reglamentos, manuales) y estudiar el **"catastrophic forgetting"** y técnicas de **replay** para mitigarlo.

#### Alcance mínimo

* Modelo base pequeño (por ejemplo, `distilgpt2`).
* Definir dos dominios:

  * **Dominio A:** por ejemplo, noticias breves o textos generales.
  * **Dominio B:** por ejemplo, documentos de la universidad / facultad / reglamentos.
* Realizar dos fases:

  * **Fase A:** fine-tuning/preentrenamiento adicional en A.
  * **Fase B:** preentrenamiento adicional en B.
* Medir desempeño en tareas de A y B después de cada fase (perplexity, accuracy en una tarea pequeña, etc.).

#### Entrega 1

* Implementar las dos fases de entrenamiento sucesivas (A, luego B) **sin replay**.
* Métricas simples (perplexity, accuracy, etc.) para A y B **antes y después** de cada fase.
* Informe corto explicando:

  * Qué es **preentrenamiento continuo**.
  * Qué es **catastrophic forgetting**.
* Video mostrando:

  * Entrenamiento en A y en B.
  * Evaluación de A y B.

#### Entrega 2

* Implementar al menos **una técnica de replay**, por ejemplo:

  * Replay uniforme: mezclar ejemplos antiguos de A con los nuevos de B.
  * Replay con subconjuntos (*subset methods*): conservar solo ejemplos "más informativos".
* Discutir uso de **PEFT / expansión de parámetros**:

  * Por ejemplo, añadir nuevos adapters para el dominio B sin modificar el modelo base.
* Comparar 3 modelos:

  1. Modelo base.
  2. Modelo con preentrenamiento continuo A->B sin replay.
  3. Modelo con preentrenamiento continuo A->B con replay.
* Discusión en el informe:

  * Cómo se ve el **catastrophic forgetting**.
  * Cómo ayuda el replay.
  * Relación con **continual pre-training** en LLMs grandes.

### R3 - *"Adapter Zoo"*: combinación, ensamblado y fusión de adapters

**Idea:**
Entrenar varios **adapters LoRA** sobre distintas tareas (por ejemplo, resumen, estilo formal/informal, explicación paso a paso) y estudiar **model ensembling**, **model fusion** y **adapter merging**.

#### Alcance mínimo

* Usar un único modelo base (por ejemplo, `flan-t5-small` o `distilgpt2`).
* Entrenar al menos **2 adapters**:

  * **Adapter A:** tarea de *summarization*.
  * **Adapter B:** tarea de *style transfer* (de informal a académico, de lenguaje simple a técnico, etc.).
* Evaluar cada adapter:

  * En su tarea principal.
  * En la otra tarea (cross-task), para ver qué tanto se transfieren capacidades.

#### Entrega 1

* Entrenar y guardar adapters A y B.
* Script/notebook para:

  * Cargar modelo base + adapter A.
  * Cargar modelo base + adapter B.
  * Evaluar en 2-3 ejemplos por tarea.
* Informe corto:

  * Explicación de **PEFT** y **LoRA**.
  * Por qué estos enfoques son "eficientes en parámetros".
* Video: demo del uso de cada adapter.

#### Entrega 2

* Implementar al menos **una estrategia de combinación**:

  * **Adapter merging**: mezclar pesos de LoRA.
  * **Model ensembling**: combinar salidas/logits de varios modelos.
  * **Model fusion**: promediar pesos finales (cuando tenga sentido).
* Evaluar:

  * Cada adapter individual.
  * La configuración fusionada / ensemble.
* En el informe:

  * Ventajas y desventajas de mantener múltiples adapters vs fusionarlos.
  * Relación con **expansión de parámetros** (añadir nuevos adapters para nuevas tareas sin tocar el modelo base).

### R4 - *"RLHF-lite & Alucinaciones"*: alineación y razonamiento con feedback humano

**Idea:**
Construir un mini-pipeline de **entrenamiento de alineación simplificado** (RLHF con pocas iteraciones o DPO/ORPO offline) con un modelo pequeño, para reducir **alucinaciones** en una tarea concreta (por ejemplo, responder preguntas factuales sobre un conjunto de documentos locales).

#### Alcance mínimo

* Crear un dataset pequeño de **pares de preferencia**:

  * Para una pregunta, generar 2 respuestas (una más correcta, otra peor) y etiquetar cuál es mejor.
  * Se pueden usar LLMs para generar candidatos y luego seleccionar manualmente.
* Usar `trl` u otra librería para:

  * Hacer **RLHF con PPO** sobre pocas iteraciones; o
  * Aplicar **DPO** (Direct Preference Optimization) u otro método offline con el dataset de preferencias.

#### Entrega 1

* Modelo base + dataset de preferencias armado.
* Entrenamiento de un primer ciclo de RLHF-like o DPO.
* Evaluación:

  * Comparar tasa de respuestas "obviamente incorrectas" antes y después del entrenamiento.
* Informe corto:

  * Definición de **entrenamiento de alineación**, **RLHF** y tipos de feedback humano.
  * Ejemplos de **alucinaciones** detectadas.
* Video: demostración del sistema antes y después del entrenamiento.

#### Entrega 2

* Explorar **técnicas de mitigación de alucinaciones**:

  * **Self-consistency**: muestrear varias respuestas y hacer voto mayoritario.
  * **Chain-of-actions / chain-of-thought**: forzar al modelo a razonar paso a paso.
  * **Recitación**: primero enumerar hechos relevantes, luego responder.
  * Opcional: integrar **RAG** como baseline con contexto recuperado.
* Implementar al menos un **verificador**:

  * Regla simple: comprobar si la respuesta cita documentos, tiene fechas plausibles, etc.
  * O un clasificador pequeño que etiquete respuestas como seguras/inseguras.
* Relacionar con:

  * Tipos de razonamiento: **deductivo, inductivo, abductivo**.
  * **Alucinaciones en contexto** vs alucinaciones por información irrelevante.

### R5 - *"RAG 101"*: pipeline mínimo con evaluación rigurosa

**Idea:**
Construir un sistema de **RAG básico** bien diseñado: pipeline completo de **ingesta -> limpieza -> chunking -> embeddings -> base vectorial -> consulta -> generación**, con **métricas de recuperación y de QA**.

#### Alcance mínimo

* Corpus de documentos (por ejemplo, PDFs de algún curso, documentación técnica, artículos breves).
* Pipeline:

  1. Ingesta y limpieza (remover ruido, normalizar idioma).
  2. Chunking con tamaño y solapamiento configurables.
  3. Embeddings (ejemplo: `all-MiniLM-L6-v2`).
  4. Base vectorial (`chromadb`, `faiss`, etc.).
  5. Retrieval + generación con un modelo pequeño (por ejemplo, `flan-t5-small`).
* Construir un conjunto pequeño de **preguntas etiquetadas** con respuesta correcta ("gold").

#### Entrega 1

* RAG funcionando end-to-end:

  * Dada una pregunta -> recuperar contextos -> generar respuesta.
* Métrica mínima:

  * **Recall@k** para el retriever.
  * Al menos una métrica de QA (Exact Match, F1, etc.) sobre unas pocas preguntas.
* Informe corto:

  * Explicación de la arquitectura **retrieve -> read -> generate**.
  * Conceptos básicos de embeddings, base vectorial, índices aproximados, distancia coseno vs L2.
* Video: demo del sistema respondiendo preguntas.

#### Entrega 2

* Experimentos:

  * Comparar al menos **dos estrategias de chunking** (ej. 256 vs 512 tokens, con y sin solapamiento).
  * Comparar al menos **dos valores de k** (ej. k=3 vs k=10).
  * Evaluar 2-3 métricas (recall@k, MRR, EM/F1, u otra; se puede usar LLM-as-a-judge si se justifica).
* Buenas prácticas:

  * Diseñar metadatos (fuente, idioma, confidencialidad, timestamp).
  * Implementar **caching de consultas** frecuentes.
* Discusión:

  * Comparar **RAG vs fine-tuning clásico**: ¿cuándo conviene cada enfoque?
  * Coste adicional de inferencia por retrieval y posibles técnicas para abaratarlo.

### R6 - *"RAG++ & Reasoning"*: retrieval híbrido, drift y razonamiento sobre contexto

**Idea:**
Extender RAG 101 a un **RAG avanzado** con **retrieval híbrido** (sparse + dense), análisis de **drift de embeddings** y tareas explícitas de **razonamiento deductivo/inductivo/abductivo** sobre el contexto recuperado.

#### Alcance mínimo

* Partir de un RAG básico (puede ser el resultado de R5 o uno nuevo).
* Incorporar:

  * Un retriever **sparse** (BM25, TF-IDF, etc.).
  * Un retriever **dense** (embeddings).
  * Un modo **híbrido** que combine puntuaciones.
* Diseñar preguntas donde se requiera:

  * Razonamiento **deductivo** (combinar 2-3 hechos explícitos).
  * **Inductivo** (detectar patrones).
  * **Abductivo** (explicar observaciones con hipótesis plausibles).

#### Entrega 1

* RAG híbrido funcionando con tres modos:

  * Solo sparse.
  * Solo dense.
  * Híbrido.
* Evaluación:

  * Comparar **recall@k** en los tres modos.
* Informe:

  * Explicar retrievers sparse/dense/híbridos.
  * Dar ejemplos de preguntas donde uno funciona mejor que otro.
* Video: demo consultando en los tres modos.

#### Entrega 2

* Análisis de **drift de embeddings**:

  * Cambiar de modelo de embeddings o refinarlo a mitad del proyecto.
  * Discutir qué ocurre con la base vectorial (reindexación, namespaces, etc.).
* Razonamiento:

  * Diseñar prompts específicos que fomenten razonamiento sobre el contexto (map-reduce, refine, citas explícitas).
  * Evaluar si el modelo realmente usa la evidencia recuperada (por ejemplo, verificando citas o contenido).
* Relación con otros bloques:

  * Cómo RAG ayuda a reducir alucinaciones (complemento a RLHF/alignment).
  * Impacto de RAG en costo de inferencia y cómo mitigarlo.

### R7 - *"Agentic Tools"*: workflows, agentes y guardrails con observabilidad

**Idea:**
Diseñar un **workflow de LLM** con un agente que pueda usar varias herramientas (RAG, calculadora, API local, etc.) siguiendo paradigmas **pasivo / explícito / autónomo**, con **guardrails y observabilidad básica**.

#### Alcance mínimo

* Definir un problema multi-paso (por ejemplo: "responder preguntas sobre planes de estudio, incluyendo cálculos de créditos y verificación de requisitos").
* Implementar:

  * Un **workflow**: preprocesamiento -> llamada al LLM -> postprocesamiento.
  * Un agente con un **agent loop** del tipo *observar -> razonar -> actuar -> observar*.
  * Al menos **2 herramientas**:

    * RAG sobre documentos.
    * Calculadora (implementada en código).
    * (Opcional) API REST local simulada.

#### Entrega 1

* Versión básica:

  * Agente **explícito**: el código decide cuándo llamar a cada herramienta, pero los prompts ya siguen la idea de agent loop.
* Logging:

  * Guardar en un log (texto o json) cada prompt, herramientas llamadas y resultados.
* Informe:

  * Definir paradigmas de interacción (pasivo, explícito, autónomo).
  * Concepto de agente, agent loop, *data stores*, y orquestación (aunque sea casera).
* Video: explicación del flujo y demo de al menos 1-2 consultas.

#### Entrega 2

* Versión más **autónoma**:

  * El modelo decide cuándo usar herramientas, guiado por un *agent loop prompt*.
* **Guardrails**:

  * Validación de parámetros de herramientas (evitar endpoints prohibidos, inputs inválidos, etc.).
  * Límites de uso de herramientas (número máximo de llamadas por consulta).
  * Registro estructurado para **auditoría** (json con trazas).
* Observabilidad:

  * Métricas simples: número de pasos por query, número de llamadas a cada herramienta, tiempo total.
  * Visualización sencilla (tabla o gráfico) de estos datos.
* Discusión:

  * Seguridad y límites de agentes (zero-trust, validación, auditoría del loop).
  * Cómo se integraría este workflow en un sistema de microservicios / APIs.

### R8 - *"Multi-Agent QA"*: Manager + Workers + Critic con RAG

**Idea:**
Construir un sistema **multi-agente** para QA empresarial o académico usando **RAG + agentes + un verificador crítico**, con énfasis en **razonamiento** y reducción de **alucinaciones**.

#### Alcance mínimo

* Elegir un dominio (reglamentos, sílabos, documentación técnica, etc.).
* Definir 3 tipos de agentes:

  1. **Manager**: recibe la pregunta y la descompone en subtareas.
  2. **Worker(s)**: ejecutan subtareas (consultar RAG, sintetizar respuestas parciales).
  3. **Critic/Reviewer** (se añade en E2): revisa y valida la respuesta final.

#### Entrega 1

* Implementar el flujo básico:

  * Pregunta -> Manager -> Workers -> respuesta final.
* RAG sencillo para que los workers obtengan contexto.
* Log de la interacción entre agentes (texto, json, etc.).
* Informe corto describiendo los roles de Manager y Workers.
* Video: demo del flujo con al menos 2-3 ejemplos.

#### Entrega 2

* Añadir el **Critic**:

  * Verifica si la respuesta usa el contexto recuperado.
  * Puede solicitar "reintento" a un worker si detecta inconsistencias.
* Técnicas de razonamiento:

  * **Chain-of-actions**: documentar la secuencia de pasos que siguen los agentes.
  * **Self-consistency**: generar varias respuestas y dejar que el Critic elija la más coherente.
  * Identificar en ejemplos si el razonamiento es mayormente **deductivo**, **inductivo** o **abductivo**.
* Métricas:

  * Comparar tasa de errores/alucinaciones con y sin Critic.
  * Medir costo (tiempo, número de llamadas) vs mejora de calidad.
* Discusión:

  * Relación con workflows multi-agente (manager + workers).
  * Posibles casos de uso reales (QA empresarial, generación de código + agente "tester").
* Video final: explicación de resultados y demo del sistema completo.

### 3. Entregables y rúbricas

#### 3.1. Entregable 1 (E1) - 13 de diciembre

**Objetivo:**
Tener el proyecto "en pie": entorno, baseline funcionando, primeros experimentos y documentación mínima.

#### Contenido esperado

* Código + trabajo en progreso (baseline bien armado).
* Primeros experimentos y métricas básicas.
* **Video de ejecución (sin exposición oral formal)**.

#### Rúbrica E1

| Código   | Criterio                                                                   | Puntos máx. |
| -------- | -------------------------------------------------------------------------- | ----------- |
| **E1-A** | Entorno y estructura del proyecto (repo ordenado, `README`, cómo ejecutar) | 4           |
| **E1-B** | Núcleo técnico / baseline funcionando (entrena/ejecuta sin romperse)       | 5           |
| **E1-C** | Datos + primeros experimentos / métricas básicas                           | 4           |
| **E1-D** | Informe corto (2-3 páginas): objetivo, método, resultados preliminares     | 3           |
| **E1-E** | Video de ejecución (5-8 min, demo técnica, sin exposición formal)          | 4           |
|          | **TOTAL E1**                                                               | **20**      |

#### 3.2. Entregable 2 (E2) - 20 de diciembre

**Objetivo:**
Entregar el proyecto **consolidado y profundo**, demostrando comprensión de las técnicas avanzadas y capacidad de explicarlas.

#### Contenido esperado

* Continuación directa y mejora del E1 (no es un proyecto nuevo).
* Experimentos más profundos, comparaciones de técnicas.
* **Video de ejecución** + **exposición oral / defensa del proyecto**.

#### Rúbrica E2

| Código   | Criterio                                                                   | Puntos máx. |
| -------- | -------------------------------------------------------------------------- | ----------- |
| **E2-A** | Continuidad y calidad del entorno (reproducible, mejora sobre E1)          | 3           |
| **E2-B** | Núcleo técnico avanzado (técnicas del curso bien integradas y funcionando) |  6          |
| **E2-C** | Experimentos y análisis (comparaciones, ablations, métricas bien usadas)   | 5          |
| **E2-D** | Informe final (6-10 páginas, claridad y conexión con teoría)               | 3           |
| **E2-E** | Video de ejecución (demo final, 5-8 min)                                   | 3          |
|          | **TOTAL E2**                                                               | **20**      |


#### 3.3. Exposición oral final - 20 de diciembre

**Objetivo:**
Evaluar la **capacidad del estudiante para explicar y defender** su proyecto, conectando la implementación con los conceptos del curso.

* La exposición se realiza el **20 de diciembre** (mismo día que E2).
* Duración orientativa: **15–30 minutos** de presentación + preguntas.
* Se puede usar diapositivas, demostración en vivo u otro medio, siempre que se vea:

  * Cómo está construido el sistema.
  * Qué técnicas avanzadas se aplicaron.
  * Qué resultados se obtuvieron y cómo se interpretan.

#### Rúbrica exposición oral

| Código   | Criterio                                                                                     | Puntos máx. |
| -------- | -------------------------------------------------------------------------------------------- | ----------- |
| **EO-A** | Claridad y estructura de la presentación (inicio, desarrollo, cierre)                        | 2           |
| **EO-B** | Explicación técnica del sistema (arquitectura, decisiones de diseño, relación con el código) | 5          |
| **EO-C** | Análisis de experimentos y resultados (qué se midió, qué se aprendió)                        | 4           |
| **EO-D** | Demostración en vivo y manejo del tiempo                                                     | 2           |
| **EO-E** | Respuestas a preguntas y dominio conceptual (fine-tuning, RAG, alignment, agentes, etc.)     | 7           |
|          | **TOTAL Exposición oral**                                                                    | **20**      |

**Cálculo de la nota del examen final**

La nota del examen final se obtiene a partir del **Entregable 2 (E2)** y de la **Exposición oral (EO)**, de la siguiente manera:

1. Primero se calcula el promedio entre E2 y EO:

   * Nota_sin_penalización = (E2 + EO)/2

2. Si el/la estudiante **entregó E1 a tiempo**, la nota final del examen es:

   * Nota_Examen_Final = Nota_sin_penalización

3. Si el/la estudiante **NO entregó E1**:

   * La nota de E1 es 0.
   * Se aplica una penalización de **-8 puntos** al promedio anterior:

     * Nota_Examen_Final = Nota_sin_penalización - 8
   * En este caso, incluso si E2 = 20 y EO = 20, la nota máxima alcanzable en el examen final será **12/20**.


