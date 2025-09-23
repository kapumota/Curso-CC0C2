## Taxonomía de modelos de lenguaje y LLMs

#### Introducción general
Esta taxonomía proporciona una clasificación exhaustiva de los modelos de lenguaje grandes (LLMs) y sus variantes, cubriendo desde arquitecturas fundamentales hasta aplicaciones prácticas en producción. 
Se enfoca en criterios clave como arquitectura, objetivos de preentrenamiento, modalidades y tamaños, extendiéndose a alineación, patrones de agentes,  selección de uso y evaluación. 

### 1. Por arquitectura

- **Encoder-only**: Representados por BERT, RoBERTa o LegalBERT, estos modelos son expertos en comprensión profunda de texto, aprendiendo a reconstruir huecos mediante enmascarado (*masked language modeling*). Son ideales para tareas como clasificación (sentimiento, NER), extracción de información y re-ranking en sistemas RAG. También generan *embeddings* robustos para búsqueda semántica y análisis de texto en dominios específicos (por ejemplo, BioBERT para textos clínicos).

- **Decoder-only**: Incluyen modelos como GPT, Llama, Mistral o Grok (creado por xAI), entrenados de forma autoregresiva para predecir el siguiente token. Son la base de aplicaciones conversacionales (*chat*), redacción guiada, generación de código (por ejemplo, CodeLlama) y tareas creativas. Su fortaleza radica en la fluidez y adaptabilidad en generación libre. Para contexto largo, técnicas operativas incluyen KV cache y KV cache reuse (prefill para inicializar contexto vs. decode para generación token-a-token), batching dinámico (agrupar consultas variables), streaming (generación en tiempo real) y speculative decoding (predicciones paralelas para reducir latencia sin perder calidad).

- **Encoder-decoder (seq2seq)**: Modelos como T5, FLAN-T5 o BART mapean entradas a salidas mediante codificación y decodificación. Entrenados con *denoising* y *span corruption* (enmascarado de fragmentos), destacan en traducción (por ejemplo, mBART para multilingüismo), resumen dirigido y reformulación. Son versátiles para tareas estructuradas que requieren transformación precisa.

- **MoE esparsos**: Los modelos de *Mixture of Experts* (como Mixtral) enrutan tokens a subconjuntos de “expertos”, logrando alta capacidad con menor coste computacional por token. Ofrecen gran *throughput* y latencias moderadas, ideales para aplicaciones de alto volumen en producción. Sin embargo, presentan riesgos como desbalanceo de carga entre expertos (expertos subutilizados), token dropping/capacity factor (sobrecarga que causa pérdida de tokens) y coste de enrutamiento (comunicación inter-GPU en clústers). Para mitigar, se sugiere load balancing (distribución equitativa), auxiliary losses (penalizaciones para equilibrar) y routing constraints (restricciones en selección de expertos).

- **Sin atención o mixtas**: Arquitecturas como los SSM (*State Space Models*, por ejemplo, Mamba), RetNet y RWKV priorizan la eficiencia en contextos largos mediante recurrencia lineal o convoluciones, reduciendo la dependencia de la atención cuadrática de los transformers. Son prometedoras para *streaming* y tareas con secuencias extensas (por ejemplo, análisis de logs o diálogos largos). Escalan bien en longitud pero a veces pierden precisión en razonamiento composicional; los híbridos (Transformer+SSM) mitigan este trade-off combinando eficiencia secuencial con atención global.

### 2. Por objetivo de preentrenamiento

- **CLM (Causal Language Modeling)**: Típico de los decoder-only, el modelo predice el siguiente token basándose en el contexto previo. Es el estándar para generación de texto fluida y aplicaciones conversacionales.

- **MLM (Masked Language Modeling)**: Característico de los encoder-only, el modelo completa huecos enmascarados, aprendiendo representaciones ricas para tareas de comprensión como clasificación o *embedding generation*.

- **Denoising con *span corruption***: Usado en encoder-decoder (T5, BART), implica reconstruir fragmentos completos de texto enmascarados, lo que es más informativo que palabras aisladas. Es ideal para tareas de mapeo entrada-salida como traducción y resumen.

- **Permutación y variantes**: Modelos como XLNet reordenan el factor de probabilidad para fomentar generalización. Aunque menos comunes hoy, ilustran enfoques alternativos al modelado del lenguaje.

- **Aclaraciones rápidas**: 
  - **UL2 / Mixture-of-Denoisers (ruidos mixtos tipo T5)**: Combina múltiples tipos de ruido (enmascarado, permutación, etc.) para ayudar a modelos *instruct* a generalizar mejor en formatos de tarea variados, mejorando la adaptabilidad en fine-tuning.
  - **Prefix-LM (atención en prefijo + autoregresión)**: Útil para few-shot learning y condicionamiento largo, permitiendo atención bidireccional en el prefijo (entrada) mientras se mantiene autoregresión en la salida.

### 3. Por modalidad

- **Solo texto**: Pueden ser generales (como Llama) o especializados por dominio (LegalBERT para textos legales, BioBERT para clínicos, o modelos entrenados en logs para ciberseguridad). Son la base de la mayoría de aplicaciones NLP.

- **Código**: Modelos como CodeLlama, StarCoder o DeepSeek están optimizados para generación y asistencia en programación, soportando agentes desarrolladores y flujos de trabajo en DevOps.

- **Multimodal**: Distingue VLM ligero (imagen<->texto, como CLIP o LLaVA) vs. omnimodal (audio, visión, acción, como GPT-4o o modelos que integran sensores IoT/robótica). Incluye OCR estructurado para tablas/diagramas, tool use visual (parseo de UI, formularios) y razonamiento sobre multimedia. Ejemplos: descripción de imágenes, búsqueda semántica visual y análisis de documentos complejos.

- **Multilingües**: Modelos como mBART o Aya están diseñados para lenguas de bajo recurso, mejorando la accesibilidad en contextos globales y soportando traducción y generación en múltiples idiomas. Retos incluyen guiones distintos (CJK para chino/japonés/coreano, RTL para árabe/hebreo), romanización (transcripción a alfabeto latino) y transferencia a bajo-recurso; se sugiere adapters (módulos ligeros por idioma), vocab shared (vocabulario compartido) vs. vocab por script para optimizar.

### 4. Por tamaño y topología

- **On-device pequeños (≤3B)**: Modelos como Phi-3, Gemma o MobileBERT priorizan privacidad y baja latencia en dispositivos móviles o IoT mediante técnicas como destilación de conocimiento y quantización (por ejemplo, 4-bit). Son ideales para *edge computing*.

- **Medios (7-30B)**: Ofrecen un equilibrio óptimo entre calidad y coste para la mayoría de aplicaciones. Ejemplos incluyen Llama-7B, Mistral-7B o Grok, que son versátiles para chat, RAG y *tool use*.

- **Grandes o *frontier* (≥70B)**: Como Llama-70B o modelos propietarios, maximizan el rendimiento a costa de infraestructura exigente. Son ideales para tareas complejas que requieren razonamiento avanzado.

- **Densos vs. MoE**: Los densos activan todos los parámetros en cada paso, mientras que los MoE (como Mixtral) enrutan selectivamente, logrando mayor capacidad con menor coste por token, aunque requieren infraestructura especializada para el enrutamiento dinámico.

### 5.  Alineación, herramientas y memoria

Tras el preentrenamiento **base**, los modelos se alinean para mejorar su utilidad y seguridad:

- **Alineación**:
  - **Supervisado (*instruction-tuning*, SFT)**: Usa pares instrucción-respuesta para adaptar modelos a diálogo y tareas específicas (por ejemplo, Grok está afinado para respuestas útiles y veraces).
  - **RLHF, DPO, ORPO**: RLHF optimiza por recompensa humana, mientras que DPO/ORPO usan preferencias sin reward model para mayor eficiencia. Advertencia: sobre-penalización puede causar regresión de factualidad, reduciendo la asertividad en respuestas factuales. Métodos como *constitutional AI* o *self-alignment* (refinamiento autónomo de respuestas) están emergiendo para reducir la dependencia de supervisión humana.
  
- **Herramientas**:
  - Incluyen *function calling*, ejecución de código, APIs externas y RAG. Este último fundamenta respuestas en fuentes verificables, usando índices vectoriales (Faiss, HNSW) para recuperar información relevante.
  - Ejemplo: Grok puede buscar en la web o analizar contenido subido (imágenes, PDFs) para responder con datos actualizados.

- **Memoria y contexto largo**:
  - Técnicas como *rotary embeddings*, ALiBi o SSM permiten manejar ventanas de contexto extensas. Diferencia: memoria episódica (historial de sesión para coherencia inmediata), semántica (vector DB para conocimiento persistente) y procedimental (herramientas para acciones reutilizables).
  - Buenas prácticas RAG: chunking por semántica (dividir texto por significado), híbrido BM25+vector search (combinar keywords y embeddings), re-ranking (priorizar relevancia), citas canónicas (referencias estandarizadas) y higiene de índice (limpieza periódica para evitar ruido).

### 6.  Patrones para agentes con LLM

Los agentes combinan LLMs con planificación, acción y verificación:

- **ReAct**: Alterna razonamiento y uso de herramientas, investigando y revisando en ciclos cortos. Es ideal para tareas dinámicas como búsqueda web o ejecución de APIs.
- **Planner-Executor o Planner-Controller-Executor**: Separan planificación, control y ejecución, con verificadores para corregir errores o reintentar pasos fallidos.
- **PAL/Program-Aided**: Genera programas intermedios para resolver sub-tareas complejas (por ejemplo, cálculos matemáticos precisos).
- **Orquestación de herramientas**: Encadena búsqueda web, RAG, calendarios, bases internas y APIs de negocio para flujos de trabajo reales (por ejemplo, automatización en CRM).
- **Self-critique y reflexión**: El agente audita sus respuestas, detecta errores y corrige en iteraciones.
- **Multi-agente**: Asigna roles especializados (planificador, codificador, tester) y promueve debate para reducir alucinaciones. Ejemplo: un equipo de agentes para desarrollo de software.
- **Workflows/DAG**: Formalizan flujos con *quality gates* de seguridad y calidad en cada nodo LLM.
- **Routers/Mixture-of-Agents**: Seleccionan el mejor modelo o herramienta por consulta, optimizando coste y calidad.
- **Tool learning**: Los modelos aprenden a usar herramientas sin *function calling* explícito, una tendencia emergente para agentes más autónomos.

Estos patrones heredan el principio del modelado del lenguaje: estimar probabilidades condicionadas y combinarlas, evolucionando desde la interpolación clásica a sistemas complejos de agentes.

### 7. ¿Cuándo usar qué?

- **Clasificación y extracción con baja latencia**: Usa encoder-only (BERT, BioBERT) o encoders en RAG para re-ranking y *embeddings*. Ejemplo: análisis de sentimiento en reseñas o NER en textos legales.
- **Generación libre, chat o código**: Decoder-only afinados (Grok, Llama-7B, CodeLlama) con *instruction-tuning* y RLHF/DPO para fluidez y control. Ejemplo: asistentes conversacionales o generación de scripts.
- **Traducción y resumen dirigido**: Encoder-decoder (T5, mBART) con *span corruption* y SFT. Ejemplo: traducción multilingüe o resúmenes de informes técnicos.
- **Contexto largo o *streaming***: SSM (Mamba) o decoder-only con *sparse attention*/*sliding window* y RAG/memoria externa. Ejemplo: análisis de logs extensos o diálogos continuos.
- **Alta calidad con coste ajustado**: MoE medianos (Mixtral) para alto *throughput*. Ejemplo: chatbots escalables en atención al cliente.
- **Tareas compuestas con herramientas**: Agentes ReAct o Planner-Executor con *tool use* y validadores. Ejemplo: automatización de flujos en e-commerce con búsqueda, APIs y verificación.
- **Aplicaciones en *edge***: Modelos pequeños (Phi-3, Gemma) con quantización para dispositivos móviles o IoT. Ejemplo: asistentes locales en smartphones.
- **Multilingüismo**: Modelos como mBART o Aya para lenguas de bajo recurso. Ejemplo: traducción en tiempo real para comunidades multilingües.

### 8. Etiquetas y evaluación

**Etiquetas** para clasificar modelos:
- **Base, instruct, chat**: Según el grado de alineación.
- **General o de dominio**: Según el corpus (legal, clínico, código).
- **Open-weights o cerrado**: Según disponibilidad (Llama vs. modelos propietarios).
- **Denso o MoE; small/medium/large**: Según topología y tamaño.
- **Solo texto, código, multimodal, multilingüe**: Según señales soportadas.
- **RAG-native o tools-native**: Si integran recuperación o herramientas de forma nativa.

**Evaluación**:
- **Perplejidad**: Métrica intrínseca que mide la calidad lingüística (baja perplejidad = mejor predicción de datos reales). Es menos relevante para multimodales o agentes.
- **Métricas de tarea**: BLEU/ROUGE para traducción/resumen, F1 para clasificación, precisión en *tool use*.
- **Benchmarks**: MMLU, Big-Bench, HELM para rendimiento general; evaluaciones humanas o juzgadores automáticos para utilidad y seguridad. Para agentes: exactitud de tool use, tasa de bucles/reintentos, coste por tarea, SLA de latencia y robustez OOD (fuera de distribución).
- **Aclaraciones rápidas**: Sesgos en LLM-as-a-judge (evaluadores automáticos pueden heredar sesgos del LLM) y riesgo de contaminación de benchmarks (modelos sobreentrenados en evals públicas).

### 9. Operación/DevOps-MLOps

Para llevar LLMs a producción, considera métricas operativas y ciclos de despliegue:

- **SLO/SLA por request**: Latencia p95/p99 (95%/99% de respuestas en <X ms), tokens/s, throughput bajo carga (usuarios concurrentes), observabilidad (trazas de tool use, fingerprint de prompt+context, guardrails para filtrar outputs riesgosos).
- **Ciclo de despliegue**: Cuantización (AWQ/GPTQ para reducir tamaño/memoria), LoRA/QLoRA para domain-fit (ajuste eficiente por dominio), A/B testing con métricas de negocio (engagement, conversión), y políticas de rollbacks (reversión automática si falla SLO).
- **Mejores prácticas**: Monitoreo continuo de drift (cambios en datos de entrada), feedback loops (recolección de interacciones reales para reentrenamiento) y escalabilidad horizontal (clústers distribuidos).

### 10.  Salida estructurada / herramientas avanzadas

Para outputs confiables y orquestación:

- **JSON fiable**: Usa *function calling* con schema definido, validación automática y self-healing (reintento si falla el parseo). Ejemplo: extracción estructurada de datos de respuestas.
- **FSM/guardrails**: Máquinas de estados finitos para orquestar agentes (transiciones lógicas entre pasos), guardrails para seguridad (filtrar PII o contenido tóxico).
- **Idempotencia en APIs**: Asegura que llamadas externas (APIs de negocio) sean repetibles sin efectos colaterales, reduciendo errores en reintentos.

### 11. Privacidad, datos y gobernanza

**Introducción a esta subsección**: La gobernanza de datos es crítica en LLMs para mitigar riesgos éticos, legales y operativos, asegurando compliance con regulaciones como GDPR o leyes de IA. Incluye rastreo de datos, protección de privacidad y pruebas de robustez, integrándose en todo el ciclo de vida del modelo.

- **Data lineage**: Rastreo del origen de datos de entrenamiento (corpus, fuentes) para auditar sesgos o copyright.
- **PII y privacidad**: Detección/anonymización de información personal identificable; técnicas como differential privacy para preentrenamiento.
- **Copyright**: Verificación de licencias en datasets; uso de datos sintéticos para evitar infracciones.
- **Red-teaming**: Pruebas adversariales para detectar vulnerabilidades (jailbreaks, alucinaciones).
- **Drift y feedback loops**: Monitoreo de cambios en datos reales; bucles de retroalimentación humana para iterar el modelo sin exponer datos sensibles.

