### 1. Introducción a NLP, LLMs y Foundation Models, y a la construcción de aplicaciones con FMs

La ingeniería moderna de IA (AI Engineering) surge en el punto de contacto entre tres fuerzas: el progreso en modelos fundacionales (Foundation Models, FMs), la disponibilidad de infraestructuras de cómputo elásticas y especializadas, y una demanda empresarial por soluciones que pasen de demostraciones aisladas a productos confiables y mantenibles. 

En el ámbito del lenguaje, el procesamiento de lenguaje natural (NLP) dejó de ser una colección de técnicas separadas para clasificación, extracción o traducción, y se convirtió en un enfoque unificado en torno a modelos preentrenados a gran escala que luego se adaptan mediante instrucciones, preferencias o recuperación aumentada de conocimiento.

"**Foundation Model**" designa una familia de modelos de propósito general entrenados a gran escala (texto, código, imágenes, audio, video o datos multimodales), que sirven como base para múltiples tareas aguas abajo. Los LLMs son un caso particular, centrados en texto que aprendieron, con billones de tokens, regularidades estadísticas de lenguaje y mundo. Ese aprendizaje los habilita para: responder preguntas, resumir, traducir, razonar en tareas limitadas, generar código y orquestar herramientas externas. 

La construcción de aplicaciones sobre FMs pasa por una guía práctica: definir casos de uso concretos, diseñar "prompts" y contextos con rigor, aislar componentes de recuperación o de agentes si corresponde, establecer líneas de verificación antes de entregar resultados a sistemas aguas abajo y, sobre todo, instrumentar evaluación continua con datos realistas.

Una aplicación bien diseñada no es solo un "prompt" llamando a un endpoint; es una tubería (pipeline) donde el contexto se construye con datos internos o externos, la salida se valida (tipo y forma), se mitigan errores y alucinaciones, y los costos y latencias se mantienen dentro de presupuestos. Por ello, la disciplina de **AI Engineering** complementa a la ciencia de datos y al **ML Engineering** con prácticas de producto, **SRE/DevOps** y seguridad que habilitan un ciclo de vida completo: ideación, prototipado, evaluación, hardening, despliegue, observabilidad y mejora continua. 

En esta ruta, bibliotecas y frameworks como PyTorch, TensorFlow, Hugging Face Transformers y stacks de orquestación de prompts y herramientas facilitan el prototipado y el paso a producción, mientras que utilidades de validación de datos (por ejemplo, modelos de datos tipados) ayudan a exigir salidas estructuradas y a reducir errores de integración.

### 2.  Auge del "AI Engineering" y de modelos de lenguaje a LLMs y FMs

El auge de AI Engineering responde a una tensión conocida: los avances de laboratorio raras veces se traducen, sin fricción, a soluciones de negocio. La primera generación de adopción de IA en empresas se centró en pipelines de ML "clásicos" para tareas cerradas. Con los LLMs y los FMs, muchas organizaciones pueden desplegar valor con menor costo de datos etiquetados, pero se enfrentan a nuevos retos: control de costos por consulta, seguridad en el manejo del contexto, prevención de fugas e inyección de prompts, y planeación de capacidad en escenarios de uso impredecible.

Históricamente, los "language models" eran n-gramas y modelos probabilísticos de baja capacidad (o redes recurrentes) con vocabularios limitados. La arquitectura Transformer habilitó el salto a LLMs con atención paralelizable y contextos crecientes. 

El siguiente escalón con los FMs se generaliza la idea: modelos multimodales o de dominio con capacidad de transferir conocimiento entre tareas y modalidades. En la práctica, la secuencia "modelos de lenguaje -> LLMs -> FMs" acompaña el aumento del tamaño de datos, la diversidad de modalidades y el refinamiento de técnicas de posentrenamiento.

### 3. Casos de uso y cómo evaluarlos (expectativas, hitos, mantenimiento)

El abanico de casos de uso incluye codificación asistida, generación de imágenes y video, redacción y revisión de textos, tutoría y educación personalizada, chatbots y asistentes de búsqueda, agregación/organización de información, automatización de flujos y soporte a la planificación. El rasgo común es que la IA deja de ser un módulo oculto para convertirse en interfaz y core de producto. Aun así, la adopción exige expectativas realistas:

* **Primeros hitos**: un prototipo que resuelve el "happy path" con datos controlados; luego, pruebas con usuarios internos y telemetría sobre utilidad, errores y fricción; después, hardening con validadores, filtros, guardrails y pruebas regresivas.
* **Mantenimiento**: actualización de bases de conocimiento y snapshots de contexto, control de drift en prompts y criterios, recalibración de costos, y "playbooks" ante degradaciones (por ejemplo, caídas de disponibilidad del proveedor o cambios en políticas de rate limits).
* **Evaluación**: combinar tests automatizados con revisiones humanas selectivas. No basta con medir precisión promedio; hay que observar distribuciones, colas de latencia, métricas de rechazo, y, para tareas de generación o decisión, tests de robustez con inputs adversariales.

### 4. La pila de AI Engineering (tres capas)

**Capa de aplicación (experiencias y verificación).** Aquí se diseñan experiencias para personas y sistemas. Se orquesta el contexto (RAG, bases vectoriales, reglas de negocio), se estructuran los prompts (system, developer y user), se llama a herramientas externas (búsqueda, bases, calculadoras, funciones) y se valida la salida. La evaluación es continua: conjuntos de pruebas realistas, revisiones humanas bien muestreadas y telemetría que mida utilidad, seguridad y estabilidad. Un buen front no es solo UI: incluye una línea de verificación previa a entregar a [sistemas aguas abajo](https://medium.com/@ogunodabas/downstream-upstream-system-c1dc6cf4b59e) (validar tipos, esquemas, invariantes).

**Capa de modelo (preentrenar, afinar, optimizar).** Aquí viven la ingeniería de datasets, la elección de arquitecturas, el ajuste supervisado con instrucciones (SFT) y el ajuste por preferencias (PPO/DPO/ORPO, etc.). También se decide la estrategia de optimización para inferencia (cuantización, KV-cache, batching, estrategias de decodificación). Aunque muchos equipos consumen modelos ya entrenados, dominar esta capa permite adaptar comportamientos a dominios específicos, reducir sesgos y mejorar señales de razonamiento.

**Capa de infraestructura (servicio, datos y cómputo).** Proporciona colas, escalado, planeación de capacidad, bases vectoriales para RAG, cachés y observabilidad. Aquí se materializan acuerdos de servicio (latencia objetivo, tasas de éxito, costos por consulta). Una infraestructura bien diseñada evita cuellos de botella y habilita mejoras sin interrumpir el negocio.

**AI Engineering vs ML Engineering vs full-stack.** El ML Engineering tradicional se enfoca en pipelines de datos etiquetados, entrenamiento y despliegue de modelos específicos. AI Engineering incorpora eso y suma: diseño de experiencias, seguridad y gobernanza del contexto, evaluación con jueces automáticos y humanos, y SRE/DevOps orientado a latencia, costo y calidad. 

A diferencia del full-stack clásico, AI Engineering opera con componentes probabilísticos y necesita "líneas de verificación" que controlen la variabilidad de salidas y la interacción con herramientas externas.

