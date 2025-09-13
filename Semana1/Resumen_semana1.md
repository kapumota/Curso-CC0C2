### 1. Introducción

La Ingeniería de IA nace en la intersección de modelos fundacionales de gran escala, infraestructura de cómputo elástica y la presión por convertir prototipos en productos confiables. En lenguaje natural, el campo pasó de técnicas dispersas a un enfoque unificado alrededor de modelos preentrenados que se adaptan con instrucciones, preferencias y recuperación aumentada de conocimiento. 

Los modelos fundacionales sirven como base para múltiples tareas en texto, código, imágenes y otras modalidades. Los LLMs son un caso central orientado a texto que habilita respuesta, resumen, traducción, generación y orquestación de herramientas. Una aplicación sólida con FMs no es un prompt aislado. Es un pipeline completo con construcción de contexto, inferencia bajo presupuestos de costo y latencia, validación estructural, resguardos de seguridad y telemetría de extremo a extremo.

### 2. Auge de la Ingeniería de IA

La disciplina surge para cerrar la brecha entre avances de laboratorio y soluciones de negocio. La primera ola de adopción se apoyó en pipelines clásicos de aprendizaje automático. La llegada de LLMs y FMs reduce la dependencia de grandes volúmenes de datos etiquetados y acelera el tiempo a valor, pero introduce nuevos retos. Control del costo por consulta, seguridad del contexto, prevención de inyección o fuga de prompts y planeación de capacidad ante picos impredecibles se vuelven preocupaciones de primer orden. 

La arquitectura Transformer habilitó saltos en escala y en longitud de contexto, y los FMs generalizaron el marco hacia lo multimodal y hacia dominios específicos.

### 3. Casos de uso y evaluación temprana

La lista es amplia. Asistentes de codificación, generación de medios, redacción y revisión, tutoría, búsqueda conversacional, organización de información y automatización de flujos. La IA se convierte en interfaz y en núcleo de producto. El recorrido sano comienza con un prototipo que resuelve el **camino feliz** bajo datos controlados. Continúa con pruebas internas y telemetría de utilidad, errores y fricción. Luego llega el endurecimiento con validadores de formato, filtros de seguridad y pruebas regresivas. El mantenimiento exige actualizar bases de conocimiento y capturas de contexto, vigilar el desplazamiento de prompts y criterios, recalibrar costos y preparar manuales de respuesta ante degradaciones o cambios de políticas.

### 4. La pila de Ingeniería de IA

En la capa de **aplicación** se diseñan experiencias para personas y sistemas. Se orquesta el contexto mediante recuperación aumentada, reglas de negocio y versiones claras de prompts system, developer y user. Se validan salidas contra un contrato estructural antes de tocar sistemas aguas abajo, y se evalúa de forma continua con conjuntos realistas y revisiones humanas bien muestreadas.

En la capa de **modelo** se trabaja la ingeniería de datos, la elección de arquitecturas y el afinado. El ajuste con instrucciones enseña formato y estilo. La optimización por preferencias alinea con valores deseados. La inferencia se optimiza con cuantización, KV cache y estrategias de decodificación.

En la capa de **infraestructura** se materializan colas, escalado, planeación de capacidad, índices vectoriales, cachés y observabilidad con acuerdos de servicio que fijan latencias objetivo, tasas de éxito y costos por solicitud.

La Ingeniería de IA se diferencia de la Ingeniería de ML tradicional porque opera con componentes probabilísticos y necesita líneas de verificación que controlen variabilidad y uso de herramientas. También se diferencia del desarrollo full stack porque exige gobernanza del contexto y evaluación robusta con jueces automáticos y humanos.

### 5. Modelos fundacionales en contexto

Un FM sólido descansa en tres pilares.

**Datos**, **arquitectura** y **posentrenamiento**.

En **datos** se parte de corpora masivos y diversos con deduplicación a gran escala, filtrado de calidad y control de licencias. Se combina evaluación automática con auditorías humanas acotadas. En entornos multilingües se cuida la proporción por idioma y se aplican cuotas que evitan la invisibilización de 
lenguas minoritarias. En dominios sensibles se exige anonimización, trazabilidad de origen y verificación de derechos.

En **arquitectura** predomina la familia Transformer. La diversidad y el currículo importan tanto como el tamaño. Métodos posicionales como RoPE o ALiBi favorecen contextos extensos. Modelos decoder only dominan la generación, mientras que encoder decoder mantienen ventajas en traducción y tareas condicionales. Mezcla de expertos activa subredes especializadas y reduce costo efectivo de inferencia a cambio de complejidades de enrutamiento y servicio.

En **posentrenamiento** el ajuste con instrucciones enseña formatos y procedimientos. La optimización por preferencias refuerza comportamientos deseables. Métodos directos como DPO u ORPO comparan respuestas preferidas frente a descartadas y evitan bucles inestables. La alineación responsable integra datos de seguridad, rechazos con fundamento y políticas de contenido claras.

### 6. Inferencia y decodificación en producción

La calidad de salida depende del **modelo** y de la **estrategia de decodificación**. Si la tarea exige fidelidad y reproducibilidad, conviene un enfoque más **determinista**: temperatura baja, greedy o beam pequeño y, cuando aplique, salidas restringidas por gramáticas o JSON Schema. En tareas creativas u abiertas se regula el equilibrio entre **diversidad** y **coherencia** con temperatura, top-k o top-p y, si hace falta, penalizaciones de repetición.

La **búsqueda contrastiva** (*contrastive search*) mejora la precisión en pasajes complejos porque desincentiva repeticiones y frases calcadas al comparar el siguiente token con el contexto reciente. Suele dar textos más fluidos, aunque añade algo de cómputo. La **decodificación especulativa** (*speculative decoding*) acelera la inferencia: un modelo ligero propone varios tokens y el modelo objetivo los verifica. Cuando la verificación es correcta, se mantiene el comportamiento del modelo grande con menor latencia. El **cómputo en tiempo de inferencia** (*test-time compute*) eleva la calidad generando y re-ordenando múltiples borradores, usando *self-consistency* o pasos de verificación. Aporta exactitud a cambio de más tiempo y presupuesto.

Operativamente, un sistema en producción sano define **límites por ruta de producto**. Esto incluye objetivos de latencia, tope de costo por solicitud, longitud máxima de contexto y políticas de seguridad. Además **registra** por petición la versión del modelo, la plantilla de prompt y los parámetros de decodificación para poder auditar y reproducir. Cuando el entorno presiona, habilita **modos de degradación**: respuestas más cortas, uso de un modelo más pequeño, desactivar *best-of* o reducir temperatura y top-p. Así se protege la experiencia del usuario y el presupuesto sin perder trazabilidad ni control.

### 7. Metodología de evaluación

Evaluar un sistema de IA requiere medir en tres capas complementarias. Primero, la **adecuación estadística** del modelo para saber si aprende bien la distribución del lenguaje. Segundo, el **desempeño en tareas** concretas con criterios apropiados a cada tipo de problema. Tercero, el **rigor del proceso de evaluación**, que incluye control de sesgos, estabilidad en el tiempo y validez estadística. Sin este encuadre, las métricas pueden llevar a conclusiones engañosas.

**Métricas de modelado.** Entropía y cross entropy cuantifican la probabilidad que el modelo asigna a la secuencia correcta. Con log base e, la perplexity es exp(H). Con base 2 es 2^H. Los bits por carácter o por byte miden compresión promedio y sirven para comparar tokenizaciones a nivel de carácter o byte. Para que estas cifras sean comparables mantén fijo el tokenizador, el corpus y la partición de evaluación. Úsalas para contrastar entrenamientos, evaluar cuantización y detectar degradaciones entre checkpoints.

**Preguntas cerradas.** Reporta Exact Match y F1 con reglas de normalización claras. Por ejemplo, pasar a minúsculas y eliminar puntuación y artículos. Promedia por pregunta para no favorecer respuestas largas. Si existen múltiples respuestas correctas, define listas de aceptables o equivalencias canónicas.

**Código y uso de herramientas.** Prioriza la **corrección funcional**. Compila o ejecuta en un entorno aislado con límites de tiempo y memoria. Define un conjunto de pruebas y reporta porcentaje que pasa, errores de compilación, fallos en ejecución y violaciones del contrato de salida. Para llamadas a herramientas mide tasa de respuestas 2xx, validez de parámetros y coherencia entre la salida de la herramienta y la consulta original.

**Similitud con referencias y jueces automáticos.** ROUGE y BLEU son útiles en resumen y traducción, pero se debilitan cuando existen muchas redacciones válidas. Complementa con ChrF o BERTScore y, en tareas abiertas, con **IA como juez** que puntúe utilidad, veracidad o estilo siguiendo **rúbricas** explícitas. Para reducir sesgos, oculta la identidad del candidato, fija una plantilla estable para el juez, combina varios jueces de familias distintas y añade **anclas humanas** en cada lote para calibrar.

**Rigor estadístico y robustez.** Reporta intervalos de confianza con bootstrap y tamaño de efecto, no solo promedios. Controla **fuga de test** con deduplicación y chequeo de solapados respecto al entrenamiento. Verifica **estabilidad temporal** repitiendo la evaluación en días distintos y con semillas diferentes. Estratifica por tipo de ítem, longitud y dificultad para entender dónde y por qué cambia el desempeño.

### 8. Evaluación del sistema de punta a punta

Un sistema con FMs se evalúa en dos planos complementarios: por componentes para localizar mejoras y en forma end-to-end para confirmar valor real en producción. En la recuperación tipo RAG no basta con mirar recall\@k y MRR. Conviene monitorear la "index freshness", entendida como la edad mediana de los documentos citados, el porcentaje de citas por debajo de un umbral de antigüedad y el tiempo que tarda el índice en reflejar cambios en las fuentes. También ayuda medir cobertura por consulta y duplicados, con SLOs de frescura distintos para colecciones vivas y estáticas. En prompting interesa la validez estructural de la salida contra un contrato como JSON y la robustez ante variaciones mínimas de prompt o contexto, registrando la tasa de reintentos de reparación, las clases de error y la sensibilidad a los parámetros de decodificación.

Cuando el sistema llama herramientas externas, se sigue la tasa de éxito con respuestas válidas, los códigos 2xx, la latencia por herramienta y la corrección funcional de la cadena que incluye la ejecución y la integración del resultado. En el comportamiento del modelo se comparan calidades bajo distintos parámetros de muestreo y semillas para estimar estabilidad, controlando longitud de respuesta, repetición y consistencia factual cuando hay evidencia. En validación se cuantifica el cumplimiento de esquemas, el costo y la latencia de las reparaciones, el número promedio de reintentos y la tasa de fallas que no se pueden recuperar, clasificando errores por campo para focalizar arreglos.

La vista end-to-end conecta todo con métricas de negocio como tiempo ahorrado, tasa de resolución de tickets, conversión y esfuerzo humano residual, además de desglosar la latencia por tramo y el costo por solicitud. Con esos datos se definen puertas de calidad previas a producción que combinan recuperación, formato, seguridad, calidad de tarea y SLOs de latencia y costo. Un ejemplo útil es exigir recall\@5 por encima del valor pactado, JSON válido por encima de 99.5 por ciento, cero fugas de PII en el set adversarial, p95 bajo el objetivo y costo promedio dentro del presupuesto. Si cualquier puerta falla no se promueve el cambio. Operativamente, todo esto se automatiza en CI para reejecutar la batería ante cualquier modificación de modelo, prompts, índices o parámetros, y en producción se usa shadow y canary con telemetría detallada, monitoreo de drift, refresco periódico de índice y embeddings, y bancos de pruebas "vivos" que crecen con casos reales e incidentes.

### 9. Ingeniería de prompts

El aprendizaje en contexto funciona cuando la instrucción es inequívoca, el objetivo está acotado y el contexto aporta evidencia pertinente. En zero-shot conviene declarar con claridad la tarea, las restricciones y el formato esperado. En few-shot se eligen ejemplos que cubran la variedad real de entradas y se explica por qué cada ejemplo es representativo. En tareas compuestas es útil separar en pasos y pedir verificación de cada etapa mediante herramientas externas o comprobaciones internas.

El **system prompt** define identidad, estilo y responsabilidades. El **user prompt** capta intención y datos. Ambos deben versionarse con cambios atómicos para correlacionar variaciones de calidad con modificaciones de plantilla. Se puede añadir un nivel de developer para reglas de interacción con herramientas y formatos internos. La eficiencia del contexto es crítica porque impacta costo y latencia. 

Se resume, se eliminan duplicados y se normalizan formatos. Cuando se requiere veracidad y trazabilidad se usa recuperación aumentada con citas. Las aplicaciones que interactúan con sistemas externos necesitan contratos de salida con JSON o gramáticas y validadores estrictos antes de ejecutar acciones. El prompting defensivo asume adversarialidad. Se detectan intentos de inyección, se aísla el system prompt, se evitan secretos y datos sensibles en plantillas, y se registran versiones y parámetros con cuidado de redacción en logs.

### 10. Recuperación aumentada y agentes

La **recuperación aumentada** ancla la generación a evidencia verificable. El pipeline típico incluye ingesta de documentos, particionado en fragmentos con buena granularidad, generación de embeddings y almacenamiento en un índice vectorial. En la consulta, se ejecuta recuperación **densa**, **léxica** o **híbrida**, se reordena con **re-rankers** y se construye el contexto con citas y metadatos. El reto está en decidir la granularidad de los chunks y el solape. Fragmentos muy grandes meten ruido. Fragmentos mínimos fragmentan las ideas. Un punto de partida eficaz son ventanas del orden de párrafos con solape moderado y una etapa de **fusión de evidencias** que agrupe pasajes relacionados. Mide **recall\@k**, **MRR** y **cobertura por consulta**. Añade la noción de **index freshness** o frescura del índice. Cuantifica qué tan rápido incorporas cambios relevantes y qué edad tienen las fuentes citadas. En dominios vivos, un índice envejecido degrada la calidad aunque el modelo sea excelente.

RAG **multimodal** se vuelve necesario cuando hay tablas, imágenes, planos o audio. Esto añade un paso de extracción para convertir señales no textuales en representaciones consultables y citables. Tablas se normalizan a celdas o filas con metadatos. Imágenes pueden generar descripciones con modelos visuales y añadir coordenadas para referencia. El principio sigue igual. Recupera evidencia, cítala y valida consistencia entre texto y fuente.

Los **agentes** van un paso más allá. No sólo recuperan y redactan. También **planifican** y **actúan** invocando herramientas. El corazón del agente no es el modelo, sino su **orquestación**. Delimita el espacio de acciones, define contratos para cada herramienta y valida parámetros antes de ejecutarlas. Evita bucles contando pasos y detectando ausencia de progreso. Usa **sandboxes** y políticas de aprobación humana para operaciones riesgosas. Mide éxito por tarea, latencia total, coste y errores de herramienta. Evalúa sesgos y seguridad. Un agente útil no hace todo. Hace lo que está permitido, de forma trazable y con controles de daños.

La **memoria** de agentes se divide en corto plazo, largo plazo y episódica. La de corto plazo mantiene el hilo del diálogo. La de largo plazo guarda preferencias u otros hechos persistentes del usuario. La episódica captura el historial de acciones y resultados en una tarea. Define reglas de retención y privacidad. Sin cuidado, una memoria demasiado amplia filtra datos o acumula contradicciones. Con buen diseño reduce fricción y repeticiones.

### 11. Fine-tuning con criterio

El **fine-tuning** tiene sentido cuando el dominio es altamente específico, cuando hay políticas de estilo y seguridad que requieren alineación fina, o cuando hay tareas recurrentes con formato rígido. Si **RAG y prompting** logran la calidad objetivo, quizá no convenga entrenar. El coste no es sólo GPU y tiempo. También es deuda operativa por mantener versiones y rutas de degradación. Evalúa siempre la opción híbrida. Un FM base con RAG y validadores puede cubrir la mayoría de casos. El finetuning queda para cerrar brechas duraderas.

En cuanto a **cuellos de botella de memoria**, el entrenamiento consume memoria para parámetros, gradientes y activaciones. Aquí entran técnicas como **gradiente acumulado**, **checkpointing de activaciones**, **mezcla de precisión** y **offloading** hacia CPU o discos rápidos cuando el hardware lo permite. En finetuning parcial, elige parámetros entrenables. Ajustar todas las capas raras veces es necesario. **Adapters** o **LoRA** y variantes permiten entrenar deltas compactas con buena transferencia. Define una **estrategia de freezing** de capas que preserve conocimiento general y acelere convergencia.

Sobre **representaciones numéricas y cuantización**, reducir precisión a **INT8 o INT4** acorta latencias y reduce memoria. La cuantización **post-training** es rápida, con ligera pérdida de calidad. La cuantización **aware** durante el ajuste suele dar mejor equilibrio en tareas sensibles. Evalúa el impacto contra un banco de pruebas real. No te quedes con una métrica global. Observa colas, casos raros y formatos estrictos.

Para escenarios complejos, la **mezcla de adapters** facilita versiones por dominio. Es útil cuando diferentes equipos comparten un mismo backbone y cada uno inyecta su comportamiento. Evita **olvido catastrófico** con regularización y evalúa capacidades generales tras cada ajuste. En **multitarea**, define pesos de muestreo para que las tareas pequeñas no queden diluidas. Y documenta cada dataset con una **tarjeta** que explique procedencia, licencia, sesgos conocidos y límites.

El plan táctico de finetuning debería incluir curación de instrucciones, balance de datos, criterios de parada y **pruebas de regresión**. Cada nuevo checkpoint pasa por quality gates de formato, seguridad y calidad de tarea. Sin eso, un modelo puede mejorar en un subconjunto y romper otros flujos.

### 12. Ingeniería de datasets

La **calidad** antecede a la **cantidad**. Comienza con una auditoría de fuentes y una política de **deduplicación** que elimine exactos y cercanos. Minimiza ruido con filtros de toxicidad y detección de lenguaje inapropiado. Mide **cobertura** en idiomas, estilos y casos frontera. Piensa en **balance**. Un dataset sobrerrepresentado en tipologías fáciles puede inducir una sensación de calidad falsa.

En **adquisición y anotación**, define guías claras y ejemplos límite. Controla consistencia entre anotadores con acuerdos medidos y revisiones por muestreo. Con datos sensibles, aplica **anonimización** y define reglas de acceso. Cuando trabajas con **preferencias** para alinear estilo o seguridad, la claridad del criterio es tan importante como el tamaño del conjunto.

El **aumento y la síntesis** de datos puede ayudar, pero hay que medir el retorno. **Paráfrasis**, **back-translation** y **perturbaciones controladas** mejoran robustez. La **destilación** desde modelos más grandes acelera la creación de conjuntos instructivos. Monitorea la posible introducción de sesgos o **alucinaciones** si generas datos con el propio modelo. Tu objetivo es mejorar cobertura sin degradar la veracidad.

El **procesamiento** pide pipelines reproducibles. Inspecciona outliers, normaliza, deduplica, limpia y estandariza. En NLP, domina la **tokenización de subpalabras**. Al preparar lotes, diseña **collate functions** que manejen longitudes desiguales con padding correcto y máscaras bien definidas. Esto evita errores sutiles en entrenamiento e inferencia. Define **contratos de formato** para datasets de instrucciones. Un registro consistente de `[instrucción, entrada, salida]` con metadatos de procedencia facilita auditorías y depuración.

No olvides el **data lineage**. Cada ejemplo debe rastrearse a su origen con licencias claras y posibles restricciones de uso. Esto es vital para cumplimiento y para explicar decisiones en contextos regulados.

### 13. Optimización de serving

Operar e inferir con FMs es, ante todo, un problema de colas y de ingeniería de sistemas. Se monitorean cuatro familias de señales: rendimiento en tokens por segundo, latencias p50 p95 y p99, estabilidad temporal frente a picos y costo por mil tokens que suma prompt, respuesta y uso de herramientas. No basta con promedios, hay que vigilar saturaciones y mantener p95 y p99 dentro de objetivos incluso en horas de carga.

En hardware dominan GPUs y TPUs gracias a su memoria de alto ancho de banda y soporte de baja precisión. En entornos on-premise o en el borde conviene combinar cuantización con lotes pequeños y cachés de contexto para amortizar costos. Con contextos muy largos se emplea atención paginada y reciclaje de KV-cache por sesión bajo reglas estrictas de permisos y tiempos de vida. Para reducir arranques en frío se usan colas con prioridades, agrupamiento dinámico continuo y precalentamiento de pesos. Cada ruta de producto opera con un presupuesto por solicitud y cortacircuitos que se activan cuando se excede el límite. Cuando la tasa de aceptación lo permite, la decodificación especulativa acelera manteniendo el comportamiento del modelo objetivo.

La optimización se apoya en tres palancas. Primera, el modelo. Cuantiza a INT8 o INT4 si la caída de calidad es aceptable, usa KV-cache de forma activa y evalúa extensiones de contexto con esquemas posicionales que no degraden los últimos tokens. La poda puede ser útil en el borde, aunque suele requerir reentrenamiento. Segunda, la decodificación. Ajusta temperatura y top-p según el perfil del producto y, en salidas rígidas, aplica decodificación restringida por gramática o validación en bucle que haga cumplir el contrato. Tercera, el serving. Implementa batching dinámico continuo, colas con prioridades y prefetch de tokens cuando la arquitectura lo soporte. Minimiza cold-start con pools de réplicas calientes y define presupuestos claros por solicitud para cortar rutas caras.

Para gobernar el sistema, construye paneles que separen la latencia por tramo del pipeline, recuperación, inferencia y llamadas a herramientas. Sigue la tasa de formato inválido, la tasa de reintentos y el costo medio por ruta de prompt. La caché aporta mucho valor. Usa caché de contexto en conversación y caché de resultados para prompts idénticos cuando la tarea lo permita, con TTL realista e invalidación correcta. En RAG, cachea resultados de recuperación si los índices no han cambiado.

El autoscaling debe reaccionar a latencia y también a la longitud promedio del contexto. Una ola de entradas largas puede romper supuestos si solo miras QPS. Añade circuit breakers que desactiven rutas de alto costo o reduzcan tamaño de respuesta en picos y define modos de degradación, por ejemplo usar un modelo más pequeño o devolver resúmenes. En despliegue, ejecuta shadow con tráfico real sin impacto al usuario y compara outputs y métricas, luego canary con una fracción pequeña y rollback automatizado si p95, formato o seguridad se deterioran. Mantén un banco de pruebas vivo con casos reales y adversariales, reevalúa de forma periódica y verifica que la **index freshness** se mantiene dentro de límites y que embeddings e índices se recalculan con la cadencia acordada.

La suma de estas prácticas produce un sistema que no solo genera texto convincente. Entrega respuestas útiles, verificables y seguras, con costos y latencias predecibles, prompts versionados y defendibles, RAG con evidencia y agentes que actúan dentro de límites claros.

### 14. Gobernanza y trazabilidad

La gobernanza comienza antes de producción y se mantiene durante todo el ciclo de vida.
Define un inventario con versión y estado de cada activo del sistema. Incluye prompts, plantillas, reglas de validación, herramientas externas, modelos, datasets, índices y políticas de seguridad. Para cada elemento registra propietario, fecha de cambio, motivo y evidencia de pruebas que justifican la promoción.

La trazabilidad de datos y licencias se sustenta con registro de procedencia y restricciones de uso. Documenta fuente, tipo de licencia, límites de reutilización, presencia de PII y medidas de anonimización. Mantén tarjetas de dataset y de modelo con alcance, sesgos conocidos, riesgos y exclusiones. Cuando la app cita evidencias guarda el vínculo estable a la versión del documento para poder reproducir resultados.

Define una **política de retención**. Especifica vida útil de cachés de contexto y de resultados, tiempos de expiración de índices y periodicidad de recálculo de embeddings. Toda caché debe tener dueño, TTL claro, reglas de invalidación y métricas de aciertos y errores.

Establece un plan de gestión de cambios. Todo cambio en prompts, modelos, parámetros de decodificación, pipelines o políticas de seguridad pasa por revisión técnica, pruebas automatizadas y aprobación. Usa controles de acceso con principio de mínimo privilegio y doble control para acciones sensibles como rotación de claves.

Estandariza el registro por solicitud. Graba modelo y versión, versión de prompt, parámetros de decodificación, validadores activos, versiones de herramientas, evidencias citadas, latencias por tramo y costo estimado. Aplica redacción y cifrado a cualquier dato sensible. Programa pruebas periódicas de fuga de prompt e inyección, y audita que los logs no contengan secretos ni PII.

Define responsabilidades claras. Producto decide criterios de aceptación, ingeniería aplica controles y SRE vela por SLOs y continuidad. Seguridad y legal supervisan privacidad, licencias y cumplimiento. Reúne a estas funciones en revisiones regulares de riesgo.

### 15. Observabilidad y costo operativo

La observabilidad separa tramos del pipeline para ubicar cuellos de botella.
Recuperación reporta recall@k, MRR, cobertura por consulta y frescura del índice con edad de documentos citados y tiempo de incorporación de cambios. Inferencia monitoriza tasa de formato inválido, tasa de reintentos, longitud de contexto, y latencias p50 p95 p99 con jitter. El uso de herramientas informa proporción de respuestas válidas, códigos de error, coherencia entre resultado y consulta y tiempo medio por herramienta.

Diseña paneles que muestren métricas por ruta de producto. Distingue flujos con y sin RAG, con y sin tool use, y con diferentes contratos de salida. Incorpora trazas distribuídas con IDs de correlación para seguir una solicitud de extremo a extremo. Aplica muestreo inteligente para no sobrecargar almacenamiento y conserva al detalle los eventos de error.

Gestiona el costo con etiquetas por equipo, caso de uso y ruta. Calcula costo por mil tokens, costo por solicitud y costo por tarea completada. Añade el costo de herramientas y de almacenamiento. Publica alertas por desvíos frente a presupuesto y habilita límites por usuario, por equipo y por ruta. La longitud media de contexto es una señal crítica. Una ola de entradas largas puede mantener QPS constante y aun así disparar latencia y gasto. Ajusta el autoscaling con señales de tokens por segundo y longitud de secuencia.

La liberación responsable aplica **sombra** y **canario** de forma escalonada. **Sombra** es un despliegue en paralelo donde la nueva versión procesa el mismo tráfico real que la actual, pero sus respuestas no se muestran al usuario. Sirve para comparar métricas y detectar problemas sin riesgo operativo. Si supera los quality gates, pasa a **canario**, que envía solo una pequeña fracción del tráfico a la nueva versión para observar su comportamiento en producción real y contrastarlo con la versión estable. Si los percentiles de latencia, el formato de salida o los controles de seguridad empeoran, se ejecuta **rollback** automatizado. Estas prácticas se integran con quality gates y con un banco de pruebas vivo que incorpora incidentes reales y casos adversariales para mantener la fiabilidad del sistema a lo largo del tiempo.

### 16. Riesgos y patrones a evitar

Existen señales de alerta conocidas y repetitivas. Confiar en prompts sin validación estructural provoca errores silenciosos que contaminan sistemas aguas abajo. Hacer recuperación sin controlar frescura y cobertura del índice produce respuestas obsoletas o incompletas. Delegar la evaluación en jueces automáticos sin ciego, sin rúbricas claras y sin combinar el veredicto de varios jueces amplifica sesgos y facilita el **gaming**, entendido como la manipulación de la métrica o del juez para obtener mejor puntaje sin mejorar la calidad real. Ejemplos de gaming incluyen escribir con frases o formatos que el juez favorece, saturar de palabras clave, citar de forma superficial o inflar tonos de seguridad y certeza para "convencer" al evaluador, aunque la respuesta sea pobre o inexacta.

Ignorar trazabilidad de versiones impide explicar regresiones y bloquea auditorías. Operar sin modos de degradación ni plan de reversión degrada la experiencia durante picos o incidentes. Mantener cachés sin reglas de expiración ni indicadores de acierto confunde a usuarios con información desactualizada. Exponer herramientas sin contratos de parámetros, sin validación previa y sin zonas seguras abre la puerta a acciones peligrosas o irreversibles.

Otros patrones a evitar. Afinar modelos cuando RAG y buen prompting ya cumplen objetivos introduce deuda operativa sin retorno claro. Cambiar parámetros de decodificación sin control de versión y sin reejecución de pruebas dificulta la reproducción de incidentes. Mezclar datos sensibles en contextos y logs sin redacción ni cifrado vulnera privacidad y cumplimiento. Evaluar con conjuntos poco representativos crea métricas optimistas que no correlacionan con el uso real.

La postura sana es defensa en profundidad. Define contratos de entrada y de salida. Valida y repara antes de ejecutar acciones. Ancla la veracidad con RAG y cita fuentes. Separa permisos por herramienta con listas de permitidos y límites de acción. Observa señales de latencia, costo, formato y seguridad en tiempo real. Ante degradación activa degradaciones controladas y ejecuta rollback. Registra todo con el nivel de detalle necesario para explicar qué pasó, por qué pasó y cómo se corrigió.

