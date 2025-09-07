### **Prueba de entrada para el curso NLP & LLMs**

#### Formato

* **Entrega**:
  Un cuaderno Jupyter (`Prueba_Entrada_NLP.ipynb`) con secciones claras. Cada respuesta debe incluir:

  * Explicación teórica con ejemplos propios.
  * Ejercicios resueltos con texto, pseudocódigo o fragmentos de código (cuando se indique).
  * Reflexiones personales (evitar respuestas copiadas de fuentes externas).

* **Evaluación**:

   * Claridad en las explicaciones.
   * Capacidad de aplicar conceptos a ejemplos sencillos.
   * Creatividad y originalidad (evitar definiciones de manual).
   * Relación con los temas del curso.

#### Día 1 - Fundamentos y datos (4 puntos)

**Ejercicio 1. Conceptual** (2 puntos)
Explica con tus propias palabras:

1. ¿Qué diferencia hay entre *procesar texto* y *comprender lenguaje natural*?
2. Crea un ejemplo donde una tarea puede resolverse con reglas simples (regex) y otra donde eso es insuficiente y se necesita un modelo de lenguaje.

**Ejercicio 2. Práctico** (2 puntos)
En un pseudocódigo muy sencillo, describe cómo cargarías un conjunto de oraciones desde un archivo `.txt`, limpiarías puntuación innecesaria y contarías la frecuencia de las palabras.

* No se acepta copiar código de internet, debe ser **tu versión simplificada**.
* Incluye una pequeña tabla de frecuencias con al menos 5 palabras inventadas o tomadas de un ejemplo propio.


#### Día 2 - Tokenización y Preprocesamiento (4 puntos)

**Ejercicio 3. BPE y subpalabras** (2 puntos)

1. Divide manualmente la palabra `descomposición` en sub-palabras aplicando un razonamiento tipo *BPE* (fusiona pares frecuentes paso a paso). Muestra **al menos 3 fusiones intermedias**.
2. Explica cómo manejarías una palabra inventada como `hipertransductómetro` que no está en tu vocabulario.

**Ejercicio 4. Preprocesamiento mixto** (2 puntos)
Dado el texto:

```
"AI models aprendeN rápido, pero   necesitan   clean_data!!!"
```

* Normaliza el texto para que sea uniforme (criterio propio).
* Escribe 2 versiones:

  1. Versión para un modelo basado en reglas.
  2. Versión para un modelo neuronal moderno. Explica las diferencias.

#### Día 3 - Modelos de lenguaje y evaluación (6 puntos)

**Ejercicio 5. N-gramas** (3 puntos)

1. Construye un modelo **bigramas** con el mini-corpus siguiente:

   ```
   el gato duerme
   el perro ladra
   el gato come
   ```

   * Calcula al menos **3 probabilidades de bigramas** distintas.
2. Explica con un ejemplo propio por qué este modelo tiene limitaciones para textos largos.

**Ejercicio 6. Perplejidad y métricas** (3 puntos)

1. Con tus palabras, explica qué significa que un modelo tenga **alta perplejidad** sobre un texto.
2. Diseña un mini-ejemplo (2 oraciones) donde calcular BLEU dé un valor alto y otro donde dé un valor bajo. No uses ejemplos de manual, inventa frases propias.

#### Día 4 - Puente hacia Transformers (6 puntos)

**Ejercicio 7. Seq2Seq simplificado** (3 puntos)
Imagina que quieres traducir **números en cifras** a **números en palabras** (ej: `15 -> quince`).

1. Explica cómo lo haría un modelo encoder-decoder de manera muy resumida.
2. Propón un ejemplo de entrada y salida y muestra **cómo fallaría un decodificador greedy** frente a uno con beam search (invéntalo, no importa si es realista).

**Ejercicio 8. Reflexión inicial sobre LLMs** (3 puntos)

1. Escribe en un párrafo: ¿qué esperas aprender en este curso que no podrías lograr solo leyendo documentación de HuggingFace?
2. Da un ejemplo real o inventado donde usarías un LLM en tu área de interés, pero con una limitación ética que debas considerar.
