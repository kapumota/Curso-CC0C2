### Ejercicios 11 CC0C2

#### `Tutorial-HuggingFaces.ipynb` (tokenización e inferencia con encoders)

#### Ejercicio 1-Tokenizador "Fast" vs "no Fast"

**Contexto:** Ver cómo cambian offsets y máscaras entre versiones del tokenizador.
- **Instrucciones:** Tokeniza 5 frases con `DistilBertTokenizer` y `DistilBertTokenizerFast`. Compara `input_ids`, `attention_mask` y offsets (si existen).
- **Entrega mínima:** Una tabla con una fila por frase y columnas: `modelo`, `coinciden_ids (Sí/No)`, `tiene_offsets (Sí/No)`.
- **Validación rápida:** Los `input_ids` deberían coincidir; solo el "Fast" expone offsets.

#### Ejercicio 2-Padding y atención

**Contexto:** Entender cómo el padding afecta la `attention_mask`.
- **Instrucciones:** Crea un lote con 3 frases de longitudes distintas usando `padding=True` y `truncation=True`.
- **Entrega mínima:** Imprime `input_ids` y `attention_mask`.
- **Validación rápida:** La suma de `attention_mask` por fila debe igualar el número de tokens no-padding.

#### Ejercicio 3-Truncation visible

**Contexto:** Ver el efecto práctico de `max_length`.
- **Instrucciones:** Toma un párrafo largo y tokenízalo con `max_length=16` (con y sin `truncation`).
- **Entrega mínima:** Muestra los primeros 20 `input_ids` y los tokens decodificados.
- **Validación rápida:** Con `truncation=True` la longitud debe ser exactamente 16 tokens.

#### Ejercicio 4-Inferencia de sentimiento básica

**Contexto:** Ejecutar un modelo de clasificación listo para usar.
- **Instrucciones:** Usa `AutoTokenizer` y `AutoModelForSequenceClassification` (RoBERTa de sentimiento). Predice 5 frases (al menos 2 negativas).
- **Entrega mínima:** Tabla con columnas: `texto`, `label_pred`, `prob_max`.
- **Validación rápida:** La probabilidad (`softmax`) por fila debe sumar ≈ 1.

#### Ejercicio 5-"Ablación" de special tokens (demo)

**Contexto:** Ver por qué `[CLS]/[SEP]` importan.
- **Instrucciones:** Fuerza un *input* sin special tokens (remueve manualmente) y predice 3 frases.
- **Entrega mínima:** Comparación de `label_pred` con y sin special tokens.
- **Validación rápida:** Debe cambiar al menos 1 predicción o la confianza.

#### Ejercicio 6-Mini-visualización de atención (opcional si tu modelo lo permite)

**Contexto:** Relacionar atención con palabras importantes.
- **Instrucciones:** Activa `output_attentions=True` y grafica 1 *head* de 1 capa para una frase con negación.
- **Entrega mínima:** Imagen (heatmap) y 2 líneas explicando el patrón.
- **Validación rápida:** Debe apreciarse foco en términos de polaridad/negación.


#### `Modelo_GPT_decodificador.ipynb` (decoder-only y generación causal)

#### Ejercicio 1-Máscara causal triangular

**Contexto:** Confirmar la restricción "solo mira a la izquierda".
- **Instrucciones:** Genera la máscara para secuencia de 8 y muéstrala (matriz).
- **Entrega mínima:** Print o figura de la matriz.
- **Validación rápida:** La parte superior derecha debe ser `-inf` (o valores muy negativos).

#### Ejercicio 2-`x` y `y` desplazados para CLM

**Contexto:** Preparar datos para *next-token prediction*.
- **Instrucciones:** Dada una secuencia `[a,b,c,d]`, construye `x=[a,b,c]` y `y=[b,c,d]`. Hazlo en lote (2 ejemplos).
- **Entrega mínima:** `x`, `y` impresos y una aserción que verifique `y[:-1]==x[1:]`.
- **Validación rápida:** La aserción debe pasar.

#### Ejercicio 3-Forward + pérdida

**Contexto:** Medir la *cross-entropy* del modelo sobre un batch pequeño.
- **Instrucciones:** Pasa un batch por el modelo y calcula la pérdida.
- **Entrega mínima:** Un número (pérdida) y los `logits.shape`.
- **Validación rápida:** `logits.shape == (batch, seq_len, vocab)`.

#### Ejercicio 4-Generación greedy

**Contexto:** Generar texto de forma determinista.
- **Instrucciones:** Implementa una función `generate_greedy(model, prompt, max_new_tokens=20)`.
- **Entrega mínima:** Salida generada para 2 *prompts*.
- **Validación rápida:** La longitud de la salida debe ser `len(prompt_tokens)+20` (si no hay EOS prematuro).

#### Ejercicio 5-Temperature y top-k (simple)

**Contexto:** Controlar diversidad.
- **Instrucciones:** Implementa sampling con `temperature` y `top_k`. Genera desde el mismo *prompt* con `T∈{0.7,1.0}` y `k∈{0,50}`.
- **Entrega mínima:** 4 muestras breves.
- **Validación rápida:** Con `top_k=50` y `T=1.0` se observa más variedad que con greedy.

#### Ejercicio 6-Comparación rápida con GPT-2 (HF)

**Contexto:** Poner en contexto tu generador con un modelo preentrenado.
- **Instrucciones:** Usa `GPT2Tokenizer` y `GPT2LMHeadModel` para el mismo *prompt*.
- **Entrega mínima:** 1 muestra de tu modelo y 1 de GPT-2, lado a lado.
- **Validación rápida:** La salida de GPT-2 debería ser más fluida/coherente.

#### `Fine-tuning-Transformers-HuggingFace.ipynb` (clasificación con BERT)

#### Ejercicio 1-Tokenización consistente

**Contexto:** Evitar errores por shapes distintos.
- **Instrucciones:** Aplica el mismo `tokenizer` con `padding="max_length"` y `truncation=True` a train/val/test.
- **Entrega mínima:** Imprime `input_ids.shape` de 1 batch por split.
- **Validación rápida:** Todas las formas deben coincidir en `seq_len`.

#### Ejercicio 2-DataLoader mínimo

**Contexto:** Conectar dataset -> lotes.
- **Instrucciones:** Crea `DataLoader` para train y test con `batch_size=4`.
- **Entrega mínima:** Imprime el tamaño del primer batch y claves del dict.
- **Validación rápida:** Deben existir `input_ids`, `attention_mask`, `labels`.

#### Ejercicio 3-Paso de entrenamiento

**Contexto:** Recorrido básico del ciclo de entrenamiento.
- **Instrucciones:** Ejecuta 1 época con `AdamW(lr=5e-4)` y un `LambdaLR` lineal a 0.
- **Entrega mínima:** Pérdida promedio de la época y LR final.
- **Validación rápida:** El LR final debe ser ≈ 0; la pérdida debería bajar frente a los primeros pasos.

#### Ejercicio 4-Evaluación y accuracy

**Contexto:** Medir desempeño simple.
- **Instrucciones:** Evalúa en test sin gradientes y calcula `accuracy`.
- **Entrega mínima:** Valor de `accuracy` y conteo de ejemplos.
- **Validación rápida:** `0 ≤ accuracy ≤ 1` y coincide con (aciertos / total).

#### Ejercicio 5-Congelar encoder (prueba rápida)

**Contexto:** Notar el impacto de entrenar solo la cabeza.
- **Instrucciones:** Fija `requires_grad=False` para todas las capas del encoder y entrena 1 época.
- **Entrega mínima:** Tabla con `trainable_params` y `accuracy` obtenida.
- **Validación rápida:** Los parámetros entrenables deben reducirse drásticamente; la `accuracy` suele bajar vs. *full fine-tune*.

#### Ejercicio 6-Guardar y recargar

**Contexto:** Reproducibilidad en inferencia.
- **Instrucciones:** Guarda `state_dict`, recarga en CPU y vuelve a evaluar.
- **Entrega mínima:** `accuracy` antes y después (deben coincidir aproximadamente).
- **Validación rápida:** Diferencia de `accuracy` ≤ 0.01 (variación por redondeo).


**Consejo:** fija una semilla (`random`, `numpy`, `torch`) al inicio de cada cuaderno para minimizar variaciones en resultados.
