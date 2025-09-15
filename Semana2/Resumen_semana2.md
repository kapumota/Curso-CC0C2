### Datasets y DataLoaders para FM y LLM

### 1. Primeros bloques con PyTorch y torchtext

#### Datasets y DataLoaders para FM y LLM

Los modelos fundacionales y los modelos grandes de lenguaje requieren pipelines de datos estables que conviertan texto en tensores listos para entrenamiento o evaluación.

La dupla *Dataset* y *DataLoader* en PyTorch permite aislar la lógica de acceso y transformación de ejemplos respecto del armado de lotes, el ordenamiento, el relleno por longitud y la paralelización de lectura.

Una señal clara del contexto de trabajo es la línea que desactiva los avisos de deprecación y declara las importaciones base. Esto aclara que se busca un flujo didáctico pero consciente de la evolución de la librería.

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
```

Antes de usar utilidades de alto nivel conviene inspeccionar la versión instalada.

```python
import torchtext
print(torchtext.__version__)
```

Esto es útil para dejar evidencia de entorno y reproducibilidad, una práctica clave cuando se entrena o evalúa FM y LLM.

#### Dataset minimalista y DataLoader básico

La unidad básica es un *Dataset* que expone `__len__` y `__getitem__`. Un patrón mínimo que devuelve la oración sin transformar facilita separar la tokenización hacia la función de colación.

```python
# Datos de ejemplo y tamaño de lote
sentences = ["hola mundo", "pytorch con nlp", "datasets y dataloaders"]
batch_size = 2

# Tokenizador y vocabulario con <unk> y <pad>
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(map(tokenizer, sentences), specials=["<unk>","<pad>"])
vocab.set_default_index(vocab["<unk>"])
PAD_ID = vocab["<pad>"]

# Dataset que devuelve texto crudo
class CustomDataset(Dataset):
  def __init__(self, s):
    self.s = s
  def __len__(self):
    return len(self.s)  
  def __getitem__(self, idx):
    return self.s[idx]

custom_dataset = CustomDataset(sentences)
```

Con este *Dataset* es directo crear un *DataLoader* que baraja y entrega lotes. El valor didáctico está en que `__getitem__` no realiza aún conversiones, de modo que la colación puede encargarse del paso a tensores y del padding dinámico. Este desacople es especialmente útil en FM y LLM, ya que permite cambiar la política de tokenización y vocabulario sin reescribir el conjunto de datos.

#### Tokenización y vocabulario con torchtext

Para un flujo inicial se puede usar el tokenizador básico y construir un vocabulario a partir de una lista de oraciones. Con esto se ilustran los conceptos de tokenización y numerización sin introducir dependencias externas.

```python
# Tokenizador
tokenizer = get_tokenizer("basic_english")

# Construye el vocabulario
vocab = build_vocab_from_iterator(map(tokenizer, sentences), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])
PAD_ID = vocab["<pad>"]
```

En proyectos de FM y LLM el vocabulario suele venir de un tokenizador entrenado o predefinido, pero este ejemplo es útil para comprender la mecánica esencial.

#### Colación y padding con batch\_first verdadero y falso

El *DataLoader* puede delegar en una *collate\_fn* la responsabilidad de convertir cada ejemplo en tensor y de homogeneizar longitudes vía *pad\_sequence*. Se muestra primero la versión con *batch\_first* verdadero.

```python
def collate_fn(batch):
    # tokeniza -> numera -> pad -> máscara
    ids = [torch.tensor([vocab[t] for t in tokenizer(x)], dtype=torch.long) for x in batch]
    input_ids = pad_sequence(ids, batch_first=True, padding_value=PAD_ID)
    attention_mask = (input_ids != PAD_ID).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask}

dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

batch = next(iter(dataloader))
itos = vocab.get_itos()
# Decodificado por fila, omitiendo <pad>
decoded = [[itos[int(i)] for i in row[row != PAD_ID]] for row in batch["input_ids"]]
print("Shape:", tuple(batch["input_ids"].shape))
print("Mask shape:", tuple(batch["attention_mask"].shape))
print("Decodificado:", decoded)
```

La variante con *batch\_first* falso conserva la convención tiempo, lote que a veces resulta útil para modelos recurrentes clásicos o para inspección de formas.

```python
def collate_fn_time_first(batch):
    ids = [torch.tensor([vocab[t] for t in tokenizer(x)], dtype=torch.long) for x in batch]
    x_tb = pad_sequence(ids, batch_first=False, padding_value=PAD_ID)  # (T, B)
    return x_tb

dl_tf = DataLoader(custom_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn_time_first)
xb_tb = next(iter(dl_tf))
print("Shape (T, B):", tuple(xb_tb.shape))
```

La estructura modular facilita extender la `collate_fn` con truncado a longitud máxima, construcción de máscaras adicionales o preparación de objetivos desplazados en tareas autoregresivas.

#### Colación que tokeniza y numeriza

Para cerrar el ciclo de texto a tensor en la colación, se incorpora la tokenización con el vocabulario y el relleno por lote. Esta función es un patrón general.

```python
def collate_fn_padmask(batch):
    ids = [torch.tensor([vocab[t] for t in tokenizer(x)], dtype=torch.long) for x in batch]
    input_ids = pad_sequence(ids, batch_first=True, padding_value=PAD_ID)
    attention_mask = (input_ids != PAD_ID).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask}
```

En FM y LLM, la `collate_fn` frecuentemente añade más pasos como truncado a longitud máxima, construcción de máscaras de atención o preparación de objetivos desplazados en tareas autoregresivas. La estructura modular del ejemplo facilita esas extensiones.

La arquitectura mostrada demuestra cómo PyTorch, con apoyo de torchtext, cubre de manera clara la ruta desde texto crudo hasta tensores por lote listos para modelos de lenguaje, tanto en clasificación como en traducción.

En el contexto actual, con torchtext congelado, esta base convive bien con el uso de Hugging Face para tokenización moderna y manejo industrial de datasets, sin perder el control fino que *DataLoader* y una `collate_fn` bien diseñada ofrecen. Con estos bloques, equipos que trabajan con FM y LLM pueden escalar desde prototipos didácticos hasta pipelines robustos, manteniendo legibilidad, reproducibilidad y capacidad de evolución.

### 2. Abstracciones en PyTorch para texto

Dataset define cómo acceder a una muestra en modo map style con `__len__` y `__getitem__` y en modo iterable mediante generadores cuando el origen es infinito o no indexable.

*DataLoader* envuelve el *Dataset* para construir lotes con tamaño configurable, barajar con un *Sampler*, paralelizar lectura con varios procesos y aplicar una función *collate* que transforma listas de ejemplos en tensores. Este desacople permite reemplazar tokenizadores, aplicar normalizaciones y producir máscaras sin tocar la definición del dataset base. En FM y LLM es clave porque la política de subpalabras y la longitud de contexto cambian con frecuencia durante la experimentación.

Factores que más influyen en rendimiento práctico aquí son:

* *collate* dinámico que calcula padding al máximo del lote para no desperdiciar cómputo.
* Máscaras de atención que acompañan al relleno y permiten a los **Transformers** ignorar ceros.
* *samplers* deterministas y distribuidos cuando se usa entrenamiento paralelo.
* Paralelismo con *num\_workers* y *prefetch* para mantener la GPU ocupada.
* Memoria fijada con *pin\_memory* que acelera la transferencia de host a dispositivo.

Ejemplo mínimo de *Dataset map style* y *DataLoader* con *collate* dinámico:

```python
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

PAD_ID = 0  # Nota: este snippet es independiente del vocabulario anterior. Aquí fijamos PAD_ID=0 a modo de ejemplo.

class TextPairs(Dataset):
    def __init__(self, tokenized_pairs):
        self.data = tokenized_pairs  # lista de dicts con input_ids y labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_batch(batch):
    ids = [torch.tensor(x["input_ids"]) for x in batch]
    y = [torch.tensor(x["labels"]) for x in batch]
    ids_padded = pad_sequence(ids, batch_first=True, padding_value=PAD_ID)
    y_padded = pad_sequence(y, batch_first=True, padding_value=-100)
    attention_mask = (ids_padded != PAD_ID).long()
    return {"input_ids": ids_padded, "attention_mask": attention_mask, "labels": y_padded}

loader = DataLoader(
    TextPairs(tokenized_pairs=[]),
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_batch,
    persistent_workers=True
)
```

### 3. Longitudes variables, padding y máscaras

Los lotes de texto tienen longitudes dispares. Padear con *pad\_sequence* produce tensores rectangulares que el modelo puede procesar. Ese relleno se acompaña con *attention\_mask* donde uno marca posiciones válidas y cero señala relleno. En entrenamiento causal se suma una máscara triangular inferior para impedir que cada posición atienda al futuro.

En la práctica muchos modelos implementan esta máscara internamente y basta con suministrar la de padding.

Para contextos muy largos se recomienda truncar a *max\_length* consistente con el *checkpoint* elegido y registrar la distribución de longitudes reales por lote para entender costos de memoria.

### 4. Empaquetado de longitud fija para LLMs causales

El empaquetado por bloques minimiza relleno y maximiza tokens útiles. Se tokeniza el corpus, se concatena la lista de IDs, se recorta al múltiplo de *block\_size* y se parte en ventanas de ese tamaño.

La entrada es la secuencia desplazada a la izquierda y las etiquetas son la misma secuencia desplazada a la derecha en una posición:

```python
def pack_blocks(ejemplos, block_size):
    flat = []
    for ids in ejemplos["input_ids"]:
        flat.extend(ids)
    total_len = (len(flat) // block_size) * block_size
    flat = flat[:total_len]
    blocks = [flat[i:i+block_size] for i in range(0, total_len, block_size)]
    inputs = [torch.tensor(b[:-1]) for b in blocks]
    labels = [torch.tensor(b[1:]) for b in blocks]
    return {"input_ids": inputs, "labels": labels}
```

### 5. Torchtext en estado congelado

Torchtext proporcionó utilidades de tokenización básica, vocabularios y acceso a corpora. Hoy se considera congelado en funciones clave. Hay ciertas recomendaciones cuando un cuaderno depende de torchtext:

* torchtext continúa útil para vocabularios didácticos, pero la ruta de producción debe migrar a Datasets/Tokenizers de Hugging Face.
* Evitar pipelines rígidos acoplados a sus clases.
* Migrar tokenización y carga a Hugging Face Datasets y Tokenizers.
* Preservar PyTorch como motor de entrenamiento sin atar ingestión a objetos de torchtext.

Ejemplo con vocabulario de torchtext y *DataLoader* moderno.

```python
from torchtext.vocab import build_vocab_from_iterator

def yield_tokens(texts):
    for t in texts:
        yield t.lower().split()

vocab = build_vocab_from_iterator(
    yield_tokens(["hola mundo", "mundo nlp", "hola nlp"]),
    specials=["<unk>", "<pad>"]
)
vocab.set_default_index(vocab["<unk>"])

def encode(sample):
    return {"input_ids": [vocab[token] for token in sample.lower().split()]}

dataset = [{"text": "hola mundo"}, {"text": "hola nlp"}]

class PlainText(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return encode(self.items[idx]["text"])
```

Para este dataset (que ya devuelve `input_ids`) usa una `collate_fn` que solo paddea y crea la máscara:

```python
def collate_ids_only(batch):
    ids = [torch.tensor(x["input_ids"]) for x in batch]
    ids_padded = pad_sequence(ids, batch_first=True, padding_value=vocab["<pad>"])
    attention_mask = (ids_padded != vocab["<pad>"]).long()
    return {"input_ids": ids_padded, "attention_mask": attention_mask}

loader = DataLoader(PlainText(dataset), batch_size=2, collate_fn=collate_ids_only)
```

### 6. Hugging Face Datasets y Tokenizers

*Datasets* usa formato columnar **Apache Arrow** que habilita mapeo de memoria, caché y transformaciones encadenadas. Permite cargar colecciones populares o propias, transformar con `map`, filtrar con `filter`, mezclar con `shuffle`, dividir con `train_test_split` y operar en *streaming*.

*Tokenizers* ofrece implementaciones rápidas escritas en Rust que integran con **Transformers**.

Ejemplo que carga, tokeniza y crea *DataLoaders* con *collator* de *padding* dinámico.

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

raw = load_dataset("ag_news")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Si el tokenizer no trae pad_token (p. ej., modelos causales), asigna eos como pad:
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tok(ejemplos):
    return tokenizer(ejemplos["text"], truncation=True)

tokenized = raw.map(tok, batched=True, remove_columns=["text"])

collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(
    tokenized["train"],
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collator
)

valid_loader = DataLoader(
    tokenized["test"],
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=collator
)
```

Agrupado por bloques para entrenamiento causal:

```python
block_size = 1024

def group_texts(ejemplos):
    concat = sum(ejemplos["input_ids"], [])
    total_len = len(concat) // block_size * block_size
    concat = concat[:total_len]
    chunks = [concat[i:i+block_size] for i in range(0, total_len, block_size)]
    inputs = [c[:-1] for c in chunks]
    labels = [c[1:] for c in chunks]
    return {"input_ids": inputs, "labels": labels}

tokenized_text = raw["train"].map(
    lambda e: tokenizer(e["text"]),
    batched=True,
    remove_columns=raw["train"].column_names
)

lm_dataset = tokenized_text.map(group_texts, batched=True)
lm_loader = DataLoader(lm_dataset, batch_size=4, shuffle=True)
```

### 7. Rendimiento y estabilidad del DataLoader

El *DataLoader* puede transformarse en el cuello de botella cuando el modelo acelera con varias GPU o con precisión mixta y la CPU no alcanza a preparar los lotes a la misma velocidad. Para sostener el **throughput** ajusta estos diales de manera informada y mide su efecto en tiempos por lote y utilización de GPU.

Algunos aspectos a tener en cuenta:

**num\_workers**
Incrementa el número de procesos de carga y colación para paralelizar lectura desde disco y armado de tensores. Sube gradualmente hasta que la utilización de GPU se mantenga alta sin que el sistema empiece a hacer swapping o a competir en exceso por CPU.

**persistent\_workers**
Mantén vivos los procesos de trabajo entre épocas para evitar el coste de creación y destrucción de workers. Resulta especialmente útil con datasets grandes y collations no triviales.

**prefetch\_factor**
Haz que cada worker prepare varios lotes por adelantado. Un rango típico de dos a cuatro reduce burbujas de inactividad entre lotes. Ajusta en conjunto con `num_workers` para no sobrecargar memoria.

**pin\_memory y non\_blocking**
Fija la memoria en host y realiza copias a GPU sin bloqueo para acortar la fase de transferencia. Combinar *pin\_memory* con llamadas a [torch.Tensor.to](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html) con el argumento *non\_blocking* suele mejorar la superposición entre cómputo y entrada/salida.

**generator con semillas fijas**
Controla la aleatoriedad del barajado y de cualquier muestreo asociado al *DataLoader*. Esto facilita comparar ejecuciones y depurar diferencias entre experimentos.

**drop\_last según la fase**
En entrenamiento conviene descartar el último lote si queda incompleto para mantener un tamaño de lote constante y un rendimiento más estable. En evaluación y pruebas desactívalo para aprovechar todas las muestras.

> Nota Windows: si usas `num_workers>0`, envuelve el punto de entrada con
>
> ```python
> if __name__ == "__main__":
>     # construcción de loaders / entrenamiento
> ```

Aplica estos ajustes de forma incremental. Observa el radio entre tiempo de preparación de datos y tiempo de cómputo. Si el *DataLoader* sigue siendo el límite simplifica la función *collate*, adelanta la tokenización con pasos previos de mapeo y usa caché en disco cuando sea posible.

```python
g = torch.Generator()
g.manual_seed(1234)

train_loader = DataLoader(
    lm_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=8,
    prefetch_factor=2,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
    generator=g
)
```

### 8. Entrenamiento distribuido y Samplers

En entrenamiento distribuido con **[DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)** cada proceso gestiona su propia GPU y debe consumir un **fragmento exclusivo** del dataset para evitar lecturas duplicadas. Aquí entra **DistributedSampler**, que calcula los índices que verá cada réplica y coordina el barajado entre épocas mediante `set_epoch`.

Pautas clave para que el patrón funcione bien:

* **Un DataLoader por proceso**. Cada proceso crea su propio DataLoader con el mismo dataset pero con el **mismo** `DistributedSampler`.
* **No uses `shuffle=True` si pasas `sampler=`**. El sampler ya controla el barajado.
* **`drop_last=True`** evita desbalances cuando el tamaño del dataset no es divisible por el número de réplicas.
* **Tamaño de lote efectivo**. Es `batch_size × world_size`. Usa acumulación de gradiente si necesitas un global batch mayor sin agotar memoria.
* **Reproducibilidad**. Fija semillas en cada proceso y llama a `sampler.set_epoch(epoch)` para introducir variación por época de forma sincronizada entre todas las réplicas.

Esqueleto mínimo (por proceso):

```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(lm_dataset, shuffle=True, drop_last=True)
loader = DataLoader(lm_dataset, batch_size=8, sampler=sampler, num_workers=8)

for epoch in range(epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        pass
```

### 9. Streaming e IterableDataset

Cuando el corpus es demasiado grande para residir en disco local, `load_dataset(..., streaming=True)` ofrece un iterable que produce ejemplos sobre la marcha. No hay materialización completa ni longitud conocida, por lo que ciertas operaciones como **len** o particiones exactas no están disponibles.

La integración con PyTorch se hace envolviendo ese iterable en un **IterableDataset** simple.

```python
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

stream = load_dataset("oscar", "unshuffled_deduplicated_es", split="train", streaming=True)

class HFStream(IterableDataset):
    def __iter__(self):
        for ex in stream:
            yield ex["text"]

dl = DataLoader(HFStream(), batch_size=32)
```

Buenas prácticas con streaming:

* **Transforma lo costoso fuera del DataLoader** cuando sea posible. Tokeniza en pasos previos o usa funciones ligeras en el iterador.
* **Shuffling con buffer**. En modo streaming el barajado perfecto no existe. Si necesitas aleatoriedad, utiliza barajado por buffer en el origen o intercalado de múltiples fuentes.
* **Backpressure y memoria**. Ajusta `batch_size` y `num_workers` para que el ritmo de producción de ejemplos no sature la RAM.
* **Paralelismo seguro**. Con `IterableDataset`, el *DataLoader* reparte el rango de datos por worker. Asegúrate de que tu iterador sea **reentrante** o independiente por worker para evitar repeticiones.
* **Resiliencia**. El streaming tolera interrupciones mejor que las descargas completas y permite empezar a entrenar mientras aún se leen datos remotos.

### 10. Validación, limpieza y control de calidad

El desempeño de FM y LLM depende de la **calidad de los datos** tanto como de la arquitectura. Antes de alimentar el *DataLoader* conviene establecer una línea de higiene reproducible que capture reglas y métricas.

**Deduplicación**

* Elimina duplicados exactos calculando un hash por documento.
* Reduce casi duplicados con normalizaciones previas para no conservar copias triviales.

**Normalización**

* Homogeneiza codificaciones y aplica **Unicode NFKC** para unificar caracteres equivalentes.
* Limpia espacios, controla saltos de línea y corrige artefactos frecuentes en scraping.

**Filtros por longitud y estructura**

* Descarta textos demasiado cortos que aportan poca señal y documentos excesivamente largos que rompen presupuestos de contexto.
* Añade reglas por idioma o alfabeto cuando el modelo es monolingüe o cuando el dominio lo requiere.

**Trazabilidad y métricas**

* Registra estadísticas por **fuente**, **tema** y **longitud**.
* Mide proporciones de caracteres no alfanuméricos, URLs y números.
* Versiona el pipeline para reproducir resultados y detectar regresiones.

Con **Hugging Face Datasets** estas operaciones se expresan de forma declarativa con `map` y `filter`, lo que genera cachés reutilizables y **fingerprints** que documentan cada transformación. La secuencia típica es:

1. `map` para normalizar y anotar metadatos como longitud y conteos por tipo de carácter
2. `filter` para aplicar umbrales y reglas de exclusión
3. `train_test_split` o split propio para evitar fuga entre conjuntos por autor, tema o tiempo
4. `shuffle` para romper ordenaciones no deseadas antes de materializar DataLoaders

### 11. Integración con Transformers para entrenamiento

La ruta típica con **Transformers** parte de un *checkpoint* y su tokenizer para heredar el vocabulario y las reglas de segmentación. Después se construyen los *DataLoaders* con un **collator** coherente con la tarea y se entrena con **Trainer** o con un bucle propio cuando se necesita control absoluto.

**Checkpoints y tokenizer**
Elegir el checkpoint alinea el espacio de subpalabras y el tamaño de contexto. Para clasificación es habitual un encoder tipo BERT o RoBERTa. Para resumen o traducción se usa un modelo encoder–decoder como T5 o BART. En causal LM se opta por un decoder como GPT. El tokenizer correspondiente garantiza que los ids generados coincidan con los embeddings del modelo y define el `pad_token_id`, crítico para el *collator* y las máscaras.

**Collators por tarea**

* **DataCollatorWithPadding** para clasificación o NLI. Hace padding dinámico por lote con el `pad_token_id` del tokenizer y entrega `attention_mask`.
* **DataCollatorForSeq2Seq** para tareas encoder–decoder porque aplica padding consistente en entradas y etiquetas y puede manejar truncados asimétricos.
* **DataCollatorForLanguageModeling** para causal LM con enmascaramiento (MLM) o preparación de etiquetas desplazadas si ya empaquetaste bloques.

**Entrenamiento con Trainer**
`TrainingArguments` controla casi todo lo operativo. Los campos de mayor impacto son el tamaño de lote por dispositivo, la estrategia de evaluación y guardado, el número de épocas y el uso de precisión mixta. `fp16=True` reduce memoria y acelera en GPU compatibles. Para modelos recientes se prefiere `bf16=True` en tarjetas que lo soportan por mayor estabilidad. La acumulación de gradientes permite simular un lote global mayor. `max_grad_norm` recorta gradientes; `learning_rate`, `warmup_ratio`/`warmup_steps` y el *scheduler* influyen en la convergencia.

**Callbacks y escalado**
Es común añadir `EarlyStoppingCallback` y `load_best_model_at_end`. Para modelos grandes, `gradient_checkpointing` ahorra memoria a costa de cómputo. Cuando el tamaño ya excede una sola GPU, se puede incorporar **DeepSpeed** o **Fully Sharded Data Parallel** desde `TrainingArguments`.

Esqueleto mínimo:

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

modelo = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    save_steps=1000,
    logging_steps=100,
    num_train_epochs=2,
    fp16=True
)

trainer = Trainer(
    model=modelo,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=collator
)

trainer.train()
```

Para lenguaje causal se sustituyen el modelo por `AutoModelForCausalLM` y el dataset por el empaquetado en bloques. En ese caso las etiquetas suelen llevar `ignore_index=-100` en el padding para que la pérdida no compute sobre el relleno.

### 12. Casos de uso y decisiones de diseño

**Clasificación y NLI**

* Usar `DataCollatorWithPadding` y fijar `max_length` si hay columnas con textos ruidosos o extremadamente largos.
* Equilibrar clases o aplicar ponderaciones si la distribución es desbalanceada.
* Medir métricas macro cuando hay clases minoritarias.

**QA extractivo y resumen abstractivo**
Cuando el documento supera el contexto, se aplican **ventanas deslizantes** con solape para no perder evidencia. En extractivo se generan pares pregunta–contexto y se predicen posiciones de inicio y fin. En abstractivo se controla `max_source_length` y `max_target_length` y se vigila la **razón de truncado**. La decodificación usa *beam search* o muestreo con temperatura y *top-k*/*top-p*.

**Entrenamiento causal**

* Elegir `block_size` de acuerdo con el contexto efectivo del modelo.
* Verificar desplazamiento de etiquetas y `ignore_index` en padding.
* Mezclar dominios o currículos por longitud si las fuentes son muy heterogéneas.

**RAG y chat**
Conserva signos, unidades y URLs. En chat, respeta el **formato de conversación** del checkpoint si lo requiere.

**Multilingüe**
Mide métricas por lengua. Si el objetivo es monolingüe, un checkpoint específico suele rendir mejor a igual tamaño. Normalización y filtrado por alfabeto ayudan a evitar ruido por mezcla de idiomas.

### 13. Reproducibilidad y depuración

**Determinismo básico**
Fija semillas en PyTorch y NumPy y pasa un `generator` con semilla al *DataLoader*. En entornos exigentes, activa algoritmos deterministas (con posible coste).

**Versionado de datos**
Congela versiones de datasets y valida **fingerprints** con *Datasets*. Guarda metadatos de origen y fechas.

**Métricas operativas del DataLoader**
Tiempo por lote, longitud promedio por lote y radio de padding. Úsalas para ajustar `num_workers`, `prefetch_factor`, `pin_memory` y detectar fugas de memoria.

**Aislar fallos**
Reproduce con batch size 1 y sin `shuffle`. Usa `with_indices=True`, `select` o `shard` para localizar ejemplos problemáticos.

**Comprobaciones de sanidad**
Formas coherentes de `input_ids`, `attention_mask`, `labels`; `pad_token_id` correcto; tasa de truncado razonable; pérdida desciende en un *smoke test* de cientos de pasos.

### 14. Migración desde torchtext a Datasets

Migrar hacia **Hugging Face Datasets** + **AutoTokenizer** mejora velocidad, ergonomía y escalabilidad.

**Fase 1.** Conserva el bucle de entrenamiento en PyTorch.
**Fase 2.** Sustituye tokenización de torchtext por `AutoTokenizer.from_pretrained(...)`. Revisa *cased/uncased*, `max_length` efectivos y dominio/idioma.
**Fase 3.** Construye el dataset en Arrow. Usa `map`/`filter` y caché persistente.
**Fase 4.** Usa `DataCollatorWithPadding`/`DataCollatorForSeq2Seq`/`DataCollatorForLanguageModeling` cuando aplique.
**Fase 5.** Collate manual solo para casos especiales (NER alineado, QA extractivo, causal LM empaquetado).

**Verificaciones**
Smoke test corto: formas, truncado/padding, pérdida, métricas por clase/longitud, tiempos por lote antes/después.

### 15. Patrones para datos de gran escala

**Sharding por documento/archivo** para paralelizar `map` con `num_proc` y minimizar contención.
**Intercalado (interleaving) con pesos** para controlar proporciones por dominio/idioma o curricula por dificultad.
**Cacheo en disco** de pasos costosos como la tokenización y estadísticas.
**Validación continua** (proporción de símbolos/URLs/dígitos, Unicode NFKC, detección de idioma, deduplicación exacta y casi-duplicados).
**Telemetría por lote** (longitud media, fracción de padding, idioma, TPS, utilización de GPU) para ajustar `batch_size`, `max_length`, `num_workers` y ventanas deslizantes.

Estos patrones permiten configurar pipelines reproducibles y escalables en FM y LLM, y fortalecen el control operativo sobre la calidad y el rendimiento durante el entrenamiento y la evaluación.
