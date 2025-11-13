### **Ejercicios 10 CC0C2**

#### Ejercicios - Cuaderno: **Preparación_datos_BERT**

1. ¿Qué pasos concretos se siguen en el cuaderno desde el texto crudo hasta obtener `input_ids`, `token_type_ids` (segment IDs) y `attention_mask` para BERT? Enumera las transformaciones en orden.

   ```python
   from transformers import BertTokenizer

   tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
   max_seq_len = 64

   text_a = "This is the first sentence."
   text_b = "Here comes the second sentence."

   encoded = tokenizer(
       text_a,
       text_b,
       padding="max_length",
       truncation=True,
       max_length=max_seq_len,
       return_tensors="pt",
       return_token_type_ids=True,
       return_attention_mask=True,
   )

   input_ids      = encoded["input_ids"]        # (1, max_seq_len)
   token_type_ids = encoded["token_type_ids"]   # (1, max_seq_len)
   attention_mask = encoded["attention_mask"]   # (1, max_seq_len)
   ```

2. ¿Cuál es la diferencia entre vocabulario del tokenizer, IDs de tokens y tokens especiales como `[CLS]`, `[SEP]`, `[PAD]` y `[MASK]` en el flujo del cuaderno?

   ```python
   vocab_size = tokenizer.vocab_size
   cls_id = tokenizer.cls_token_id
   sep_id = tokenizer.sep_token_id
   pad_id = tokenizer.pad_token_id
   mask_id = tokenizer.mask_token_id

   print("Vocab size:", vocab_size)
   print("CLS:", tokenizer.cls_token, cls_id)
   print("SEP:", tokenizer.sep_token, sep_id)
   print("PAD:", tokenizer.pad_token, pad_id)
   print("MASK:", tokenizer.mask_token, mask_id)
   ```

3. En el contexto del cuaderno, ¿cómo se construye el conjunto de datos para MLM a partir de una lista de oraciones? Describe el procedimiento de enmascaramiento (porcentajes y reglas).

   ```python
   import torch
   import random

   mlm_probability = 0.15

   def mask_tokens(input_ids, tokenizer, mlm_probability=0.15):
       labels = input_ids.clone()
       probability_matrix = torch.full(labels.shape, mlm_probability)
       special_tokens_mask = [
           tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
           for val in labels.tolist()
       ]
       special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
       probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

       masked_indices = torch.bernoulli(probability_matrix).bool()
       labels[~masked_indices] = -100   # Ignorar en la pérdida

       return input_ids, labels, masked_indices
   ```

4. ¿Qué criterio se usa para decidir qué tokens se enmascaran en el MLM del cuaderno? ¿Cómo se evita enmascarar tokens especiales?

   ```python
   special_tokens_mask = [
       tokenizer.get_special_tokens_mask(seq, already_has_special_tokens=True)
       for seq in input_ids.tolist()
   ]
   special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

   probability_matrix = torch.full(input_ids.shape, mlm_probability)
   probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

   masked_indices = torch.bernoulli(probability_matrix).bool()
   ```

5. Explica la regla "80/10/10" usada para el enmascaramiento de tokens en MLM y cómo se implementa en el código del cuaderno.

   ```python
   masked_indices = masked_indices & (input_ids != tokenizer.pad_token_id)
   labels = input_ids.clone()
   labels[~masked_indices] = -100  # solo calcular pérdida en tokens enmascarados

   # 80% -> [MASK]
   indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
   input_ids[indices_replaced] = tokenizer.mask_token_id

   # 10% -> token aleatorio
   indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
   indices_random = indices_random & masked_indices & ~indices_replaced
   random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
   input_ids[indices_random] = random_words[indices_random]

   # 10% -> mantener token original (no se toca input_ids)
   ```

6. ¿Cómo se generan los pares de oraciones para NSP (`is_next` vs `not_next`) en el cuaderno? Describe el procedimiento para construir pares positivos y negativos.

   ```python
   import random

   def create_nsp_pairs(sentences, neg_ratio=0.5):
       examples = []
       for i in range(len(sentences) - 1):
           a = sentences[i]
           if random.random() < neg_ratio:
               # not next
               b = random.choice(sentences)
               label = 0  # not_next
           else:
               # is next
               b = sentences[i + 1]
               label = 1  # is_next
           examples.append((a, b, label))
       return examples
   ```

7. ¿Qué estructura de datos se utiliza para representar una muestra del dataset combinando MLM y NSP (por ejemplo, diccionario con claves específicas)? Enumera las claves y qué contiene cada una.

   ```python
   sample = {
       "input_ids": input_ids[0],                # Tensor 1D
       "token_type_ids": token_type_ids[0],      # Tensor 1D
       "attention_mask": attention_mask[0],      # Tensor 1D
       "labels_mlm": labels_mlm[0],              # Tensor 1D con -100 y targets
       "labels_nsp": torch.tensor(nsp_label),    # 0 o 1
   }
   ```

8. ¿Cómo se manejan las secuencias que superan `max_seq_length` en el cuaderno (truncamiento)? ¿Qué estrategia se usa con las dos oraciones (A y B)?

   ```python
   def truncate_pair(tokens_a, tokens_b, max_len):
       while len(tokens_a) + len(tokens_b) > max_len:
           if len(tokens_a) > len(tokens_b):
               tokens_a.pop()
           else:
               tokens_b.pop()
       return tokens_a, tokens_b
   ```

9. ¿De qué forma se generan `attention_mask` y `token_type_ids` a partir de `input_ids` en el cuaderno? Explica la lógica básica.

   ```python
   # token_type_ids: 0 para oración A, 1 para oración B
   # attention_mask: 1 donde hay tokens reales, 0 donde hay padding

   sequence = [tokenizer.cls_token_id] + tokens_a + [tokenizer.sep_token_id] + tokens_b + [tokenizer.sep_token_id]
   token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

   padding_length = max_seq_len - len(sequence)
   input_ids = sequence + [tokenizer.pad_token_id] * padding_length
   token_type_ids = token_type_ids + [0] * padding_length
   attention_mask = [1] * len(sequence) + [0] * padding_length
   ```

10. ¿Cómo se construye el `DataLoader` final para BERT en el cuaderno y qué parámetros (batch size, `shuffle`, `collate_fn`, etc.) se configuran para que sea eficiente?

```python
from torch.utils.data import Dataset, DataLoader

class BertPretrainingDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples  # lista de diccionarios

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    keys = batch[0].keys()
    collated = {k: torch.stack([item[k] for item in batch]) for k in keys}
    return collated

dataset = BertPretrainingDataset(samples)
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=False,
)
```

11. ¿Qué papel cumple la función de `collate_fn` definida en el cuaderno y qué operaciones realiza sobre la lista de ejemplos individuales del dataset?

```python
def collate_fn(batch):
    # batch: lista de diccionarios
    batch_input_ids = torch.stack([item["input_ids"] for item in batch])
    batch_token_type_ids = torch.stack([item["token_type_ids"] for item in batch])
    batch_attention_mask = torch.stack([item["attention_mask"] for item in batch])
    batch_labels_mlm = torch.stack([item["labels_mlm"] for item in batch])
    batch_labels_nsp = torch.stack([item["labels_nsp"] for item in batch])

    return {
        "input_ids": batch_input_ids,
        "token_type_ids": batch_token_type_ids,
        "attention_mask": batch_attention_mask,
        "labels_mlm": batch_labels_mlm,
        "labels_nsp": batch_labels_nsp,
    }
```

12. ¿Qué diferencias hay entre la preparación de datos para MLM y para NSP dentro del mismo pipeline del cuaderno? Indica qué partes del código afectan a cada objetivo.

```python
# Parte asociada a NSP (pares A-B + etiqueta is_next/not_next)
sentence_pairs = create_nsp_pairs(sentences)

# Parte asociada a MLM (enmascarado de input_ids y labels_mlm)
input_ids, labels_mlm, masked_indices = mask_tokens(input_ids, tokenizer, mlm_probability)
```

#### Ejercicios - Cuaderno: **Pre-entrenamiento_BERT**

1. ¿Cuáles son los dos objetivos de preentrenamiento implementados en el cuaderno y qué tipo de salida produce el modelo para cada uno (MLM y NSP)?

   ```python
   from transformers import BertForPreTraining

   modelo = BertForPreTraining.from_pretrained("bert-base-uncased")

   outputs = modelo(
       input_ids=input_ids,
       attention_mask=attention_mask,
       token_type_ids=token_type_ids,
       labels=labels_mlm,
       next_sentence_label=labels_nsp,
   )

   prediction_scores = outputs.prediction_logits     # para MLM
   seq_relationship_scores = outputs.seq_relationship_logits  # para NSP
   ```

2. ¿Cómo se construye el modelo BERT usado en el cuaderno (desde la configuración/base preentrenada hasta las cabezas de MLM y NSP)? Describe los componentes principales.

   ```python
   from transformers import BertConfig, BertForPreTraining

   config = BertConfig.from_pretrained("bert-base-uncased")
   config.num_labels = 2  # NSP

   modelo = BertForPreTraining(config)  # incluye BERT + heads MLM/NSP
   ```

3. ¿De qué forma se combinan las pérdidas de MLM y NSP en la función de pérdida total del cuaderno? Explica cómo se calculan y se agregan.

   ```python
   outputs = modelo(
       input_ids=input_ids,
       attention_mask=attention_mask,
       token_type_ids=token_type_ids,
       labels=labels_mlm,
       next_sentence_label=labels_nsp,
   )

   loss = outputs.loss
   loss_mlm = outputs.loss  # si el cuaderno separa, puede usar outputs.prediction_logits
   # Ejemplo manual:
   # loss_mlm = mlm_loss_fn(prediction_scores.view(-1, vocab_size), labels_mlm.view(-1))
   # loss_nsp = nsp_loss_fn(seq_relationship_scores.view(-1, 2), labels_nsp.view(-1))
   # loss_total = loss_mlm + loss_nsp
   ```

4. ¿Qué pasos siguen las épocas de entrenamiento en el cuaderno (loop sobre batches) desde la carga de los datos hasta la actualización de los pesos? Enumera las fases principales.

   ```python
   for epoch in range(num_epochs):
       modelo.train()
       for batch in dataloader:
           batch = {k: v.to(device) for k, v in batch.items()}

           outputs = modelo(
               input_ids=batch["input_ids"],
               attention_mask=batch["attention_mask"],
               token_type_ids=batch["token_type_ids"],
               labels=batch["labels_mlm"],
               next_sentence_label=batch["labels_nsp"],
           )
           loss = outputs.loss

           optimizer.zero_grad()
           loss.backward()
           torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
           optimizer.step()
           scheduler.step()
   ```

5. ¿Cómo se realiza el envío de tensores a GPU/CPU en el cuaderno? ¿Qué variables se mueven explícitamente de dispositivo y en qué punto del código?

   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   modelo.to(device)

   for batch in dataloader:
       batch = {k: v.to(device) for k, v in batch.items()}
       outputs = modelo(**batch)
   ```

6. ¿Qué métricas de evaluación se calculan en la sección de evaluación del cuaderno para MLM y NSP, y cómo se obtienen a partir de las predicciones del modelo?

   ```python
   def compute_nsp_accuracy(logits, labels):
       preds = logits.argmax(dim=-1)
       correct = (preds == labels).sum().item()
       total = labels.size(0)
       return correct / total

   def compute_mlm_accuracy(prediction_scores, labels_mlm):
       masked_positions = labels_mlm != -100
       preds = prediction_scores.argmax(dim=-1)
       correct = (preds[masked_positions] == labels_mlm[masked_positions]).sum().item()
       total = masked_positions.sum().item()
       return correct / max(total, 1)
   ```

7. ¿Cómo se implementa la fase de inferencia en el cuaderno para predecir tokens enmascarados o la relación entre dos oraciones? Describe el flujo desde el texto de entrada hasta la salida decodificada.

   ```python
   modelo.eval()
   with torch.no_grad():
       encoded = tokenizer(
           text_a,
           text_b,
           return_tensors="pt",
           padding="max_length",
           truncation=True,
           max_length=max_seq_len,
       ).to(device)

       outputs = modelo(**encoded)
       prediction_scores = outputs.prediction_logits
       seq_relationship_scores = outputs.seq_relationship_logits

       mlm_preds = prediction_scores.argmax(dim=-1)
       decoded = tokenizer.decode(mlm_preds[0], skip_special_tokens=True)
   ```

8. ¿Qué estrategia de guardado de checkpoints (modelo, optimizador, scheduler) se utiliza en el cuaderno? ¿Qué información se almacena y cómo se puede restaurar más adelante?

   ```python
   checkpoint = {
       "model_state_dict": modelo.state_dict(),
       "optimizer_state_dict": optimizer.state_dict(),
       "scheduler_state_dict": scheduler.state_dict(),
       "epoch": epoch,
   }
   torch.save(checkpoint, "bert_pretraining_checkpoint.pt")

   # Restaurar
   ckpt = torch.load("bert_pretraining_checkpoint.pt", map_location=device)
   modelo.load_state_dict(ckpt["model_state_dict"])
   optimizer.load_state_dict(ckpt["optimizer_state_dict"])
   scheduler.load_state_dict(ckpt["scheduler_state_dict"])
   start_epoch = ckpt["epoch"] + 1
   ```

9. ¿Qué optimizador y scheduler de tasa de aprendizaje se usan en el cuaderno y por qué son adecuados para entrenar BERT? Menciona los parámetros configurados más relevantes.

   ```python
   from transformers import get_linear_schedule_with_warmup

   optimizer = torch.optim.AdamW(modelo.parameters(), lr=5e-5, weight_decay=0.01)

   num_training_steps = len(dataloader) * num_epochs
   num_warmup_steps = int(0.1 * num_training_steps)

   scheduler = get_linear_schedule_with_warmup(
       optimizer,
       num_warmup_steps=num_warmup_steps,
       num_training_steps=num_training_steps,
   )
   ```

10. ¿Cómo se controla el tamaño de batch efectivo en el cuaderno (por ejemplo, gradient accumulation, recorte de gradientes, etc.) y con qué propósito?

```python
gradient_accumulation_steps = 4
optimizer.zero_grad()

for step, batch in enumerate(dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = modelo(**batch)
    loss = outputs.loss
    loss = loss / gradient_accumulation_steps
    loss.backward()

    if (step + 1) % gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

11. ¿De qué manera el cuaderno separa las fases de "entrenamiento", "evaluación" e "inferencia" en el código? Indica qué bloques de código pertenecen a cada fase y qué los diferencia.

```python
# Entrenamiento
def train_one_epoch(modelo, dataloader, optimizer, scheduler, device):
    modelo.train()
    ...

# Evaluación
def evaluate(modelo, dataloader, device):
    modelo.eval()
    with torch.no_grad():
        ...

# Inferencia
def predict_masked_tokens(modelo, tokenizer, text_a, text_b=None):
    modelo.eval()
    with torch.no_grad():
        ...
```

12. ¿Qué cambios se requerirían en el cuaderno para reutilizar el BERT preentrenado en una tarea de clasificación de oraciones (downstream task) usando parte del mismo pipeline ya implementado?

```python
from transformers import BertForSequenceClassification

num_labels = 3  # ejemplo

cls_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
)

# Reutilizar tokenizer y DataLoader, pero ahora definiendo labels de clasificación
outputs = cls_model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    token_type_ids=batch["token_type_ids"],
    labels=batch["labels_cls"],
)
loss_cls = outputs.loss
logits = outputs.logits
```
