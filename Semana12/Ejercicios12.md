### Ejercicios 12 CC0C2

### 1. Ejercicios - Cuaderno `Adapters_pytorch.ipynb`

#### 1.1. Ejercicio A1 - Entender el *pipeline* de datos

**Contexto:**
Tienen un dataset de texto, un `vocab`, `text_pipeline` y `collate_batch`.

**Enunciado:**

1. Escribe en pseudocódigo los pasos que sigue `collate_batch` desde una lista de `(label, text)` hasta obtener:

   * `labels: Tensor[batch]`
   * `text_padded: Tensor[batch, max_seq_len]`
2. Explica por qué se utiliza `pad_sequence` y qué pasaría si intentaras hacer un `stack` directo de las secuencias.

*(Pregunta conceptual, sin código.)*

#### 1.2. Ejercicio A2 - Completar la clase `FeatureAdapter`

**Contexto:**
El *adapter* es un MLP con cuello de botella y conexión residual: la entrada y la salida tienen dimensión `model_dim`.

**Plantilla:**

```python
import torch
from torch import nn

class FeatureAdapter(nn.Module):
    def __init__(self, bottleneck_size: int = 50, model_dim: int = 100):
        super().__init__()
        # TODO: definir un MLP pequeño con cuello de botella: 
        #       model_dim -> bottleneck_size -> model_dim
        self.bottleneck_transform = nn.Sequential(
            nn.Linear(____, ____),  # TODO: capa de proyección al cuello de botella
            nn.ReLU(),
            nn.Linear(____, ____)   # TODO: capa de regreso a model_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, model_dim]
        returns: [batch, seq_len, model_dim]
        """
        # TODO: aplicar la transformación con cuello de botella
        transformed = self.bottleneck_transform(x)
        # TODO: conexión residual: salida = transformed + x
        output = ____
        return output
```

**Tareas:**

1. Completar los `____`.
2. Probar la clase con un tensor aleatorio `x = torch.randn(32, 20, 100)` y verificar que la salida tiene la misma forma.


#### 1.3. Ejercicio A3 - Envolver una capa `Linear` con `Adapted`

**Contexto:**
Queremos reemplazar `linear1`/`linear2` de cada capa del encoder por una versión con *Adapter*.

**Plantilla:**

```python
class Adapted(nn.Module):
    def __init__(self, linear: nn.Linear, bottleneck_size: int | None = None):
        super().__init__()
        self.linear = linear
        model_dim = linear.out_features

        if bottleneck_size is None:
            # TODO: elegir un tamaño de cuello de botella por defecto
            #       (por ejemplo, la mitad de model_dim)
            bottleneck_size = ____

        # TODO: crear un FeatureAdapter usando model_dim y bottleneck_size
        self.adapter = FeatureAdapter(
            bottleneck_size=____,
            model_dim=____
        )

        # Nota: la idea es aplicar primero la capa lineal original
        #       y luego el adapter (que agrega la parte entrenable extra).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) aplicar la capa lineal original
        h = self.linear(x)
        # 2) aplicar el adapter sobre la salida de la lineal
        h = self.adapter(h)
        return h
```

**Tareas:**

1. Completar el código.
2. Escribir un pequeño script que:

   * Cree un `nn.Linear(100, 100)`.
   * Lo envuelva con `Adapted`.
   * Imprima cuántos parámetros entrenables hay **antes y después** de envolver.


#### 1.4. Ejercicio A4 - Inyectar *Adapters* en el `TransformerEncoder`

**Contexto:**
El modelo `Net` tiene un `self.transformer_encoder` con `num_layers` capas.

**Enunciado (código a completar):**

```python
def inject_adapters_in_encoder(modelo: nn.Module, bottleneck_size: int = 24):
    """
    Reemplaza linear1 y linear2 en cada TransformerEncoderLayer
    por Adapted(linear, bottleneck_size).
    """
    for layer_idx, encoder_layer in enumerate(modelo.transformer_encoder.layers):
        # TODO: envolver encoder_layer.linear1 si existe
        if hasattr(encoder_layer, "linear1") and encoder_layer.linear1 is not None:
            encoder_layer.linear1 = Adapted(____, bottleneck_size=bottleneck_size)
        # TODO: envolver encoder_layer.linear2 si existe
        if hasattr(encoder_layer, "linear2") and encoder_layer.linear2 is not None:
            encoder_layer.linear2 = Adapted(____, bottleneck_size=bottleneck_size)
```

**Tareas:**

1. Completar los `____`.
2. Ejecutar `inject_adapters_in_encoder(modelo)` y verificar, con un `print`, que `type(layer.linear1)` ahora es `Adapted`.


#### 1.5. Ejercicio A5 - Solo entrenar *Adapters*

**Contexto:**
Queremos hacer *fine-tuning* eficiente.

**Enunciado:**

1. Escribe el código necesario para:

   * Congelar todos los parámetros del modelo (`requires_grad=False`).
   * Reactivar solo los parámetros de los *adapters* (puedes identificar los módulos por tipo `Adapted` o `FeatureAdapter`).
2. Modifica el bucle de entrenamiento para imprimir el número total de parámetros y el número de parámetros entrenables.

*(Puedes darles a los estudiantes un esqueleto de función que recorra `named_parameters()` y cuente.)*

### 2. Ejercicios - Cuaderno `LoRA.ipynb`

#### 2.1. Ejercicio L1 - Completar `LoRALayer`

**Contexto:**
LoRA define `ΔW = A B` con *rank* bajo (baja dimensión).

**Plantilla:**

```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        # TODO: inicialización inspirada en el paper: std ~ 1 / sqrt(rank)
        std_dev = 1.0 / torch.sqrt(torch.tensor(rank, dtype=torch.float32))

        # A: [in_dim, rank]
        self.A = nn.Parameter(torch.randn(____, ____) * std_dev)
        # B: [rank, out_dim]
        self.B = nn.Parameter(torch.zeros(____, ____))

        # alpha controla la escala de la actualización LoRA
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, in_dim]
        returns: [batch, out_dim]
        """
        # TODO: implementar x @ A @ B y escalar con alpha
        #       (delta = alpha * (x @ A @ B))
        delta = ____
        return delta
```

**Tareas:**

1. Completar la inicialización y el `forward`.
2. Probar con `x = torch.randn(32, in_dim)` y comprobar la forma de la salida.


#### 2.2. Ejercicio L2 - Completar `LinearWithLoRA`

**Contexto:**
Queremos combinar la salida de la capa lineal original con la corrección LoRA.

**Plantilla:**

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = base_linear
        self.lora = LoRALayer(
            in_dim=____,   # TODO: usar base_linear.in_features
            out_dim=____,  # TODO: usar base_linear.out_features
            rank=rank,
            alpha=alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: combinación de salida base + corrección LoRA
        base = self.linear(x)      # salida de la capa lineal original
        delta = self.lora(x)       # corrección de bajo rango
        return ____                # TODO: devolver la suma base + delta
```

**Tareas:**

1. Rellenar los `in_dim` y `out_dim` a partir de `base_linear`.
2. Devolver la suma correcta.
3. Probar con una capa `nn.Linear(100, 128)` y verificar formas.


#### 2.3. Ejercicio L3 - Congelar y recalibrar el modelo con LoRA

**Contexto:**
Partimos de un `TextClassifier` preentrenado para 4 clases y queremos adaptarlo a 2 clases (sentiment).

**Enunciado:**

1. Escribe el código que:

   * Carga el `state_dict` preentrenado.
   * Congela todos los parámetros (`requires_grad=False`).
   * Reemplaza `fc2` por una nueva capa `nn.Linear(128, 2)` *entrenable*.
   * Reemplaza `fc1` por `LinearWithLoRA(fc1, rank=4, alpha=0.1)`.

2. Muestra:

   * Número total de parámetros del modelo.
   * Número de parámetros entrenables (solo LoRA + nueva `fc2`).


#### 2.4. Ejercicio L4 - Comparar distintos `rank`

**Contexto:**
Queremos ver el efecto del *rank* en número de parámetros y calidad del modelo.

**Enunciado:**

1. Programa un bucle que entrene el modelo con LoRA para distintos valores de `rank` en `fc1`:
   `rank ∈ {2, 4, 8, 16}` (pocas épocas, por ejemplo 1-2 epochs, solo para ver la tendencia general).
2. Para cada `rank`, registra:

   * `num_trainable_params`
   * `valid_accuracy`
3. Presenta los resultados en una pequeña tabla (o `list` de diccionarios) que pueda imprimirse:

```python
results = [
    {"rank": 2, "params": ..., "valid_acc": ...},
    ...
]
```

### 3. Tarea comparativa - Adapters vs LoRA

#### 3.1. Descripción general

**Objetivo:**
Comparar empíricamente **Adapters** y **LoRA** como métodos de *fine-tuning* eficiente sobre el *mismo* modelo base de clasificación de texto.


#### 3.2. Parte 1 - *Setup* común

1. Elijan un dataset de texto (puede ser el mismo de `Adapters_pytorch` o IMDB).
2. Entrenen (o usen pesos dados por el profesor) un **modelo base**:

   * O bien el `Net` con Transformer encoder.
   * O bien `TextClassifier` con GloVe.
3. Guarden el `state_dict` base congelado para usarlo en ambos enfoques.


#### 3.3. Parte 2 - Variante Adapters

1. Inyecten adapters en las capas indicadas (por ejemplo, en todas las capas FFN del encoder).
2. Congelen todos los parámetros originales.
3. Entrenen solo los adapters (y, si se desea, la capa de clasificación).
4. Regístren:

   * `bottleneck_size` usado.
   * Número de parámetros entrenables.
   * *Accuracy* en valid/test.
   * Tiempo aproximado por época.

#### 3.4. Parte 3 - Variante LoRA

1. Apliquen LoRA a las mismas “zonas de impacto” *o a algo comparable*:

   * Si usaron Transformer: a `linear1` y/o `linear2`, o a las proyecciones de atención.
   * Si usaron MLP sencillo: a `fc1` (como en el cuaderno).
2. Congelen el resto del modelo.
3. Entrenen solo LoRA (y capa de salida si corresponde).
4. Regístren los mismos indicadores:

   * `rank` usado.
   * Parámetros entrenables.
   * *Accuracy* en valid/test.
   * Tiempo por época.

#### 3.5. Parte 4 - Informe corto

Entrega un informe (máx. 3-4 páginas) que incluya:

1. **Descripción del *setup***

   * Dataset, modelo base, configuración de entrenamiento (optimizador, LR, epochs).

2. **Tabla comparativa**
   Filas (configuraciones) y columnas como:

 | Método  | Config | Params ent. | Valid Acc | Test Acc | Tiempo/época |
|--------|--------|-------------|-----------|----------|--------------|
| Adapter | b=24  | ...         | ...       | ...      | ...          |
| LoRA    | r=8   | ...         | ...       | ...      | ...          |

3. **Análisis crítico**

   * ¿Qué método da mejor *trade-off* **calidad / parámetros** en su experimento?
   * ¿Cuál fue más fácil de integrar en el código?
   * ¿Qué decisiones de diseño creen que más afectaron el resultado (número de capas con Adapter/LoRA, tamaño de cuello de botella/*rank*, etc.)?


#### 4. Sugerencia de estructura para la plantilla de tarea (archivo `tarea_adapters_lora.py`)

Puedes utilizar este archivo con secciones vacías:

```python
# tarea_adapters_lora.py
# Plantilla base para comparar Adapters vs LoRA en un clasificador de texto

import torch
from torch import nn

# 1) Carga de datos + vocabulario + dataloaders
def build_dataloaders(...):
    """
    Construye y devuelve:
      - train_loader
      - valid_loader
      - test_loader
    A PARTIR del código de los cuadernos.
    """
    # TODO: implementar o reutilizar del cuaderno
    return train_loader, valid_loader, test_loader


# 2) Modelo base
class BaseTextClassifier(nn.Module):
    """
    Modelo base de clasificación de texto.
    Puede ser:
      - Un Transformer encoder (como en Adapters_pytorch),
      - O un MLP con embeddings GloVe (como en LoRA.ipynb).
    """
    # TODO: copiar/adaptar desde el cuaderno correspondiente
    ...


# 3) Adapters
class FeatureAdapter(nn.Module):
    # TODO: Completa la implementación
    ...


class Adapted(nn.Module):
    # TODO: Completa la implementación
    ...


def inject_adapters(modelo: nn.Module, bottleneck_size: int):
    """
    Recorre las capas del modelo y envuelve las capas lineales
    deseadas con Adapted(linear, bottleneck_size).
    """
    # TODO: Completa la implementación
    ...


# 4) LoRA
class LoRALayer(nn.Module):
    # TODO: Completa la implementación
    ...

class LinearWithLoRA(nn.Module):
    # TODO: Completa la implementación
    ...

def inject_lora(modelo: nn.Module, rank: int, alpha: float):
    """
    Recorre las capas del modelo y aplica LinearWithLoRA
    a las capas lineales seleccionadas.
    """
    # TODO: Completa la implementación (decidir dónde aplicar)
    ...

# 5) Utilidades
def count_trainable_params(modelo: nn.Module) -> int:
    """
    Cuenta el número de parámetros entrenables (requires_grad=True)
    en el modelo dado.
    """
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)

def train_model(...):
    """
    Bucle de entrenamiento estándar:
      - Recorre epochs
      - Calcula pérdida en train
      - Evalúa en valid
      - (Opcional) imprime métricas por época
    """
    # TODO: reutilizar del cuaderno; se puede simplificar
    ...


def main():
    """
    Orquestador principal:
      - Construye dataloaders
      - Carga el modelo base
      - Ejecuta experimento con Adapters
      - Ejecuta experimento con LoRA
      - Imprime/resume resultados comparativos
    """
    # TODO: Completa la la lógica de orquestación
    ...


if __name__ == "__main__":
    main()
```
