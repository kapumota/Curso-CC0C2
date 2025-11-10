### Proyecto 1 - Rayos X del Transformer
**Tema:** Fundamentos (MHSA, FFN, residual, Norm, máscaras).

#### Objetivo
Implementar un bloque Transformer mínimo y demostrar el uso de **máscara causal** y **padding**. Visualizar mapas de atención.

#### Dataset
IMDb (binario) o AG News (4 clases).

#### Entregables
- Notebook con bloque Transformer, entrenamiento y visualizaciones.
- Figuras: mapas de atención (≥3 ejemplos), curvas de pérdida/accuracy.
- `metrics/` con JSON final.

#### Métricas
Accuracy/F1 (clasificación), pérdida train/val.

#### Pasos
1) Tokenización + batching con `attention_mask`  
2) MHSA (scaled dot-product) + máscara causal opcional  
3) FFN (GeLU/SiLU), residual, (RMS)Norm, Dropout  
4) Entrenar 3-5 épocas y graficar.

#### Video (~10 min)
Guion sprint -> demo atención -> métricas -> cierre (lecciones aprendidas).
