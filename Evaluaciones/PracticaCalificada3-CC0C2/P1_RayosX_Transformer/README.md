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
4) Entrenar 3-5 épocas y graficar

#### Video (~5 min)
Guion sprint -> demo atención -> métricas -> cierre (lecciones aprendidas).

### Ejecutar con Docker

> Requisitos: Docker Desktop o Docker Engine reciente.

1. **Construir imagen** (una sola vez o cuando cambie `requirements.txt`):
   ```bash
   make build
   ```

2. **Levantar entorno con Jupyter** (mapea el proyecto en `/workspace`):
   ```bash
   make jupyter
   # Abrir en el navegador: http://localhost:8888
   ```

3. **Shell dentro del contenedor** (opcional, para ejecutar scripts):
   ```bash
   make sh
   ```

4. **Detener**:
   ```bash
   make stop
   ```

#### Notas
- Los notebooks y resultados se guardan en tu carpeta local gracias al volumen `-v $PWD:/workspace`.
- Si necesitas GPU, puedes construir desde una imagen CUDA oficial y ejecutar con `--gpus all` (ver ejemplo abajo).

<details><summary>Ejemplo (opcional) con GPU CUDA</summary>

**Dockerfile.gpu** (ejemplo, no obligatorio):
```Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip git curl tini && rm -rf /var/lib/apt/lists/*
RUN useradd -ms /bin/bash appuser
USER appuser
WORKDIR /workspace
COPY --chown=appuser:appuser requirements.txt .
RUN python3 -m pip install --upgrade pip && pip install -r requirements.txt
EXPOSE 8888
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password=''"]
```

**Build & Run**:
```bash
docker build -t p1_rayosx_transformer:gpu -f Dockerfile.gpu .
docker run --rm -d --gpus all -p 8888:8888 -v $PWD:/workspace p1_rayosx_transformer:gpu
```
</details>
