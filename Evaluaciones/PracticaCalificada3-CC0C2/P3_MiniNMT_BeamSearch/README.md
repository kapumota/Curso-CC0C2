# Proyecto 3 — Mini-NMT: Encoder-Decoder y Beam Search
**Tema:** NMT con encoder-decoder; decodificación.

## Objetivo
Fine-tuning pequeño y comparación greedy vs beam (k=1/4/8) con y sin length penalty.

## Dataset
Tatoeba o subset WMT EN↔ES (limpieza mínima).

## Entregables
- Notebook: fine-tuning, inferencia, evaluación.
- Tabla BLEU/chrF y muestras cualitativas.

## Métricas
BLEU, chrF; tiempo por oración.

## Pasos
1) Limpieza/normalización básica  
2) Fine-tune (2–3 épocas)  
3) Decodificar y comparar búsquedas

## Video
Demo traducción + toggles de beam/penalty.

---

## Ejecutar con Docker

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

### Notas
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
docker build -t p3_mininmt_beamsearch:gpu -f Dockerfile.gpu .
docker run --rm -d --gpus all -p 8888:8888 -v $PWD:/workspace p3_mininmt_beamsearch:gpu
```
</details>
