### Proyecto 2 - El tiempo en el espacio: Positional Encodings

**Tema:** Sinusoidal, learned, RoPE, ALiBi, extrapolación a largo contexto.

#### Objetivo
Comparar cuatro codificaciones posicionales entrenando a 256 tokens y evaluando a 512/1024.

#### Dataset
Tarea sintética (copiar/duplicar secuencias) + IMDb (longitud variable).

#### Entregables
- Notebook con las 4 variantes.
- Gráficas rendimiento vs longitud,  estabilidad numérica (gradientes/pérdida).

#### Métricas
Accuracy (tarea sintética), F1/accuracy (IMDb), pérdida por longitud.

#### Pasos
1) Inyectar cada PE en el mismo modelo  
2) Entrenar corto a 256 tokens, probar a 512/1k  
3) Analizar degradación y justificación técnica

#### Video
Demostración de fallo/éxito por longitud y conclusiones.

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
docker build -t p2_positional_encodings:gpu -f Dockerfile.gpu .
docker run --rm -d --gpus all -p 8888:8888 -v $PWD:/workspace p2_positional_encodings:gpu
```
</details>
