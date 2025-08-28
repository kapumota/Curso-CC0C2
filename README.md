### Procesamiento de Lenguaje Natural (CC0C2) 

#### Descripción del curso

El curso Procesamiento de Lenguaje Natural (NLP) es un curso electivo de 4 créditos que explora los fundamentos y aplicaciones avanzadas de NLP y Modelos de Lenguaje de Gran Escala (LLMs). A través de una combinación de teoría (2 horas semanales) y práctica (4 horas semanales), los estudiantes desarrollarán habilidades para diseñar, implementar y optimizar sistemas de IA basados en NLP, utilizando herramientas modernas como HuggingFace, PyTorch y LangChain. 

El curso culmina en un proyecto final donde los estudiantes crean un prototipo de IA inspirado en una idea de startup.

#### Objetivos

- Comprender los fundamentos, arquitecturas y técnicas de entrenamiento de NLP y LLMs.
- Implementar pipelines de NLP utilizando frameworks modernos.
- Diseñar y evaluar modelos para casos de uso reales, aplicando técnicas como RAG, ingeniería de prompts y sistemas basados en agentes.
- Desarrollar proyectos de IA innovadores y presentarlos efectivamente.

#### Estructura del curso
El curso se divide en 5 unidades impartidas a lo largo de 17 semanas (25 de agosto - 20 de diciembre de 2025):

- Fundamentos de NLP e Ingeniería de IA (Semanas 1-4): Introducción a NLP, LLMs, tokenización, n-gramas y herramientas como HuggingFace.
- Técnicas centrales de NLP y Arquitecturas de Modelos (Semanas 5-8): Representaciones vectoriales, RNNs, Transformers y preparación para el examen parcial.
- Arquitecturas avanzadas de LLMs y Evaluación (Semanas 9-12): Transformers, BERT, ajuste fino (LoRA), métricas de evaluación y optimización.
- Aplicaciones prácticas de LLMs (Semanas 13-16): RAG, agentes de IA, ingeniería de prompts y curación de datos.
- Proyecto final y síntesis (Semanas 16-17): RLHF, DPO, optimización de inferencia y presentación de proyectos finales.

#### Metodología

- Teoría (2h/semana, jueves): Clases magistrales sobre fundamentos, arquitecturas y técnicas avanzadas, complementadas con lecturas recomendadas.
- Práctica (4h/semana, sábado): Laboratorios con Jupyter Notebooks para implementar algoritmos y experimentar con modelos.
- Exposiciones: 5 presentaciones (2 grupales, 2 individuales, 1 final) para discutir avances y proyectos.
- Proyecto final: Desarrollo de un prototipo de IA con documentación técnica y presentación tipo pitch.

#### Evaluación

- Sistema G: Evaluación basada en 5 prácticas (P1-P5), examen parcial (EP), examen final (EF) y un examen sustitutorio opcional (ES).  
- Promedio del curso (PC):  
    * PC = [(PP + EP + EF) / 3], donde PP es el promedio de las 4 mejores prácticas.
    * Si se rinde ES: PC = [(PP + máx(EP, EF) + ES) / 3].

Entregables: Jupyter Notebooks (prácticas), exposiciones y proyecto final.

#### Requisitos previos

- Curso CC421 (pre-requisito).
- Conocimientos básicos de Python, aprendizaje automático y álgebra lineal.

#### Herramientas y tecnologías

- Lenguajes y frameworks: Python, HuggingFace, PyTorch, LangChain, LlamaIndex.
- Entorno: Jupyter Notebooks para prácticas y experimentos.


#### Instrucciones de uso de cuadernos 2025-2

#### 1. Preparar la carpeta de trabajo

1. Crea una carpeta (por ejemplo, `nlp-curso`) y coloca en ella:
   - El archivo `Dockerfile`.
   - El archivo `requirements.txt`.
   - Tus cuadernos Jupyter (`.ipynb`) que quieras usar en el curso (si ya los tienes preparados).

2. La estructura de tu carpeta podría ser:
   ```
   nlp-curso/
   ├── Dockerfile
   ├── requirements.txt
   └── notebooks/
       ├── notebook1.ipynb
       └── notebook2.ipynb
   ```
   *Los notebooks pueden estar en la misma carpeta o en una subcarpeta, según tu preferencia.*

#### 2. Construir la imagen Docker

##### 2.1 Desde Docker Desktop en Windows

1. Abre **Docker Desktop** y asegúrate de que esté corriendo correctamente.
2. Abre una terminal en Windows (símbolo del sistema o PowerShell).
3. Navega hasta la carpeta que contiene el `Dockerfile` y el `requirements.txt`. Por ejemplo:
   ```bash
   cd C:\ruta\a\nlp-curso
   ```
4. Ejecuta el comando de construcción de la imagen:
   ```bash
   docker build -t mi-imagen-nlp .
   ```
   Aquí, `-t mi-imagen-nlp` asigna el nombre (`mi-imagen-nlp`) a la imagen que se creará y el `.` indica el contexto de construcción actual (la carpeta donde está el Dockerfile).

##### 2.2 Desde línea de comandos en Linux

1. Asegúrate de tener Docker instalado y ejecutándose.
2. Abre una terminal y navega hasta la carpeta con tu `Dockerfile` y `requirements.txt`:
   ```bash
   cd /ruta/a/nlp-curso
   ```
3. Ejecuta:
   ```bash
   docker build -t mi-imagen-nlp .
   ```
   *Igual que en Windows, `-t` especifica el nombre de la imagen.*


#### 3. Ejecutar el contenedor y acceder a JupyterLab

##### 3.1 Indicaciones generales

1. Para que puedas editar y guardar los cuadernos desde tu máquina (y no sólo dentro del contenedor), es recomendable montar la carpeta de notebooks en el contenedor. Esto se hace con la opción `-v` (o `--volume`).
2. Expondremos el puerto 8888 (tal como se definió en el Dockerfile) y lo mapearemos al puerto 8888 de la máquina anfitriona. Esto se hace con la opción `-p 8888:8888`.

##### 3.2 Ejemplo de comando:

```bash
docker run -it --rm \
    -p 8888:8888 \
    -v /ruta/a/nlp-curso/notebooks:/home/jovyan/work \
    mi-imagen-nlp
```

- `-it`: modo interactivo con pseudo-TTY (para poder ver la salida y, si es necesario, entrar en bash).
- `--rm`: para eliminar el contenedor al salir (deja tu disco limpio).
- `-p 8888:8888`: mapea el puerto interno 8888 del contenedor al mismo puerto en tu máquina.
- `-v /ruta/a/nlp-curso/notebooks:/home/jovyan/work`: monta la carpeta local con los notebooks en la ruta `/home/jovyan/work` dentro del contenedor (que es donde JupyterLab, por defecto, ubica los archivos de trabajo).
   - En Windows (PowerShell), la ruta local podría verse así: `C:\ruta\a\nlp-curso\notebooks:/home/jovyan/work`
   - En Linux, algo como: `/home/usuario/nlp-curso/notebooks:/home/jovyan/work`

Al ejecutar este comando, verás en la terminal la salida de JupyterLab, que mostrará una URL con un token de acceso (por ejemplo, `http://127.0.0.1:8888/?token=...`). 

##### 3.3 Acceder a JupyterLab

1. Copia y pega la URL que se muestra en la terminal en tu navegador (por ejemplo, `http://127.0.0.1:8888/?token=<tu-token>`).
2. Verás el entorno de JupyterLab en tu navegador, con los cuadernos que tengas en la carpeta montada (`/home/jovyan/work`).


#### 4. Uso en Docker Desktop (Windows) de forma gráfica

Además de la línea de comandos, Docker Desktop en Windows también permite:

1. Ir a la pestaña **Images**.
2. Buscar la imagen que creaste (`mi-imagen-nlp`).
3. Hacer clic en **Run**.
4. Configurar los valores de puerto (8888) y el volumen (montar la carpeta de notebooks) en las opciones gráficas.
5. Hacer clic en **Run** para iniciar el contenedor.

Después, para acceder, repites el proceso de abrir la URL con el token en el navegador.

#### 5. Ajustes y consejos finales

- Si en lugar de `jupyter lab` prefieres usar `jupyter notebook`, puedes cambiar el comando final en el `Dockerfile`, o bien usar `jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser`.  
- Revisa que tu Dockerfile y `requirements.txt` estén en orden y que no haya conflictos de versiones.
- Si requieres instalar más paquetes del sistema (usando `apt-get`) o Python (usando `pip`), añádelos en el Dockerfile antes de exponer tu puerto para que queden en la imagen.




#### Bibliografía recomendada

- Jurafsky & Martin, Speech and Language Processing (2020).
- Philipp Koehn, Statistical Machine Translation (2010).
- Ben Auffarth, Generative AI with LangChain (2023).
- Yoav Goldberg, Neural Network Methods for Natural Language Processing (2017).

Recursos adicionales: Documentación de HuggingFace, artículos de ACL/EMNLP.

#### Cronograma destacado

- Semana 1: Introducción a NLP y LLMs (28 de agosto).
- Semana 8: Examen parcial (18 de octubre).
- Semana 17: Examen final (20 de diciembre) y examen sustitutorio (27 de diciembre).
- Feriados: 30 de agosto y 1 de noviembre (sin clases).
