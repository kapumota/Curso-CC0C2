### Proyecto 2 - El tiempo en el espacio: Positional Encodings

**Tema:** Sinusoidal, RoPE, ALiBi, extrapolación a largo contexto.

#### Objetivo
Comparar cuatro codificaciones posicionales entrenando a 256 tokens y evaluando a 512/1024.

#### Dataset
Tarea sintética (copiar/duplicar secuencias) + IMDb (longitud variable).

#### Entregables
- Notebook con las 4 variantes.
- Gráficas rendimiento vs longitud, estabilidad numérica (gradientes/pérdida).

#### Métricas
Accuracy (tarea sintética), F1/accuracy (IMDb), pérdida por longitud.

#### Pasos
1) Inyectar cada PE en el mismo modelo  
2) Entrenar corto a 256 tokens; probar a 512/1k  
3) Analizar degradación y justificación técnica

#### Video
Demostración de fallo/éxito por longitud y conclusiones.
