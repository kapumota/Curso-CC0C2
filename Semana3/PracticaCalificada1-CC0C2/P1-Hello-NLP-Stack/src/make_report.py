# make_report.py - produce un reporte en Markdown
import argparse, os, json, datetime
TEMPLATE = """# Reporte de resultados

Fecha: {}

## Métricas
```
{
METRICS}
```

## Figuras
- Matriz de confusión: `out/cm.png` (si aplica)

## Notas
- Describe decisiones, riesgos y próximos pasos.
"""
def main():
    # Definición de argumentos de línea de comandos
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="Directorio que contiene metrics.json")
    ap.add_argument("--figs", required=True, help="Directorio de figuras (out)")
    ap.add_argument("--out", required=True, help="Directorio de salida (por ejemplo, docs/)")
    args = ap.parse_args()

    # Crea el directorio de salida si no existe
    os.makedirs(args.out, exist_ok=True)

    # Construye la ruta al archivo de métricas
    mfile = os.path.join(args.metrics, "metrics.json")

    # Lee las métricas desde el archivo si existe; si no, coloca un objeto vacío
    metrics = open(mfile).read() if os.path.exists(mfile) else "{}"

    # Llena la plantilla con las métricas y la fecha actual
    report = TEMPLATE.replace("{METRICS}", metrics).format(datetime.date.today().isoformat())

    # Escribe el reporte en un archivo Markdown
    with open(os.path.join(args.out, "reporte.md"), "w", encoding="utf-8") as f:
        f.write(report)

    # Mensaje de éxito
    print("[OK] Escribe docs/reporte.md")

# Punto de entrada principal del script
if __name__ == "__main__":
    main()
