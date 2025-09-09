# make_report.py — genera reporte Markdown mínimo (comentarios en español)
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
    ap = argparse.ArgumentParser(description="Crea docs/reporte.md desde metrics.json.")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--figs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mfile = os.path.join(args.metrics, "metrics.json")
    metrics = open(mfile).read() if os.path.exists(mfile) else "{}"
    report = TEMPLATE.replace("{METRICS}", metrics).format(__import__("datetime").date.today().isoformat())
    with open(os.path.join(args.out, "reporte.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print("[OK] Reporte en docs/reporte.md")
if __name__ == "__main__":
    main()
