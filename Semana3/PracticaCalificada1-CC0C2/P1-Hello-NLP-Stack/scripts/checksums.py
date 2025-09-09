# checksums.py -manifiesto SHA256 
import argparse, hashlib, os  # Importación de módulos necesarios

# Función para calcular el hash SHA256 de un archivo
def sha256_of(path, chunk=1<<20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)  # Lee el archivo en bloques (por defecto, 1MB)
            if not b: break    # Termina si no hay más datos
            h.update(b)        # Actualiza el hash con el bloque leído
    return h.hexdigest()       # Devuelve el hash en formato hexadecimal

def main():
    ap = argparse.ArgumentParser()  # Analizador de argumentos de línea de comandos
    ap.add_argument("--root", default="data", help="Directorio raíz a hashear")
    ap.add_argument("--out", default="evidencias/data_sha256.txt", help="Ruta del manifiesto")
    args = ap.parse_args()

    records = []  # Lista para almacenar tuplas (ruta, hash)
    for r, _, files in os.walk(args.root):  # Recorre recursivamente los archivos desde el directorio raíz
        for fn in files:
            if fn.endswith((".csv", ".conllu", ".jsonl", ".txt")):  # Filtra por extensiones relevantes
                p = os.path.join(r, fn)  # Construye la ruta completa
                records.append((p, sha256_of(p)))  # Calcula y guarda el hash
    records.sort()  # Ordena las entradas alfabéticamente

    os.makedirs(os.path.dirname(args.out), exist_ok=True)  # Crea el directorio de salida si no existe
    with open(args.out, "w", encoding="utf-8") as f:  # Abre el archivo de salida
        for p, h in records:
            f.write(f"{h}  {p}\n")  # Escribe cada línea: hash + ruta
    print(f"[OK] {len(records)} archivos con hashes -> {args.out}")  # Mensaje de confirmación

if __name__ == "__main__":
    main()  # Punto de entrada del script
