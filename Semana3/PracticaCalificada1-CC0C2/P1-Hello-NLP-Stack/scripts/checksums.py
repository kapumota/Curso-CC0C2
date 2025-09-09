# checksums.py — SHA256 manifest (comments in English)
import argparse, hashlib, os

def sha256_of(path, chunk=1<<20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="Root dir to hash")
    ap.add_argument("--out", default="evidencias/data_sha256.txt", help="Manifest path")
    args = ap.parse_args()

    records = []
    for r, _, files in os.walk(args.root):
        for fn in files:
            if fn.endswith((".csv", ".conllu", ".jsonl", ".txt")):
                p = os.path.join(r, fn)
                records.append((p, sha256_of(p)))
    records.sort()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p, h in records:
            f.write(f"{h}  {p}\n")
    print(f"[OK] {len(records)} files hashed → {args.out}")

if __name__ == "__main__":
    main()
