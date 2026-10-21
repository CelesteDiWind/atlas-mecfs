"""Paso o02: embeddings SPECTER2 (mismo modelo y adapter que el atlas, scripts/03_embed.py) del corpus de OpenAlex
-> openalex/out/E_raw.npy (alineado con works_raw.json; o01b lo recorta a E_specter2.npy). Texto = título + [SEP] + resumen. Reanudable por bloques (out/emb_parts/)."""
import importlib.util
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OA = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(OA))
OUT = os.path.join(OA, "out")
PARTS = os.path.join(OUT, "emb_parts")
os.makedirs(PARTS, exist_ok=True)
spec = importlib.util.spec_from_file_location("embed", os.path.join(ROOT, "scripts", "03_embed.py"))
embed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embed)
BLOCK = 256


def main():
    W = json.load(open(os.path.join(OUT, "works_raw.json"), encoding="utf-8"))
    tok, model = embed.load_specter2()
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    texts = [w["title"] + tok.sep_token + (w["abstract"] or "") for w in W]
    for b0 in range(0, len(texts), BLOCK):
        path = os.path.join(PARTS, f"{b0:06d}.npy")
        if os.path.exists(path) and len(np.load(path)) == min(BLOCK, len(texts) - b0):
            continue  # bloque completo ya calculado (uno incompleto, de una tanda anterior más corta, se rehace)
        embs = []
        with torch.no_grad():
            for i in range(b0, min(b0 + BLOCK, len(texts)), 16):
                batch = tok(texts[i:min(i + 16, b0 + BLOCK)], padding=True, truncation=True, max_length=512, return_tensors="pt")
                embs.append(model(**batch).last_hidden_state[:, 0].numpy())
        np.save(path, np.vstack(embs).astype(np.float32))
        print(f"{min(b0 + BLOCK, len(texts))}/{len(texts)}", flush=True)
    E = np.vstack([np.load(os.path.join(PARTS, f"{b0:06d}.npy")) for b0 in range(0, len(texts), BLOCK)])
    assert len(E) == len(W)
    np.save(os.path.join(OUT, "E_raw.npy"), E)
    print("embeddings:", E.shape)


if __name__ == "__main__":
    main()
