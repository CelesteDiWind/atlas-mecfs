"""Paso 3: embeddings SPECTER2 (base + adapter de proximidad [PRX]) -> data/embeddings.npy
El adapter se aplica a mano (sin el paquete `adapters`), replicando su config Pfeiffer:
    a   = LN(h + x)
    out = LN(Up(ReLU(Down(a))) + h + x)       # h = salida densa del FFN, x = entrada residual
"""
import glob
import json
import os
import types

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")


def load_specter2():
    tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoModel.from_pretrained("allenai/specter2_base")
    adapter_dir = snapshot_download("allenai/specter2")
    sd = torch.load(glob.glob(os.path.join(adapter_dir, "pytorch_adapter.bin"))[0], map_location="cpu")
    for i, layer in enumerate(model.encoder.layer):
        p = f"bert.encoder.layer.{i}.output.adapters.[PRX]."
        down = torch.nn.Linear(768, 48)
        up = torch.nn.Linear(48, 768)
        down.weight.data, down.bias.data = sd[p + "adapter_down.0.weight"], sd[p + "adapter_down.0.bias"]
        up.weight.data, up.bias.data = sd[p + "adapter_up.weight"], sd[p + "adapter_up.bias"]
        out = layer.output
        out.adapter_down, out.adapter_up = down, up

        def forward(self, hidden_states, input_tensor):
            h = self.dropout(self.dense(hidden_states))
            a = self.LayerNorm(h + input_tensor)
            return self.LayerNorm(self.adapter_up(torch.relu(self.adapter_down(a))) + h + input_tensor)

        out.forward = types.MethodType(forward, out)
    model.eval()
    return tok, model


def doc_text(p, sep):
    body = p["abstract"]
    if len(body) < 200:  # sin abstract: usar el comienzo del texto del PDF
        body = (body + " " + " ".join(p.get("_text", "").split())[:2500]).strip()
    return p["title"] + sep + body


def main():
    with open(os.path.join(DATA, "papers.json"), encoding="utf-8") as fh:
        papers = json.load(fh)
    tok, model = load_specter2()
    texts = [doc_text(p, tok.sep_token) for p in papers]
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = tok(texts[i:i + 16], padding=True, truncation=True, max_length=512, return_tensors="pt")
            embs.append(model(**batch).last_hidden_state[:, 0].numpy())
            print(f"{i + 16}/{len(texts)}", flush=True)
    E = np.vstack(embs).astype(np.float32)
    np.save(os.path.join(DATA, "embeddings.npy"), E)
    print("embeddings:", E.shape)


if __name__ == "__main__":
    main()
