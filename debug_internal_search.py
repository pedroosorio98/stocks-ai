# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 15:39:25 2026

@author: Pedro
"""

import json
import numpy as np
import faiss
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-large"  # MUST match your build_index.py
FAISS_PATH = "index/internal.faiss"
META_PATH = "index/internal_meta.jsonl"

def load_meta():
    metas = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

client = OpenAI()
index = faiss.read_index(FAISS_PATH)
metas = load_meta()

q = "Nubank key products and services"
resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
qv = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
faiss.normalize_L2(qv)  # IMPORTANT for your cosine setup

D, I = index.search(qv, 5)

print("Top internal hits:")
for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
    m = metas[idx]
    txt = m.get("text", "")
    print("\n#", rank, "score=", float(score))
    print("ticker:", m.get("ticker"))
    print("source:", m.get("source"))
    print("type:", m.get("type"))
    print("text_snippet:", (txt[:400] + " ...") if len(txt) > 400 else txt)
