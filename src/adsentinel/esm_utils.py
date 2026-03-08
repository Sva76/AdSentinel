AdSentinel – ESM-2 embedding wrapper.

NOTE: This module is NOT used in the current pipeline.
ESM embeddings are loaded from pre-computed CSV files (AbSentinel_vectors_1280.csv)
via features.py → load_esm_vectors().

This code is kept as reference for regenerating embeddings from scratch.
Requires: pip install torch fair-esm

Usage:
    embedder = ESMEmbedder()
    vector = embedder.embed("EVQLVESGGGLVQ...")  # returns numpy array (1280,)
"""

# import torch
# import esm
#
#
# class ESMEmbedder:
#     def __init__(self, device=None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#         self.model.eval()
#         self.model.to(self.device)
#         self.batch_converter = self.alphabet.get_batch_converter()
#
#     def embed(self, seq: str):
#         batch = [("seq", seq)]
#         _, _, toks = self.batch_converter(batch)
#         toks = toks.to(self.device)
#
#         with torch.no_grad():
#             out = self.model(toks, repr_layers=[33])
#
#         emb = out["representations"][33][0, 1:-1].mean(0).cpu().numpy()
#         return emb
```
