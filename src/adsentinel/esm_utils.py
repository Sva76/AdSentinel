import torch
import esm
import numpy as np

def esm_embed_sequence(seq):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    batch = [("seq", seq)]
    _, _, toks = batch_converter(batch)

    with torch.no_grad():
        out = model(toks, repr_layers=[33])
    emb = out["representations"][33][0, 1:-1].mean(0).numpy()
    return emb
