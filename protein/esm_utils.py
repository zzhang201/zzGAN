# gan/protein/esm_utils.py
import torch
import numpy as np
import esm
import tensorflow as tf
from typing import List

# Load once at import time (fast, cached)
_ESM_MODEL, _ESM_ALPHABET, _ESM_LAYER = None, None, None

def _load_esm(model_name: str = "esm2_t6_8M_UR50D"):
    global _ESM_MODEL, _ESM_ALPHABET, _ESM_LAYER
    if _ESM_MODEL is None:
        if model_name == "esm2_t6_8M_UR50D":
            m, A = esm.pretrained.esm2_t6_8M_UR50D(); layer = 6
        elif model_name == "esm2_t12_35M_UR50D":
            m, A = esm.pretrained.esm2_t12_35M_UR50D(); layer = 12
        else:
            m, A = esm.pretrained.esm2_t6_8M_UR50D(); layer = 6
        _ESM_MODEL, _ESM_ALPHABET, _ESM_LAYER = m.eval(), A, layer
    return _ESM_MODEL, _ESM_ALPHABET, _ESM_LAYER

def esm_embed(seqs: List[str],
              model_name: str = "esm2_t6_8M_UR50D",
              batch_size: int = 64,
              device: str = "cpu") -> np.ndarray:
    model, alphabet, layer = _load_esm(model_name)
    model = model.to(device)
    bc = alphabet.get_batch_converter()
    embeddings = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i+batch_size]
        _, _, toks = bc([("seq", s) for s in chunk])
        toks = toks.to(device)
        with torch.no_grad():
            results = model(toks, repr_layers=[layer])
            rep = results["representations"][layer][:, 1:-1, :].mean(1)  # mean-pool, drop CLS/EOS
        embeddings.append(rep.cpu().numpy())
    return np.vstack(embeddings) if embeddings else np.zeros((0, 0))

def esm_mse_loss(gen_seqs: List[str], real_seqs: List[str], **kwargs) -> tf.float32:
    """Simple MSE on mean-pooled embeddings—smooth, fast, and works great as aux loss."""
    E_gen = esm_embed(gen_seqs, **kwargs)
    E_real = esm_embed(real_seqs, **kwargs)
    return tf.reduce_mean(tf.square(E_gen - E_real))

def esm_cosine_loss(gen_seqs: List[str], real_seqs: List[str], **kwargs) -> tf.float32:
    E_gen = tf.nn.l2_normalize(esm_embed(gen_seqs, **kwargs), axis=1)
    E_real = tf.nn.l2_normalize(esm_embed(real_seqs, **kwargs), axis=1)
    return tf.reduce_mean(1 - tf.reduce_sum(E_gen * E_real, axis=1))

def frechet(mu1, C1, mu2, C2, eps=1e-6):
    C1 = C1 + np.eye(C1.shape[0]) * eps
    C2 = C2 + np.eye(C2.shape[0]) * eps
    diff = (mu1 - mu2)
    covmean = sqrtm(C1.dot(C2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float(diff.dot(diff) + np.trace(C1 + C2 - 2 * covmean))

def esm_fid(E_real, E_gen):
    if len(E_real)==0 or len(E_gen)==0: return np.nan
    mu_r, C_r = E_real.mean(0), np.cov(E_real, rowvar=False)
    mu_g, C_g = E_gen.mean(0), np.cov(E_gen, rowvar=False)
    return frechet(mu_r, C_r, mu_g, C_g)

def _fig_to_img(fig, close=True):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    if close: plt.close(fig)
    return img

def esm_umap_image(E_real, E_gen, title="UMAP (ESM) Real vs Gen"):
    if len(E_real)==0 or len(E_gen)==0:
        return None
    mapper = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0)
    Zr = mapper.fit_transform(E_real)
    Zg = mapper.transform(E_gen)
    fig = plt.figure(figsize=(5, 4))
    plt.scatter(Zr[:,0], Zr[:,1], s=6, alpha=0.30, label="Real")
    plt.scatter(Zg[:,0], Zg[:,1], s=8, alpha=0.70, label="Gen")
    plt.legend(loc="best"); plt.title(title); plt.tight_layout()
    return _fig_to_img(fig)