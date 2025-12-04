import numpy as np
import cv2
from scipy import sparse
from scipy.sparse.linalg import spsolve

def ensure_rgb(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return img[:, :, :3]
    return img

def build_index_map(mask):
    h, w = mask.shape
    idx = -np.ones((h, w), dtype=int)
    coords = np.argwhere(mask)
    for k, (y, x) in enumerate(coords):
        idx[y, x] = k
    return idx, coords

def neighbors_4(y, x, h, w):
    for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
        if 0 <= ny < h and 0 <= nx < w:
            yield ny, nx

def solve_channel_illumination(img_channel, mask, beta=0.2):
    h, w = img_channel.shape
    
    img_log = np.log1p(img_channel.astype(np.float64))

    gy, gx = np.gradient(img_log)
    grad_norm = np.sqrt(gy**2 + gx**2)
    avg_grad = np.mean(grad_norm[mask])
    alpha = 0.2 * avg_grad

    idx, coords = build_index_map(mask)
    m = coords.shape[0]

    A_data = []
    A_rows = []
    A_cols = []
    b = np.zeros(m, dtype=np.float64)

    for k, (y, x) in enumerate(coords):
        N = list(neighbors_4(y, x, h, w))
        deg = len(N)

        A_rows.append(k)
        A_cols.append(k)
        A_data.append(deg)

        ssum = 0.0
        for ny, nx in N:
            j = idx[ny, nx]
            if j != -1:
                A_rows.append(k)
                A_cols.append(j)
                A_data.append(-1)
            else:
                # Condição de Dirichlet: valor do pixel na borda (no domínio log)
                ssum += img_log[ny, nx]
        
        b[k] += ssum

        # Campo de Orientação (Guidance Field) modificado
        for ny, nx in N:
            # Gradiente original no domínio log (f_p - f_q)
            val_p = img_log[y, x]
            val_q = img_log[ny, nx]
            grad_val = val_p - val_q
            
            # Magnitude do gradiente
            mag = abs(grad_val)
            
            if mag > 1e-10:
                v = (alpha ** beta) * (mag ** -beta) * grad_val
            else:
                v = 0.0
            
            b[k] += v

    A = sparse.csr_matrix((A_data, (A_rows, A_cols)), shape=(m, m))
    x_sol = spsolve(A, b)

    out_channel = img_channel.copy().astype(np.float64)
    for i, (y, xp) in enumerate(coords):
        out_channel[y, xp] = np.expm1(x_sol[i])

    return out_channel

def local_illumination_poisson(img, mask, beta=0.2):
   
    img = ensure_rgb(img) 
    h, w = img.shape[:2]
   
    if mask.shape != (h, w):
        raise ValueError("A máscara deve ter as mesmas dimensões da imagem.")

    out = np.zeros_like(img)

    for c in range(3):
        out[:, :, c] = solve_channel_illumination(img[:, :, c], mask, beta)

    return np.clip(out, 0, 255).astype(np.uint8)