import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def build_index_map(mask):
    h, w = mask.shape
    idx = -np.ones((h, w), dtype=int)
    coords = np.argwhere(mask)
    for k, (y, x) in enumerate(coords):
        idx[y, x] = k
    return idx, coords

def neighbors(y, x, h, w):
    vizinhos = []
    for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
        if 0 <= ny < h and 0 <= nx < w:
            vizinhos.append((ny, nx))
    return vizinhos

def solve_channel_illumination(destino, mask, beta=0.2):
    h, w = destino.shape
    
    img_log = np.log1p(destino.astype(np.float64))
    gy, gx = np.gradient(img_log)
    grad_norm = np.sqrt(gy**2 + gx**2)
    avg_grad = np.mean(grad_norm[mask])
    alpha = 0.2 * avg_grad

    idx, coords = build_index_map(mask)
    m = coords.shape[0]

    L_data = []
    L_linha = []
    A_coluna = []
    b = np.zeros(m, dtype=np.float64)

    for k, (y, x) in enumerate(coords):
        N = neighbors(y, x, h, w)
        deg = len(N)

        L_linha.append(k)
        A_coluna.append(k)
        L_data.append(deg)

        ssum = 0.0
        for ny, nx in N:
            j = idx[ny, nx]
            if j != -1:
                L_linha.append(k)
                A_coluna.append(j)
                L_data.append(-1)
            else:
                ssum += img_log[ny, nx]
        
        b[k] += ssum

        for ny, nx in N:
    
            val_p = img_log[y, x]
            val_q = img_log[ny, nx]
            grad_val = val_p - val_q
            
            mag = abs(grad_val)
            
            if mag > 1e-10:
                v = (alpha ** beta) * (mag ** -beta) * grad_val
            else:
                v = 0.0
            
            b[k] += v

    A = sparse.csr_matrix((L_data, (L_linha, A_coluna)), shape=(m, m))
    x_sol = spsolve(A, b)

    out_channel = destino.copy().astype(np.float64)
    for i, (y, xp) in enumerate(coords):
        out_channel[y, xp] = np.expm1(x_sol[i])

    return out_channel

def local_illumination(img, mask, beta=0.2):
    out = np.zeros_like(img)

    for c in range(3):
        out[:, :, c] = solve_channel_illumination(img[:, :, c], mask, beta)

    return np.clip(out, 0, 255).astype(np.uint8)