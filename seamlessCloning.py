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

def solve_channel(destino, fonte, mask, offset, mixed):
    alt_destino, lar_destino = destino.shape
    alt_fonte, lar_fonte = fonte.shape
    oy, ox = offset

    idx, coords = build_index_map(mask)
    m = coords.shape[0]

    L_data = []
    L_linha = []
    L_coluna = []
    b = np.zeros(m, dtype=np.float64)

    for k, (y, x) in enumerate(coords):
        N = neighbors(y, x, alt_destino, lar_destino)

        L_linha.append(k)
        L_coluna.append(k)
        L_data.append(len(N))

        ssum = 0.0
        for ny, nx in N:
            j = idx[ny, nx]
            if j != -1:
                L_linha.append(k)
                L_coluna.append(j)
                L_data.append(-1)
            else:
                ssum += destino[ny, nx]
        b[k] += ssum

        for ny, nx in N:
            py = y - oy
            px = x - ox
            qy = ny - oy
            qx = nx - ox

            if 0 <= py < alt_fonte and 0 <= px < lar_fonte:
                gp = float(fonte[py, px])
            else:
                gp = 0.0
            if 0 <= qy < alt_fonte and 0 <= qx < lar_fonte:
                gq = float(fonte[qy, qx])
            else:
                gq = 0.0

            sp = float(destino[y, x])
            sq = float(destino[ny, nx])

            if mixed:
                v = (sp - sq) if abs(sp - sq) > abs(gp - gq) else (gp - gq)
            else:
                v = gp - gq

            b[k] += v

    A = sparse.csr_matrix((L_data, (L_linha, L_coluna)), shape=(m, m))
    x_sol = spsolve(A, b)

    out = destino.copy().astype(np.float64)
    for i, (y, xp) in enumerate(coords):
        out[y, xp] = x_sol[i]

    return np.clip(out, 0, 255).astype(np.uint8)

def seamless_clone(destino, fonte, mascara, offset, mixed=False):    
    out = destino.copy().astype(np.float64)

    alt_destino, lar_destino = destino.shape[:2]
    alt_fonte, lar_fonte = mascara.shape
    oy, ox = offset

    mask = np.zeros((alt_destino, lar_destino), dtype=bool)

    y0 = max(0, oy)
    x0 = max(0, ox)
    y1 = min(alt_destino, oy + alt_fonte)
    x1 = min(lar_destino, ox + lar_fonte)

    sy0 = y0 - oy
    sx0 = x0 - ox
    sy1 = sy0 + (y1 - y0)
    sx1 = sx0 + (x1 - x0)

    if sy1 > sy0 and sx1 > sx0:
        mask[y0:y1, x0:x1] = mascara[sy0:sy1, sx0:sx1]

    if not np.any(mask): return destino

    for c in range(3):
        out[:, :, c] = solve_channel(
            destino[:, :, c],
            fonte[:, :, c],
            mask,
            offset,
            mixed
        )

    return out.astype(np.uint8)
