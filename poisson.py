import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from imageio import imread, imwrite
from PIL import Image

import placeMask

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

def solve_channel(target, source, mask, offset, mixed):
    h_t, w_t = target.shape
    h_s, w_s = source.shape
    oy, ox = offset

    idx, coords = build_index_map(mask)
    m = coords.shape[0]

    A_data = []
    A_rows = []
    A_cols = []
    b = np.zeros(m, dtype=np.float64)

    for k, (y, x) in enumerate(coords):
        N = list(neighbors_4(y, x, h_t, w_t))
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
                ssum += target[ny, nx]
        b[k] += ssum

        for ny, nx in N:
            py = y - oy
            px = x - ox
            qy = ny - oy
            qx = nx - ox

            if 0 <= py < h_s and 0 <= px < w_s:
                gp = float(source[py, px])
            else:
                gp = 0.0
            if 0 <= qy < h_s and 0 <= qx < w_s:
                gq = float(source[qy, qx])
            else:
                gq = 0.0

            sp = float(target[y, x])
            sq = float(target[ny, nx])

            if mixed:
                v = (sp - sq) if abs(sp - sq) > abs(gp - gq) else (gp - gq)
            else:
                v = gp - gq

            b[k] += v

    A = sparse.csr_matrix((A_data, (A_rows, A_cols)), shape=(m, m))
    x_sol = spsolve(A, b)

    out = target.copy().astype(np.float64)
    for i, (y, xp) in enumerate(coords):
        out[y, xp] = x_sol[i]

    return np.clip(out, 0, 255).astype(np.uint8)

def poisson_clone(target_rgb, source_rgb, mask_src, offset, mixed=False):
    out = target_rgb.copy().astype(np.float64)

    h_t, w_t = target_rgb.shape[:2]
    h_s, w_s = mask_src.shape
    oy, ox = offset

    mask = np.zeros((h_t, w_t), dtype=bool)

    y0 = max(0, oy)
    x0 = max(0, ox)
    y1 = min(h_t, oy + h_s)
    x1 = min(w_t, ox + w_s)

    sy0 = y0 - oy
    sx0 = x0 - ox
    sy1 = sy0 + (y1 - y0)
    sx1 = sx0 + (x1 - x0)

    mask[y0:y1, x0:x1] = mask_src[sy0:sy1, sx0:sx1]

    for c in range(3):
        out[:, :, c] = solve_channel(
            target_rgb[:, :, c],
            source_rgb[:, :, c],
            mask,
            offset,
            mixed
        )

    return out.astype(np.uint8)

if __name__ == "__main__":
    target = imread("destino.jpg").astype(np.uint8)
    source = imread("fonte.jpg").astype(np.uint8)

    raw_mask = Image.open("mascara.jpg").convert("L")
    mask_src = (np.array(raw_mask) > 127)

    offset = placeMask.escolher_offset(source,target,mask_src)

    out = poisson_clone(target, source, mask_src, offset, mixed=True)
    imwrite("saida.png", out)