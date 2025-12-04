import cv2
import numpy as np
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



def solve_channel_flatten(target, mask, edge_map):
    h, w = target.shape

    idx, coords = build_index_map(mask)
    m = coords.shape[0]

    A_rows = []
    A_cols = []
    A_data = []
    b = np.zeros(m, dtype=np.float64)

    for k, (y, x) in enumerate(coords):

        N = list(neighbors_4(y, x, h, w))
        deg = len(N)

        A_rows.append(k)
        A_cols.append(k)
        A_data.append(deg)

        rhs = 0.0
        grad_sum = 0.0

        for ny, nx in N:
            j = idx[ny, nx]

            # termo da matriz (vizinhos dentro da máscara)
            if j != -1:
                A_rows.append(k)
                A_cols.append(j)
                A_data.append(-1)
            else:
                # vizinho fora da máscara → fronteira
                rhs += float(target[ny, nx])

            # gradiente V_pq (apenas se estamos em borda)
            if edge_map[y, x] or edge_map[ny, nx]:
                v = float(target[y, x]) - float(target[ny, nx])
                grad_sum += v

        b[k] = rhs + grad_sum

    A = sparse.csr_matrix((A_data, (A_rows, A_cols)), shape=(m, m))
    x_sol = spsolve(A, b)

    out = target.copy().astype(np.float64)
    for i, (y, x) in enumerate(coords):
        out[y, x] = x_sol[i]

    return np.clip(out, 0, 255).astype(np.uint8)


def texture_flattening_poisson(target_rgb, mask_src, canny_thresholds=(100, 200)):
    target_rgb = ensure_rgb(target_rgb)
    out = target_rgb.copy().astype(np.float64)

    h_t, w_t = target_rgb.shape[:2]
    
    mask = mask_src.copy()
    if not np.any(mask): return target_rgb

    gray_target = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_target, canny_thresholds[0], canny_thresholds[1])
    edge_map = edges > 0

    for c in range(3):
        out[:, :, c] = solve_channel_flatten(
            target_rgb[:, :, c],
            mask,
            edge_map
        )

    return out.astype(np.uint8)