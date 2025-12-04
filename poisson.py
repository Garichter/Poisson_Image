import numpy as np
import cv2
from scipy import sparse
from scipy.sparse.linalg import spsolve
from imageio import imread, imwrite
import placeMask

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

def solve_channel_cloning(target, source, mask, offset, mixed):
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

def seamless_clone_poisson(target_rgb, source_rgb, mask_src, offset, mixed=False):
    target_rgb = ensure_rgb(target_rgb)
    source_rgb = ensure_rgb(source_rgb)
    
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

    if sy1 > sy0 and sx1 > sx0:
        mask[y0:y1, x0:x1] = mask_src[sy0:sy1, sx0:sx1]

    if not np.any(mask): return target_rgb

    for c in range(3):
        out[:, :, c] = solve_channel_cloning(
            target_rgb[:, :, c],
            source_rgb[:, :, c],
            mask,
            offset,
            mixed
        )

    return out.astype(np.uint8)

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

if __name__ == "__main__":
    print("--- POISSON IMAGE EDITING ---")
    print("1. Seamless Cloning")
    print("2. Texture Flattening")
    choice = input("Option (1/2): ")

    if choice == '1':
        target = imread("destino.jpg").astype(np.uint8)
        source = imread("fonte.jpg").astype(np.uint8)

        mask_src = placeMask.selecionar_mascara_manual(source)
        offset = placeMask.escolher_offset(source, target, mask_src)
        
        out = seamless_clone_poisson(target, source, mask_src, offset, mixed=True)
        imwrite("saida_clone.png", out)

    elif choice == '2':
        target = imread("fonte.jpg").astype(np.uint8)
        
        mask_src = placeMask.selecionar_mascara_manual(target)
        out = texture_flattening_poisson(target, mask_src)
        imwrite("saida_flatten.png", out)

    print("Done.")