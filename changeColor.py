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

def solve_channel_poisson(target, source, mask, idx, coords, m):
    """
    Resolve a equação de Poisson usando floats para source e target para máxima precisão.
    """
    h, w = target.shape

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

        # Gradiente da Origem (Source)
        # IMPORTANTE: source deve ser float aqui para não perder detalhes sutis
        val_p_source = source[y, x]
        
        div_g = 0.0
        for ny, nx in N:
            val_q_source = source[ny, nx]
            div_g += (val_p_source - val_q_source)
        
        b[k] += div_g

        # Condições de Contorno (Target)
        for ny, nx in N:
            j = idx[ny, nx]
            if j != -1:
                A_rows.append(k)
                A_cols.append(j)
                A_data.append(-1)
            else:
                # Target também é acessado como float
                b[k] += target[ny, nx]

    A = sparse.csr_matrix((A_data, (A_rows, A_cols)), shape=(m, m))
    x_sol = spsolve(A, b)

    return x_sol

def poisson_edit_core(target, source, mask):
    h, w = target.shape[:2]
    idx, coords = build_index_map(mask)
    m = coords.shape[0]
    
    if m == 0:
        return target

    # Converter tudo para float64 explicitamente antes do processamento
    target_float = target.astype(np.float64)
    source_float = source.astype(np.float64)
    
    out = target_float.copy()

    for c in range(3):
        sol = solve_channel_poisson(
            target_float[:, :, c], 
            source_float[:, :, c], 
            mask, 
            idx, 
            coords, 
            m
        )
        for i, (y, x) in enumerate(coords):
            out[y, x, c] = sol[i]

    return np.clip(out, 0, 255).astype(np.uint8)

def local_color_change_gray_bg(image, mask):
    image = ensure_rgb(image)
    
    # Prepara destino em cinza
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    destination = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # A source é a imagem original (colorida)
    # Não fazemos clip ou astype aqui, o poisson_edit_core vai tratar como float
    source = image
    
    result = poisson_edit_core(destination, source, mask)
    return result

def local_color_change_recolor(image, mask, multipliers=(0.4, 0.4, 2), dilate_kernel_size=5):
    """
    Aplica recoloração local.
    """
    image = ensure_rgb(image)
    
    # 1. Dilatação da máscara (loose selection)
    # Isso garante que a 'borda' do Poisson seja o fundo, não a borda do objeto
    if dilate_kernel_size > 0:
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    else:
        mask_dilated = mask

    # 2. Criar Source modificada em FLOAT
    # NÃO converter para uint8 aqui! Mantemos em float para preservar gradientes.
    source_float = image.astype(np.float64)
    source_float[:, :, 0] *= multipliers[0]
    source_float[:, :, 1] *= multipliers[1]
    source_float[:, :, 2] *= multipliers[2]
    
    # destination é a imagem original
    destination = image
    
    result = poisson_edit_core(destination, source_float, mask_dilated)
    
    return result