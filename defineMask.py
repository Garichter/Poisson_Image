import numpy as np
from PIL import Image
from collections import deque

def smallest_closed_region(img_path):
    img = Image.open(img_path).convert("L")
    I = np.array(img)

    contour = I < 128        # linhas = barreira
    h, w = contour.shape

    visited = np.zeros((h, w), dtype=bool)
    regions = []             # lista de (tamanho, máscara da região)

    for y in range(h):
        for x in range(w):
            if contour[y, x] or visited[y, x]:
                continue

            q = deque([(y, x)])
            visited[y, x] = True

            region_pixels = []
            touches_border = False

            while q:
                cy, cx = q.popleft()
                region_pixels.append((cy, cx))

                if cy == 0 or cy == h-1 or cx == 0 or cx == w-1:
                    touches_border = True

                for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                    if 0 <= ny < h and 0 <= nx < w:
                        if visited[ny, nx] or contour[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        q.append((ny, nx))

            # regiões tocando a borda não são fechadas → ignorar
            if not touches_border:
                regions.append((len(region_pixels), region_pixels))

    if not regions:
        raise ValueError("Nenhuma região fechada encontrada.")

    # pega a menor região
    _, pixels = min(regions, key=lambda r: r[0])

    # cria máscara final
    mask = np.zeros((h, w), dtype=bool)
    for y, x in pixels:
        mask[y, x] = True

    return mask
