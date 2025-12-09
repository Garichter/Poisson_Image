import numpy as np
import cv2

def selecionar_mascara_manual(source_rgb):
    img_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    window_name = "1. SELECAO: Clique nos pontos p/ contornar. ENTER p/ finalizar."
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display = img_bgr.copy()
        
        if len(points) > 0:
            for pt in points:
                cv2.circle(display, pt, 3, (0, 0, 255), -1)
            
            if len(points) > 1:
                cv2.polylines(display, [np.array(points)], False, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            if len(points) > 2:
                break
            else:
                print("Selecione pelo menos 3 pontos para formar uma área.")

    cv2.destroyAllWindows()

    h, w = source_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    cv2.fillPoly(mask, [np.array(points)], 255)

    return mask > 127

def escolher_offset(source, target, mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        raise ValueError("A máscara está vazia. Selecione uma área na imagem de origem.")
        
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    cut = source[y_min:y_max+1, x_min:x_max+1]
    mask_cut = mask[y_min:y_max+1, x_min:x_max+1]

    cut_bgr = cv2.cvtColor(cut, cv2.COLOR_RGB2BGR)
    target_bgr = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)

    max_height = 800
    max_width = 1200
    
    h_orig, w_orig = target.shape[:2]
    
    scale = 1.0
    if h_orig > max_height or w_orig > max_width:
        scale = min(max_height / h_orig, max_width / w_orig)
    
    target_disp = cv2.resize(target_bgr, None, fx=scale, fy=scale)
    cut_disp = cv2.resize(cut_bgr, None, fx=scale, fy=scale)

    mask_cut_disp = cv2.resize(mask_cut.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    
    h_cut_disp, w_cut_disp = cut_disp.shape[:2]
    
    click = [target_disp.shape[1] // 2, target_disp.shape[0] // 2]
    
    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or (flags & cv2.EVENT_FLAG_LBUTTON):
            click[0] = x
            click[1] = y

    window_name = "2. POSICIONAMENTO (Visualizacao Reduzida)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, callback)

    while True:
        temp = target_disp.copy()
        
        x_center = click[0]
        y_center = click[1]
        
        x0 = int(x_center - w_cut_disp / 2)
        y0 = int(y_center - h_cut_disp / 2)
        x1 = x0 + w_cut_disp
        y1 = y0 + h_cut_disp

        tgt_y0 = max(0, y0)
        tgt_x0 = max(0, x0)
        tgt_y1 = min(target_disp.shape[0], y1)
        tgt_x1 = min(target_disp.shape[1], x1)

        src_y0 = tgt_y0 - y0
        src_x0 = tgt_x0 - x0
        src_y1 = src_y0 + (tgt_y1 - tgt_y0)
        src_x1 = src_x0 + (tgt_x1 - tgt_x0)

        if tgt_y1 > tgt_y0 and tgt_x1 > tgt_x0:
            region = temp[tgt_y0:tgt_y1, tgt_x0:tgt_x1]
            overlay = cut_disp[src_y0:src_y1, src_x0:src_x1]
            mask_ov = mask_cut_disp[src_y0:src_y1, src_x0:src_x1]
            
            np.copyto(region, overlay, where=(mask_ov > 0)[..., None])

        cv2.imshow(window_name, temp)
        k = cv2.waitKey(1)
        
        if k == 13: 
            final_y_disp = click[1] - h_cut_disp / 2
            final_x_disp = click[0] - w_cut_disp / 2
            final_y_real = int(final_y_disp / scale)
            final_x_real = int(final_x_disp / scale)
            break

    cv2.destroyAllWindows()
    
    return (final_y_real - y_min, final_x_real - x_min)