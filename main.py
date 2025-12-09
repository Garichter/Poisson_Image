import numpy as np
from imageio import imread, imwrite
import placeMask
import localllumination as li
import seamlessCloning as sc
import textureFlattening as tf
import changeColor as cg


if __name__ == "__main__":
    print("--- POISSON IMAGE EDITING ---")
    print("1. Seamless Cloning")
    print("2. Seamless cloning mixed")
    print("3. Texture Flattening")
    print("4. Local illumination changes")
    print("5. Local color changes")
    choice = input("Escolha: ")

    if choice == '1':
        target = imread("destino.jpg").astype(np.uint8)
        source = imread("fonte.jpg").astype(np.uint8)

        mask_src = placeMask.selecionar_mascara_manual(source)
        offset = placeMask.escolher_offset(source, target, mask_src)
        
        out = sc.seamless_clone(target, source, mask_src, offset, mixed=False)
        imwrite("saida_clone.png", out)

    elif choice == '2':
        target = imread("destino.jpg").astype(np.uint8)
        source = imread("fonte.jpg").astype(np.uint8)

        mask_src = placeMask.selecionar_mascara_manual(source)
        offset = placeMask.escolher_offset(source, target, mask_src)
        
        out = sc.seamless_clone(target, source, mask_src, offset, mixed=True)
        imwrite("saida_clone.png", out)

    elif choice == '3':
        target = imread("fonte.jpg").astype(np.uint8)
        
        mask_src = placeMask.selecionar_mascara_manual(target)
        out = tf.texture_flattening(target, mask_src)
        imwrite("saida_flatten.png", out)

    elif choice == '4':
        target = imread("limao.jpg").astype(np.uint8)
        mask_src = placeMask.selecionar_mascara_manual(target)
        out = li.local_illumination(target, mask_src)
        imwrite("saida_iluminacao.png", out)
    
    elif choice == '5':
        target = imread("flor.jpg").astype(np.uint8)
        mask_src = placeMask.selecionar_mascara_manual(target)
        out = cg.local_color_change_recolor(target, mask_src)
        imwrite("saida_cor.png", out)


    print("Done.")