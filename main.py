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
    print("2. Texture Flattening")
    print("3. Local illumination changes")
    choice = input("Escolha: ")

    if choice == '1':
        target = imread("destino.jpg").astype(np.uint8)
        source = imread("fonte.jpg").astype(np.uint8)

        mask_src = placeMask.selecionar_mascara_manual(source)
        offset = placeMask.escolher_offset(source, target, mask_src)
        
        out = sc.seamless_clone_poisson(target, source, mask_src, offset, mixed=True)
        imwrite("saida_clone.png", out)

    elif choice == '2':
        target = imread("fonte.jpg").astype(np.uint8)
        
        mask_src = placeMask.selecionar_mascara_manual(target)
        out = tf.texture_flattening_poisson(target, mask_src)
        imwrite("saida_flatten.png", out)

    elif choice == '3':
        target = imread("fruta.jpg").astype(np.uint8)
        mask_src = placeMask.selecionar_mascara_manual(target)
        out = li.local_illumination_poisson(target, mask_src)
        imwrite("saida_iluminacao.png", out)
    
    elif choice == '4':
        target = imread("flor.jpg").astype(np.uint8)
        mask_src = placeMask.selecionar_mascara_manual(target)
        out = cg.local_color_change_recolor(target, mask_src)
        imwrite("saida_cor.png", out)


    print("Done.")