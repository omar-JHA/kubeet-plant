# -*- coding: utf-8 -*-
import sys, traceback
import cv2
import numpy as np
import argparse
import string
import plantcv as pcv
import os.path
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-m", "--roi", help="Input region of interest file.", required=False)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.", action="store_true")
    args = parser.parse_args()
    return args
#linea 17
### Main pipeline
def main():
    # obtiene opciones de imagen
    args = options()
#LINEA 22
    if args.debug:
       print("Debug mode turned on...")
    # lee la imagen el flags=0 indica que se espera una imagen a escala de grises
    img = cv2.imread(args.image, flags=0)
   # cv2.imshow("imagen original",img)
    # Get directory path and image name from command line arguments
    path, img_name = os.path.split(args.image)
   
#LINEA 30
    # Read in image which is the pixelwise average of background images
    img_bkgrd = cv2.imread("background_average.jpg", flags=0)
    #cv2.imshow("ventana del fondo",img_bkgrd)
    # paso del procesamiento de imagenes
    device = 0
    ######hasta qui bien
#linea 37
    # Restar la imagen de fondo de la imagen con la planta.
    device, bkg_sub_img = pcv.image_subtract(img, img_bkgrd, device, args.debug)
    #cv2.imshow("imagen resta",bkg_sub_img)
    # Threshold the image of interest using the two-sided cv2.inRange function (keep what is between 50-190)
    bkg_sub_thres_img = cv2.inRange(bkg_sub_img, 50, 190)
    if args.debug:
        cv2.imwrite('bkgrd_sub_thres.png', bkg_sub_thres_img)  
#hasta qui todo bien
#linea 46
     # Filtrado de Laplace (identificar bordes basados ​​en la derivada 2)
    device, lp_img = pcv.laplace_filter(img, 1, 1, device, args.debug)
    #cv2.imshow("imagen de filtrado",lp_img)
    if args.debug:
        pcv.plot_hist(lp_img, 'histograma_lp')

    # Lapacian image sharpening, this step will enhance the darkness of the edges detected
    device, lp_shrp_img = pcv.image_subtract(img, lp_img, device, args.debug)
    #cv2.imshow("imagen de borde lapacian",lp_shrp_img)
    if args.debug:
        pcv.plot_hist(lp_shrp_img, 'histograma_lp_shrp')
#hasta aqui todo bien linea 58
    # Sobel filtering-filtrado de sobel 
    # 1ª derivada filtrado sobel a lo largo del eje horizontal, núcleo = 1, sin escala)
    """    segun esta masl son siete,kito scale y me kedo con apertura k,chekar sobel en docs
    device, sbx_img = pcv.sobel_filter(img, 1, 0, 1, 1, device, args.debug)
   """
    device, sbx_img = pcv.sobel_filter(img, 1, 0, 1, device, args.debug)
    #cv2.imshow("imagen sobel-eje horizontal",sbx_img)
    if args.debug:
        pcv.plot_hist(sbx_img, 'histograma_sbx')

    # Filtrado de la primera derivada sobel a lo largo del eje vertical, núcleo = 1, sin escala)
    device, sby_img = pcv.sobel_filter(img, 0, 1, 1, device, args.debug)
    #cv2.imshow("imagen sobel-ejevertical",sby_img)
    if args.debug:
        pcv.plot_hist(sby_img, 'histograma_sby')

    # Combina los efectos de ambos filtros x e y mediante la suma de matrizes
    # Esto captura los bordes identificados dentro de cada plano y enfatiza los bordes encontrados en ambas imágenes
    device, sb_img = pcv.image_add(sbx_img, sby_img, device, args.debug)
    #cv2.imshow("imagen suma de sobel",sb_img)
    if args.debug:
        pcv.plot_hist(sb_img, 'histograma_sb_comb_img')
#hasta aqui todo bien linea 82
     # usar filtro pasa bajo blur para suavizar la imagen de sobel
    device, mblur_img = pcv.median_blur(sb_img, 1, device, args.debug)
    #cv2.imshow("imagen blur",mblur_img)
    device, mblur_invert_img = pcv.invert(mblur_img, device, args.debug)
    #cv2.imshow("imagen blur-invertido",mblur_invert_img)
    # Combinar la imagen suavizada del sobel con la imagen afilada del laplaciano
    # combines the best features of both methods as described in "Digital Image Processing" by Gonzalez and Woods pg. 169
    #Combina las mejores características de ambos métodos como se describe en "Digital Image Processing" por González y Woods pág. 169
    device, edge_shrp_img = pcv.image_add(mblur_invert_img, lp_shrp_img, device, args.debug)
    #cv2.imshow("imagen-combinacion-sobel-laplacian",mblur_img)
    if args.debug:
        pcv.plot_hist(edge_shrp_img, 'hist_edge_shrp_img')

    # Realizar el umbral para generar una imagen binaria
    device, tr_es_img = pcv.binary_threshold(edge_shrp_img, 125, 255, 'dark', device, args.debug)
    #cv2.imshow("imagen binaria de combinacion",tr_es_img)
#hasta aqui todo bien linea 99
    # Prepare a few small kernels for morphological filtering
    #prepara nucleos pequeños para un filtrado moorfologico
    kern = np.zeros((3,3), dtype=np.uint8)
    kern1 = np.copy(kern)
    kern1[1,1:3]=1
    kern2 = np.copy(kern)
    kern2[1,0:2]=1
    kern3 = np.copy(kern)
    kern3[0:2,1]=1
    kern4 = np.copy(kern)
    kern4[1:3,1]=1

    # prepara un nucleo grande para la dilatacion
    kern[1,0:3]=1
    kern[0:3,1]=1
    # Perform erosion with 4 small kernels
    device, e1_img = pcv.erode(tr_es_img, 1, 1, device, args.debug)
    #cv2.imshow("erosion 1",e1_img)
    device, e2_img = pcv.erode(tr_es_img, 1, 1, device, args.debug)
    #cv2.imshow("erosion 2",e2_img)
    device, e3_img = pcv.erode(tr_es_img, 1, 1, device, args.debug)
    #cv2.imshow("erosion 3",e3_img)
    device, e4_img = pcv.erode(tr_es_img, 1, 1, device, args.debug)
    #cv2.imshow("erosion 4",e4_img)
    
    # Combine eroded images
    device, c12_img = pcv.logical_or(e1_img, e2_img, device, args.debug)
    #cv2.imshow("c12",c12_img)
    device, c123_img = pcv.logical_or(c12_img, e3_img, device, args.debug)
    #cv2.imshow("c123",c123_img)
    device, c1234_img = pcv.logical_or(c123_img, e4_img, device, args.debug)
    #cv2.imshow("c1234",c1234_img)

    # Bring the two object identification approaches together.
    # Using a logical OR combine object identified by background subtraction and the object identified by derivative filter.
    device, comb_img = pcv.logical_or(c1234_img, bkg_sub_thres_img, device, args.debug)
    #cv2.imshow("comb_img",comb_img)
    # Get masked image, Essentially identify pixels corresponding to plant and keep those.
    device, masked_erd = pcv.apply_mask(img, comb_img, 'black', device, args.debug)
    #cv2.imshow("masked_erd",masked_erd)
    #cv2.imshow("imagen original chkar",img)
    # Need to remove the edges of the image, we did that by generating a set of rectangles to mask the edges
    # img is (254 X 320)
    # mask for the bottom of the image
    device,im2, box1_img, rect_contour1, hierarchy1 = pcv.rectangle_mask(img, (120,184), (215,252), device, args.debug,color='white')
    #cv2.imshow("im2",box1_img)
    # mask for the left side of the image
    device,im3, box2_img, rect_contour2, hierarchy2 = pcv.rectangle_mask(img, (1,1), (85,252), device, args.debug,color='white')
    #cv2.imshow("im3",box2_img)
    # mask for the right side of the image
    device,im4, box3_img, rect_contour3, hierarchy3 = pcv.rectangle_mask(img, (240,1), (318,252), device, args.debug,color='white')
    #cv2.imshow("im4",box3_img)
    # mask the edges
    device,im5,box4_img, rect_contour4, hierarchy4 = pcv.rectangle_mask(img, (1,1), (318,252), device, args.debug)
    #cv2.imshow("im5",box4_img)

     # combine boxes to filter the edges and car out of the photo
    device, bx12_img = pcv.logical_or(box1_img, box2_img, device, args.debug)
    device, bx123_img = pcv.logical_or(bx12_img, box3_img, device, args.debug)
    device, bx1234_img = pcv.logical_or(bx123_img, box4_img, device, args.debug)
    #cv2.imshow("combinacion logica or",bx1234_img)

     # invert this mask and then apply it the masked image.
    device, inv_bx1234_img = pcv.invert(bx1234_img, device, args.debug)
   # cv2.imshow("combinacion logica or invertida",inv_bx1234_img)
    device, edge_masked_img = pcv.apply_mask(masked_erd, inv_bx1234_img, 'black', device, args.debug)
   # cv2.imshow("edge_masked_img",edge_masked_img)
    
     # assign the coordinates of an area of interest (rectangle around the area you expect the plant to be in)
    device,im6, roi_img, roi_contour, roi_hierarchy = pcv.rectangle_mask(img, (120,75), (200,184), device, args.debug)
    #cv2.imshow("im6",roi_img)
    # get the coordinates of the plant from the masked object
    plant_objects, plant_hierarchy = cv2.findContours(edge_masked_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
   
    # Obtain the coordinates of the plant object which are partially within the area of interest
    device, roi_objects, hierarchy5, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi_contour, roi_hierarchy, plant_objects, plant_hierarchy, device, args.debug)
    
    # Apply the box mask to the image to ensure no background
    device, masked_img = pcv.apply_mask(kept_mask, inv_bx1234_img, 'black', device, args.debug)
    #cv2.imshow("mascara final",masked_img)
#/////////////////////////////////////////////////////////////
    #device, masked_img = pcv.apply_mask(kept_mask, inv_bx1234_img, 'black', device, args.debug)
    rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #cv2.imshow("rgb",rgb)
    # Generate a binary to send to the analysis function
    device, mask = pcv.binary_threshold(masked_img, 1, 255, 'light', device, args.debug)
    #cv2.imshow("mask",mask)
    mask3d = np.copy(mask)
    plant_objects_2, plant_hierarchy_2 = cv2.findContours(mask3d,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    device, o, m = pcv.object_composition(rgb, roi_objects, hierarchy5, device, args.debug)
    
    # Get final masked image
    device, masked_img = pcv.apply_mask(kept_mask, inv_bx1234_img, 'black', device, args.debug)
    #cv2.imshow("maskara final2",masked_img)
################### copia lo de arriba esta mal el tutorial
    # Obtain a 3 dimensional representation of this grayscale image (for pseudocoloring)
    #rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
   
    # Generate a binary to send to the analysis function
    #device, mask = pcv.binary_threshold(masked_img, 1, 255, 'light', device, args.debug)

    # Make a copy of this mask for pseudocoloring
    #mask3d = np.copy(mask)

    # Extract coordinates of plant for pseudocoloring of plant
    #plant_objects_2, plant_hierarchy_2 = cv2.findContours(mask3d,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #device, o, m = pcv.object_composition(rgb, roi_objects, hierarchy5, device, args.debug)

    # Extract coordinates of plant for pseudocoloring of plant
    #plant_objects_2, plant_hierarchy_2 = cv2.findContours(mask3d,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #device, o, m = pcv.object_composition(rgb, roi_objects, hierarchy5, device, args.debug)
####################################
#######################
    ### Analysis ###
    # Perform signal analysis
    #################pruebas de que esta masl el tutorial""""""""""""""""
    #ols=type(args.image)
    #print ols
    ##############pruebas de que no agarro     device, hist_header, hist_data, h_norm = pcv.analyze_NIR_intensity(img, args.image, mask, 256, device, args.debug, args.outdir + '/' + img_name)

    #print(args.outdir+'/'+img_name)
    #print(args.debug)
    #al final si salio se agrego lo qyue esta debug= and filename=
    ##################################################### debug me marca True por ello puse pritn de mas 
    #device, hist_header, hist_data, h_norm = pcv.analyze_NIR_intensity(img, rgb, mask, 256, device, debug='print', filename=False)
    device, hist_header, hist_data, h_norm = pcv.analyze_NIR_intensity(img,rgb, mask, 256, device,debug=args.debug,filename=args.outdir+'/'+ img_name)
    
    # Perform shape analysis
    device, shape_header, shape_data, ori_img = pcv.analyze_object(rgb, args.image, o, m, device,debug=args.debug, filename=args.outdir + '/' + img_name)

    # Print the results to STDOUT
    pcv.print_results(args.image, hist_header, hist_data)
    pcv.print_results(args.image, shape_header, shape_data)

    cv2.waitKey()
    cv2.destroyAllWdindows()
    
# Call program
if __name__ == '__main__':
    main()
