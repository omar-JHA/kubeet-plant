#librerias en uso
import sys, traceback
import cv2
import numpy as np
import argparse
import string
import plantcv as pcv

### Parse command-line arguments
#linea para ingresar datos por parte del usuario
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-r","--result", help="result file.", required= False )
    parser.add_argument("-w","--writeimg", help="write out images.", default=False)
    parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.", default=None)
    args = parser.parse_args()
    return args
### Main pipeline
def main():
    # Get options 1
    args = options()

    # lee imagen 2
    img, path, filename = pcv.readimage(args.image)
   # cv2.imshow("imagen",img)
    # pasos del pipeline 3
    device = 0
    debug=args.debug 

    # Convert RGB to HSV and extract the Saturation channel 4
    #convertir RGB a HSV y extraer el canal de saturacion
    device, s = pcv.rgb2gray_hsv(img, 's', device, debug)
   # cv2.imshow("rgb a hsv y extraer saturacion 4",s)
     # Threshold the Saturation image 5
     #sacar imagen binaria del canal de saturacion
    device, s_thresh = pcv.binary_threshold(s, 85, 255, 'light', device, debug)
   # cv2.imshow("imagen binaria de hsv",s_thresh)
    # Median Filter 6
    #sacar un filtro median_blur
    device, s_mblur = pcv.median_blur(s_thresh, 5, device, debug)
    device, s_cnt = pcv.median_blur(s_thresh, 5, device, debug)
   # cv2.imshow("s_mblur",s_mblur)
   # cv2.imshow("s_cnt",s_cnt)
    # Convert RGB to LAB and extract the Blue channel 7
    #convertir RGB(imagen original) a LAB Y extraer el canal azul
    device, b = pcv.rgb2gray_lab(img, 'b', device, debug)
   # cv2.imshow("convertir RGB a LAB",b)
    # Threshold the blue image 8
    #sacar imagen binaria de LAB  imagen blue
    device, b_thresh = pcv.binary_threshold(b, 160, 255, 'light', device, debug)
    device, b_cnt = pcv.binary_threshold(b, 160, 255, 'light', device, debug)
   # cv2.imshow("imagen binaria de LAB",b_thresh)
   # cv2.imshow("imagen binaria",b_cnt)
    # Fill small objects
    #device, b_fill = pcv.fill(b_thresh, b_cnt, 10, device, debug)
    
     # Join the thresholded saturation and blue-yellow images 9
    #
    device, bs = pcv.logical_or(s_mblur, b_cnt, device, debug)
   # cv2.imshow("suma logica s_mblur and b_cnt",bs)
     # Apply Mask (for vis images, mask_color=white) 10
    device, masked = pcv.apply_mask(img, bs, 'white', device, debug)
   # cv2.imshow("aplicar mascara masked",masked)
    # Convert RGB to LAB and extract the Green-Magenta and Blue-Yellow channels 11
    device, masked_a = pcv.rgb2gray_lab(masked, 'a', device, debug)
    device, masked_b = pcv.rgb2gray_lab(masked, 'b', device, debug)
   # cv2.imshow("canal verde-magenta",masked_a)
   # cv2.imshow("canal azul-amarillo",masked_b)  
    # Threshold the green-magenta and blue images 12
    device, maskeda_thresh = pcv.binary_threshold(masked_a, 115, 255, 'dark', device, debug)
    device, maskeda_thresh1 = pcv.binary_threshold(masked_a, 135, 255, 'light', device, debug)
    device, maskedb_thresh = pcv.binary_threshold(masked_b, 128, 255, 'light', device, debug)
   # cv2.imshow("threshold de canal verde-magenta dark",maskeda_thresh)
   # cv2.imshow("threshold de canal verde-magenta light",maskeda_thresh1)
   # cv2.imshow("threshold de canal azul-amarillo",maskedb_thresh)
    # Join the thresholded saturation and blue-yellow images (OR) 13
    device, ab1 = pcv.logical_or(maskeda_thresh, maskedb_thresh, device, debug)
    device, ab = pcv.logical_or(maskeda_thresh1, ab1, device, debug)
    device, ab_cnt = pcv.logical_or(maskeda_thresh1, ab1, device, debug)
   # cv2.imshow("suma logica or 1",ab1)
   # cv2.imshow("suma logica or 2 ab",ab)
   # cv2.imshow("suma logica or 3 ab_cnt",ab_cnt)
   
    # Fill small objects 14
    device, ab_fill = pcv.fill(ab, ab_cnt, 200, device, debug)
   # cv2.imshow("ab_fill",ab_fill)
    # Apply mask (for vis images, mask_color=white) 15
    device, masked2 = pcv.apply_mask(masked, ab_fill, 'white', device, debug)
   # cv2.imshow("aplicar maskara2 white",masked2)
   
    ####################entendible hasta aqui######################
    # Identify objects 16 solo print Se utiliza para identificar objetos (material vegetal) en una imagen.
    #imprime la imagen si uso print o no si uso plot no almacena la imagen pero en pritn si la aguarda
    #usa b_thresh y observa
    device,id_objects,obj_hierarchy = pcv.find_objects(masked2,ab_fill, device, debug)
  
    # Define ROI 17 solo print encierra el objeto detectato pero aun es manual aun no automatico
    device, roi1, roi_hierarchy= pcv.define_roi(masked2, 'rectangle', device, None, 'default', debug, True, 92, 80, -127, -343)
    
    # Decide which objects to keep 18
    device,roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi1, roi_hierarchy, id_objects, obj_hierarchy, device, debug)
    
    # Object combine kept objects 19
    device, obj, mask = pcv.object_composition(img, roi_objects, hierarchy3, device, debug)


    ############### Analysis ################

    outfile=False
    if args.writeimg==True:
        outfile=args.outdir+"/"+filename

    # Find shape properties, output shape image (optional)
    device, shape_header, shape_data, shape_img = pcv.analyze_object(img,'image', obj, mask, device,args.outdir + '/' + filename)

    # Shape properties relative to user boundary line (optional)
    device, boundary_header, boundary_data, boundary_img1 = pcv.analyze_bound(img, args.image, obj, mask, 1680, device, debug, args.outdir + '/' + filename)

    # Determine color properties: Histograms, Color Slices and Pseudocolored Images, output color analyzed images (optional)
    device, color_header, color_data, color_img = pcv.analyze_color(img, args.image, kept_mask, 256, device, debug, 'all', 'v', 'img', 300, args.outdir + '/' + filename)

     #Write shape and color data to results file
    result=open(args.result,"a")
    result.write('\t'.join(map(str,shape_header)))
    result.write("\n")
    result.write('\t'.join(map(str,shape_data)))
    result.write("\n")
    for row in shape_img:  
        result.write('\t'.join(map(str,row)))
        result.write("\n")
    result.write('\t'.join(map(str,color_header)))
    result.write("\n")
    result.write('\t'.join(map(str,color_data)))
    result.write("\n")
    for row in color_img:
        result.write('\t'.join(map(str,row)))
        result.write("\n")
    result.close()
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


