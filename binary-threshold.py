
import plantcv as pcv

#lee imagen

img,path,img_filename=pcv.readimage("/home/fitosmartplatform/plantCV/prueba/plan.jpg")

#contador dl paso de procesamiento de imagen
device=0

# Create binary image from a gray image based on threshold values. Targeting light objects in the image.
device, threshold_light = pcv.binary_threshold(img, 36, 255, 'dark', device, debug="print")
device, h_channel=pcv.rgb2gray_hsv(img, 'h', device, debug="print")
pcv.print_image(threshold_light, "/home/fitosmartplatform/plantCV/prueba/image-gray.jpg")
pcv.print_image(threshold_light, "/home/fitosmartplatform/plantCV/prueba/test-image.jpg")
"""notas"""
#si uso plot :imprime datos de la imagen no aguarda
#si uso print :imprime imagen y la aguarda con un nombre de la funcion


#si uso el light es un entorno diferente
#si uso  dark es otro entorno de la imagen


