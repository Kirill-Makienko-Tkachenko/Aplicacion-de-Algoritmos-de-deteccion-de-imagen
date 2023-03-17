#Comparacion algoritmos de deteccion de imagenes
#Eric Samuel Miranda Alvarez
#Oscar Omar Cepeda Vazquez
#Kirill Makienko Tkachenko
#14/03/23

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import os

def sobel_edge_detection(image, verbose=False):

    #Filtro Sobel
    

    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    new_image_x = cv.Sobel(image,cv.CV_64F,1,0,ksize=31)
 
    new_image_y = cv.Sobel(image,cv.CV_64F,0,1,ksize=31)
 

    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
 
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
 
    return gradient_magnitude



def filtro_prewitt(img): #https://gist.github.com/rahit/c078cabc0a48f2570028bff397a9e154

    kgb=gaussian_kernel(30,1.4)
    imgb=cv.filter2D(img,-1,kgb)
    #kernelx = np.array([[-3,0,3],[-10,0,-10],[-3,0,3]])
    #kernely = np.array([[3,10,3],[0,0,0],[-3,10,-3]])
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])    
    img_prewittx = cv.filter2D(imgb, -1, kernelx)
    img_prewitty = cv.filter2D(imgb, -1, kernely)
    final = np.sqrt(np.square(img_prewittx) + np.square(img_prewitty))
    final *= 255.0 / final.max()
    return final











#--------------------------Filtro Roberts----------------------

def Roberts(matriz): #https://www.geeksforgeeks.org/python-opencv-roberts-edge-detection/
    kgb=gaussian_kernel(80,1.4)
    imgb=cv.filter2D(matriz,-1,kgb)
    roberts_cross_v = np.array( [[1, 0 ],
                                [0,-1 ]], np.float32 )
    
    roberts_cross_h = np.array( [[ 0, 1 ],
                                [ -1, 0 ]], np.float32)
    vertical = ndimage.convolve(matriz, roberts_cross_v )
    horizontal = ndimage.convolve(matriz, roberts_cross_h )
    roberts = np.sqrt( np.square(horizontal) + np.square(vertical))
    return roberts

def Roberts_filtro(matriz): #https://www.geeksforgeeks.org/python-opencv-roberts-edge-detection/
    kgb=gaussian_kernel(80,1.4)
    imgb=cv.filter2D(matriz,-1,kgb)
    roberts_cross_v = np.array( [[1, 0 ],
                                [0,-1 ]], np.float32 )
    
    roberts_cross_h = np.array( [[ 0, 1 ],
                                [ -1, 0 ]], np.float32)
    vertical = ndimage.convolve(imgb, roberts_cross_v )
    horizontal = ndimage.convolve(imgb, roberts_cross_h )
    roberts = np.sqrt( np.square(horizontal) + np.square(vertical))
    return roberts

#------------------------Cortesia del documento proporcionado por el profesor-----------------------------------------
# filtro gaussiano 
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

#magnitud y dirección del gradiente (filtro sobel)
def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)

# non-maxima supression
# también conocido como adelgazamiento de bordes, porque conserva 
# los gradientes más afilados y descarta los demás. El algoritmo 
# se implementa en píxeles, dada la magnitud y dirección del gradiente
def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z

# doble umbral
# suprimir o preservar el gradiente del píxel que se procesa en relación con sus píxeles
# vecinos que apuntan en la misma dirección.
# Este paso considera la fuerza de la magnitud en toda la imagen.
def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

# seguimiento de bordes por histéresis
 # se usa para decidir si se considera un borde débil en el resultado final o no
def hysteresis(img, weak, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

#------------------------Cortesia del documento proporcionado por el profesor----------------------------------------


#--------------Necesario para hacer trabajar el filtro de Canny-------------------


arch = input('Inserta el nombre del archivo, por defecto es "Grogu.jpg"\n')

#Reemplazar con path hasta la imagen
os.chdir(r'C:\Users\elchi\Desktop\d')

if EOFError and arch == "":
    arch = "Grogu.jpg"
    img = arch
else:
    arch = arch
    img = arch

print(type(arch))

matriz = cv.imread(img)
input()
matriz = cv.cvtColor(matriz, cv.COLOR_BGR2GRAY)


#a4 = matriz.astype(np.float32)
#print(a4)
#dim = matriz.shape
#filas = dim[0]
#columnas = dim[1]
#print(columnas)
#print(filas)




#---------------------Convertir imagen a filtro de Canny--------------------
img = cv.imread(img, cv.IMREAD_GRAYSCALE)
img=img.astype(np.float32)

#desenfoque gaussiano (suavizado, eliminación de ruido)
#kgb=gaussian_kernel(5,1.4)
#imgb=cv.filter2D(img,-1,kgb)

# intensidad del gradiente (g), dirección del borde (t)
#g, t=sobel_filters(imgb)

# supresión no máxima
#Z=non_max_suppression(g, t)

# aplicación del doble umbral
#res,w,s=threshold(Z)

# hysteresis
#res_img=hysteresis(res,w,s)
#---------------------Convertir imagen a filtro de Canny--------------------

#Imprimir imagen con filtro de roberts
r_imagen = Roberts(matriz)
r2_imagen = Roberts_filtro(matriz)

#Filtro Sobel
s_imagen = sobel_edge_detection(img, verbose=True)

#filtro prewitt
#Pasamos imagen con filtro gaussiano aplicado
p_imagen = filtro_prewitt(img)





plt.figure()
#plt.subplot(1,5,1)
#plt.imshow(img,cmap="gray", interpolation="nearest")
#plt.title('Original'),plt.axis('off')


#plt.subplot(1,5,2)
#plt.imshow(res_img,cmap="gray", interpolation="nearest")
#plt.title('Filtro Canny'),plt.axis('off')


#plt.subplot(1,5,3)
plt.imshow(r_imagen,cmap="gray", interpolation="nearest")
plt.title('Filtro Roberts'),plt.axis('off')
plt.show()
plt.imshow(r2_imagen,cmap="gray", interpolation="nearest")
plt.title('Filtro Roberts'),plt.axis('off')
plt.show()
#plt.subplot(1,5,4)
#plt.imshow(s_imagen,cmap="gray", interpolation="nearest")
#plt.title('Filtro sobel'),plt.axis('off')


#plt.subplot(1,5,5)
#plt.imshow(p_imagen,cmap="gray", interpolation="nearest")
#plt.title('Filtro prewitt'),plt.axis('off')


#plt.show()

input()
#cv.imwrite("Original.jpg",img,)
#cv.imwrite("Canny.jpg",res_img)
#cv.imwrite("Roberts.png",r_imagen)
#cv.imwrite("Sobel.jpg",s_imagen)
#cv.imwrite("Prewitt.jpg",p_imagen)