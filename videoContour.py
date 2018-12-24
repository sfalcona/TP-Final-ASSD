import numpy as np
import cv2
from PIL import ImageGrab


class video2Contour:
    '''Procesador de video basado en opencv2, obtiene el contorno en tiempo real correspondiente
    a los distintos movimientos que aparecen en un video. '''

    def __init__(self, source):
        '''Como parametros recibe:
        Source: 0 si se desea usar el dispositivo predeterminado de video o
                path si se quiere analizar un video'''
        self.source = source

        # Este bloque son variables del procesamiento
        self.blurKernel = (5,5)
        self.blurAmmount = 1
        self.threshLimit = 90
        self.threshVal = 255
        self.dilateIters = 6

        # Este bloque son variables para el analisis de cada contorno
        self.maxArea = 300
        self.prevC = 0
        self.danger = False

    def setBlurParams(self, kernel:'Tamano de la muestra' = (5,5), ammount:'Cantidad de desenfoque' = 1):
        '''Seteo parametros correspondientes al desenfoque'''
        self.blurKernel = kernel
        self.blurAmmount = ammount

        
    def setThresholdParams(self, limit:'Valor a partir del cual tomo como valido' = 90, val:'Valor al que modifico' = 255):
        '''Seteo parametros correspondientes al threshold'''
        self.threshLimit = limit
        self.threshVal = val

    def setDilatedIters(self, iters:'Cantidad de iteraciones' = 6):
        self.dilateIters = iters

    def start(self):
        '''Comienza el procesamiento'''
        self.cap = cv2.VideoCapture(self.source)
        self.bgs = cv2.createBackgroundSubtractorMOG2(10,80,True)
        while(True):
    
            ret, frame = self.cap.read()    
            frame2 = self.bgs.apply(frame)
            blur = cv2.GaussianBlur(frame2,(5,5),1) 
            ret, thresh_img = cv2.threshold(blur,91,255,cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
            dilated = cv2.dilate(thresh_img,kernel,iterations = 6) 
            gradient = cv2.morphologyEx(thresh_img, cv2.MORPH_GRADIENT, kernel)
            opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
            contours =  cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]

            font = cv2.FONT_HERSHEY_COMPLEX
            color = (0,0,255) if ((self.prevC > 2) and self.danger ) else (0,255,0)
            self.danger = False
            cv2.putText(frame,'.',(10,10), font, 4,color,5,cv2.LINE_AA)

            for c in contours:
                if cv2.contourArea(c) > self.maxArea:
                    if self.prevC != 0:
                        self.danger = True
                        cv2.drawContours(frame, [c], -1, (0,255,0), 3)
                self.prevC = len(c)

            cv2.imshow('frame',frame) 

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


