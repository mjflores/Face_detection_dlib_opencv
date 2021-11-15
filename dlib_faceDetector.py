#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
                 Face detection
                 
  Example of face detection by using dlib library

@author:  mjflores
@ ref: 
 
   http://dlib.net/cnn_face_detector.py.html

   You can download the pre-trained model from:
   http://dlib.net/files/mmod_human_face_detector.dat.bz2

'''

from pathlib import Path
import cv2
import dlib
import time

dirMMOD  =r"/home/mjflores/MEGA/Mis Programas/Datos/SP/mmod_human_face_detector.dat"
dirFaces =r"/home/mjflores/MEGA/Mis Programas/Datos/Rostros/" 

def ls(ruta = Path.cwd()):
    return [arch.name for arch in Path(ruta).iterdir() if arch.is_file()]

cnn_face_detector = dlib.cnn_face_detection_model_v1(dirMMOD)
hog_face_detector = dlib.get_frontal_face_detector()

color = (0, 0, 255, 0)

def detFaces_hog_svm():    
    ltImagenes = ls(dirFaces)
    for ig in ltImagenes:
        image = cv2.resize(cv2.imread(dirFaces+ig),(640, 480))
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tic = time.time()
        dets = hog_face_detector(gray, 1)
        toc = time.time()
        print("Time: ",toc-tic," seconds")

        n_faces = len(dets)
        print("Number of faces detected: {}".format(n_faces))
        if n_faces >0:
           for d in enumerate(dets):
               cv2.rectangle(image, (d[1].left(), d[1].top()), (d[1].right(), d[1].bottom()), 255, 1)
 
        cv2.putText(image,"Numero  rostros = " + str(n_faces), (10, 14),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
              
         # Display the resulting image
        cv2.imshow('face with dlib: hog+svm', image)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    cv2.destroyAllWindows()
    print("End...\n")
    
    
def detFaces_cnn():
    ltImagenes = ls(dirFaces)
    for ig in ltImagenes:    
        image = cv2.resize(cv2.imread(dirFaces+ig), (640, 480))       
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tic = time.time()
        dets = cnn_face_detector(gray, 1)
        toc = time.time()
        print("Time: ",toc-tic," seconds")

        n_faces = len(dets)
        print("Number of faces detected: {}".format(n_faces))
        if n_faces >0:
           for d in enumerate(dets):
               cv2.rectangle(image, (d[1].rect.left(), d[1].rect.top()), (d[1].rect.right(), d[1].rect.bottom()), 255, 1)

  
        cv2.putText(image,"Numero  rostros = " + str(n_faces), (10, 14),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
         # Display the resulting image
        cv2.imshow('face with dlib: cnn', image)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    cv2.destroyAllWindows()
    print("End...\n")


#===================================================================
# Face detection
#detFaces_cnn()
detFaces_hog_svm()

