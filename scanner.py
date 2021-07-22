# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 18:18:36 2021

@author: Keerthana
"""
import numpy as np
import cv2
import skimage.filters
import user_interface
import os

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]

	if width is None and height is None:
		return image

	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)

	else:	
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized_image = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized_image

def contour_detection():
    image_path='img_1.jpeg'
    image=cv2.imread(image_path)
    ratio_old_to_new=image.shape[0]/400
    original=image.copy()
    
    #Resize the image for convinience
    image=resize(image,height=400)
    print(image.shape)
    path='F:\KeerthanaProjects\intermediate_images'
    new_path=os.path.join(path,'resizedimg_2.jpeg')
    cv2.imwrite(new_path, image)
    user_interface.display(new_path)
    #cv2.imshow("Image",image)
    '''
    #Preparation for Canny edge detection and applying it
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image=cv2.GaussianBlur(gray_image,(5,5),0)
    #cv2.imshow("Gray-Image",gray_image)
    
    
    #Remove any holes between edges
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    #cv2.imshow("Morphed-Image",morph_close)
    edge=cv2.Canny(morph_close,0,84)
    cv2.imshow("Edged-Image",edge)
    
    #Find Contours
    target=None
    
    (contours,heir)=cv2.findContours(edge,cv2.cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours=sorted(contours,key=cv2.contourArea,reverse=True)
    for c in contours:
        perimeter=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.02*perimeter,True)
        if len(approx):
            target=approx
            break
    print("Lengthof approx=",len(approx))
    print("Target=>",target,"Length of target=>",len(target))
    
    cv2.drawContours(image,[target],-1,(0,255,0),2)
    cv2.imshow("Contour_image",image)
    
    pts = np.float32([[0,0],[400,0],[400,225],[0,225]])
    
    #Get a top down view of the image
    transform_image = cv2.getPerspectiveTransform(np.float32(target),pts)
    final_image = cv2.warpPerspective(image,transform_image,(400,225))
    cv2.drawContours(final_image, [target], -1, (0, 255, 0), 2)
    cv2.imshow("Contour_image",image)
   
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
contour_detection()