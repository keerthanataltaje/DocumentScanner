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


def shape_correct(h):
    h=h.reshape((4,2))
    hnew=np.zeros((4,2),dtype=np.float32)
    add=h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]
    
    diff=np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]
    return hnew




def contour_detection():
    image_path='img_1.jpeg'
    image=cv2.imread(image_path)
    ratio_old_to_new=image.shape[0]/400
    original=image.copy()
    
    #Resize the image for convinience
    image=resize(image,height=400)
    print(image.shape)
    resized_width,resized_height,_=image.shape
    path='F:\KeerthanaProjects\intermediate_images'
    new_path=os.path.join(path,'resizedimg_2.jpeg')
    cv2.imwrite(new_path, image)
    user_interface.display(new_path)
    #cv2.imshow("Image",image)
   
    #Preparation for Canny edge detection and applying it
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image=cv2.GaussianBlur(gray_image,(9,9),0)
    #cv2.imshow("Gray-Image",gray_image)
    
    
    #Remove any holes between edges
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    morph_close=cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("Morphed-Image",morph_close)
    edge=cv2.Canny(morph_close,0,84)
    cv2.imshow("Edged-Image",edge)
    
    #Find Contours
    
    
    (contours,heir)=cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours=sorted(contours,key=cv2.contourArea,reverse=True)[:10]
    for c in contours:
        perimeter=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.02*perimeter,True)[:4]
        target=approx
        if len(approx)>=4:
            target=approx
            break
    #approx=shape_correct(approx)
    print("Lengthof approx=",len(approx),approx)
    
    #points_required=([[0,0],[resized_height,0],[resized_height,resized_width],[0,resized_width]])
    #transform_image = cv2.getPerspectiveTransform(approx,np.float32(points_required))
    #final_image = cv2.warpPerspective(image,transform_image,(500,400))
   
    cv2.drawContours(image,[approx],-1,(0,255,0),2)
    cv2.imshow("Contour_image",image)
    
    t_right= (resized_width, 0)
    b_right = (resized_width, resized_height)
    b_left = (0, resized_height)
    t_left = (0, 0)
    points_required = np.array([[t_left], [t_right], [b_right], [b_left]])
    
  
    
    #Get a top down view of the image
    transform_image = cv2.getPerspectiveTransform(np.float32(target),np.float32(points_required))
    final_image = cv2.warpPerspective(image,transform_image,(resized_width,resized_height))
    cv2.imshow("transformed image=",final_image)
    #cv2.drawContours(final_image, [target], -1, (0, 255, 0), 2)
    #cv2.imshow("Contour_image",image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
contour_detection()