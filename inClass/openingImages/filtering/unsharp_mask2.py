import cv2 
import numpy as np 

def main() -> None:
    img = cv2.imread('tyu.jpg',cv2.IMREAD_GRAYSCALE)

    blurred_img = cv2.GaussianBlur(img,(5,5),1)
    # print(blurred_img[10:20,10:20])

    mask = img - blurred_img
    # print(mask[10:20,10:20])

    scale_val = 1.5 
    unsharp_img = img + (mask * scale_val)
    
    cv2.imshow('unsharp_image',unsharp_img)
    cv2.waitKey(0)