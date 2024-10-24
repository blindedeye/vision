import cv2
import numpy as np 

def main() -> None:
    img = cv2.imread('coins.jpg')
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_grey, (0,0),1.5)
    img_edge = cv2.Canny(img_blur,50,150)

    circles = cv2.HoughCircles(img_blur,cv2.HOUGH_GRADIENT,1,20,param1=219,param2=54,minRadius=1,maxRadius=0)
    print(circles)

    if circles is not None:
        for circle in circles:
            x, y, radius = circle[0]
            x = int(x)
            y = int(y)
            cv2.circle(img, (x,y), int(radius), (0,0,255), 2)

    #cv2.imshow('edge',img_edge)
    cv2.imshow('img',img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()