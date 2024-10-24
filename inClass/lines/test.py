import cv2
import numpy as np

def on_trackbar_change(val):
    pass

def main() -> None:
    img = cv2.imread('coins.jpg')
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_grey, (0, 0), 1.5)

    cv2.namedWindow('image')
    cv2.createTrackbar('param1', 'image', 50, 300, on_trackbar_change)
    cv2.createTrackbar('param2', 'image', 30, 100, on_trackbar_change)

    while True:
        param1 = cv2.getTrackbarPos('param1', 'image')
        param2 = cv2.getTrackbarPos('param2', 'image')

        img_edge = cv2.Canny(img_blur, param1, 150)

        circles = cv2.HoughCircles(
            img_blur, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=20, 
            param1=param1, 
            param2=param2, 
            minRadius=1, 
            maxRadius=0
        )

        img_copy = img.copy()

        if circles is not None:
            circles = np.round(circles[0, :]).astype('int')
            for (x, y, r) in circles:
                cv2.circle(img_copy, (x, y), r, (0, 255, 0), 2)

        cv2.imshow('edge', img_edge)
        cv2.imshow('image', img_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
