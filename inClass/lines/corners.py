import cv2
import numpy as np

def main() -> None:
    img = cv2.imread('PXL_20211006_195833767_small.jpg')

    cv2.imshow('img',img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()