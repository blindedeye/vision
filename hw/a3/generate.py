import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_size = 200

for i in range(4):
    marker = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker = cv2.aruco.generateImageMarker(aruco_dict, i, marker_size)
    cv2.imwrite(f"marker_{i}.png", marker)
