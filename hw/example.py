"""
Example in class for HW2
"""

import cv2
import numpy as np 


def sign_lines(img_edges) -> np.ndarray:
    # HoughLinesP for two points to draw a line segment
    lines = cv2.HoughLinesP(img_edges, 
                            rho=1,
                            theta=np.pi/180,
                            threshold=35,
                            minLineLength=5,
                            maxLineGap=5)
    if lines is not None:
        return lines
    return None

def sign_line_axis(lines, sign=None):
    xaxis = np.empty(0, dtype=np.int32)
    yaxis = np.empty(0, dtype=np.int32)
    for line in lines: # even indices are x odd are y
        for x0, y0, x1, y1 in lines:
            if sign == "Green":
                # Do stuff for green sign
                pass
            elif sign == "Stop":
                # Do stuff for stop sign
                pass

            xaxis = np.append(xaxis, x0)
            xaxis = np.append(xaxis, x1)
            yaxis = np.append(yaxis, y0)
            yaxis = np.append(yaxis, y1)

def main() -> None:
    green_sign = cv2.imread("green_box.jpg")
    blank_img = np.zeros(green_sign.shape)
    green_sign_hsv = cv2.cvtColor(green_sign, cv2.COLOR_BGR2HSV)
    # print(green_sign_hsv[50:60,50:60]) # range of pixels
    color_low = np.array([52,250,165]) # margin of error <- green in sign : multiple colors to search through
    color_high = np.array([56,255,175])
    mask = cv2.inRange(green_sign_hsv, color_low, color_high)
    edges = cv2.Canny(mask, 100, 200) # edge detection on the mask

    lines = sign_lines(edges)
    # if lines is not None:
    #     for line in lines:
    #         x0,y0,x1,y1 = line[0]
    #         cv2.line(blank_img, (x0,y0), (x1,y1), (0,0,255), 2, cv2.LINE_AA)
    if lines is None:
        return None 
    
    xaxis, yaxis = sign_line_axis(lines, sign="Green")
    
    xmin = xaxis.min()
    xmax = xaxis.max()
    ymin = yaxis.min()
    ymax = yaxis.max()

    cv2.circle(green_sign, (((xmax-xmin)//2)+xmin, ((ymax-ymin)//2))+ymin, int(2), (0,0,255), 2)
    
    


    # cv2.imshow("green_box", mask) # white will be in range, black will be out of range
    # cv2.imshow("green_box", edges) # show edges
    cv2.imshow("green_box", green_sign) # show with red lines
    # cv2.imshow("green_box", blank_img) # show with red lines
    cv2.waitKey(0)


if __name__ == "__main__":
    main()