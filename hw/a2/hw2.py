import cv2
import numpy as np


def sign_lines(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of lines.

    https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    :param img: Image as numpy array
    :return: Numpy array of lines.
    """
    # HoughLinesP for two points to draw a line segment
    lines = cv2.HoughLinesP(img,
                            rho=1,
                            theta=np.pi/180,
                            threshold=35,
                            minLineLength=5,
                            maxLineGap=5)
    if lines is not None:
        return lines
    return None


def sign_circle(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of circles.
    :param img: Image as numpy array
    :return: Numpy array of circles.
    """
    raise NotImplemented


def sign_axis(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function takes in a numpy array of lines and returns a tuple of np.ndarray and np.ndarray.

    This function should identify the lines that make up a sign and split the x and y coordinates.
    :param lines: Numpy array of lines.
    :return: Tuple of np.ndarray and np.ndarray with each np.ndarray consisting of the x coordinates and y coordinates
             respectively.
    """
    xaxis = np.empty(0, dtype=np.int32)
    yaxis = np.empty(0, dtype=np.int32)
    return xaxis, yaxis

def sign_line_axis(lines, sign=None):
    xaxis = np.empty(0, dtype=np.int32)
    yaxis = np.empty(0, dtype=np.int32)

    if lines is not None:
        for line in lines:
            x0, y0, x1, y1 = line[0]
            if sign == "Green":
                # do stuff for green
                pass

            xaxis = np.append(xaxis, [x0, x1])
            yaxis = np.append(yaxis, [y0, y1])

    return xaxis, yaxis

def identify_traffic_light(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple identifying the location
    of the traffic light in the image and the lighted light.
    :param img: Image as numpy array
    :return: Tuple identifying the location of the traffic light in the image and light.
             ( x,   y, color)
             (140, 100, 'None') or (140, 100, 'Red')
             In the case of no light lit, coordinates can be just center of traffic light
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv', img_hsv)
    # cv2.waitKey(0)
    grey_low = np.array([0,0,30])
    grey_high = np.array([180,50,70])
    grey_mask = cv2.inRange(img_hsv,grey_low,grey_high)
    # cv2.imshow('grey mask',grey_mask)
    # cv2.waitKey(0)
    # Up to here I have only the traffic light
    contours, _ = cv2.findContours(grey_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_x, center_y, name = None, None, None
    if len(contours) > 0:
        traffic_light_contour = max(contours, key=cv2.contourArea) # largest contour is traffic light
        x, y, w, h = cv2.boundingRect(traffic_light_contour)
        roi = img_hsv[y:y+h, x:x+w] # Region of Interest

        red_low = np.array([0, 100, 100])
        red_high = np.array([10, 255, 255])

        yellow_low = np.array([15, 100, 100])
        yellow_high = np.array([30, 255, 255])

        green_low = np.array([40, 100, 100])
        green_high = np.array([90, 255, 255])

        # Apply color masks to the ROI
        red_mask = cv2.inRange(roi, red_low, red_high)
        yellow_mask = cv2.inRange(roi, yellow_low, yellow_high)
        green_mask = cv2.inRange(roi, green_low, green_high)

        # Count non-zero pixels in the ROI
        red_count = cv2.countNonZero(red_mask)
        yellow_count = cv2.countNonZero(yellow_mask)
        green_count = cv2.countNonZero(green_mask)

        # Find which light is lit
        if red_count > yellow_count and red_count > green_count:
            lit_color = "Red"
        elif yellow_count > red_count and yellow_count > green_count:
            lit_color = "Yellow"
        elif green_count > red_count and green_count > yellow_count:
            lit_color = "Green"
        else:
            lit_color = "None"
        center_x = x + w // 2
        center_y = y + h // 2
    else:
        x, y, lit_color = None, None, "None" # default to None
    

    # cv2.imshow('Red Mask', red_mask)
    # cv2.imshow('Yellow Mask', yellow_mask)
    # cv2.imshow('Green Mask', green_mask)
    # cv2.waitKey(0)
    # print(lit_color)

    return (center_x, center_y, lit_color)
        


def identify_stop_sign(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'stop')
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_low = np.array([0, 150, 150])
    red_high = np.array([5, 255, 255])
    mask = cv2.inRange(img_hsv, red_low, red_high)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_x, center_y, name = None, None, None
    if len(contours) > 0:
        stop_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(stop_contour)
        roi = img_hsv[y:y+h, x:x+w] # Region of Interest
        name = "Stop"
        center_x = x + w // 2
        center_y = y + h // 2
    return (center_x, center_y, name)


def identify_yield(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'yield')
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white_low = np.array([0, 0, 200])
    white_high = np.array([180, 20, 255])
    red_low = np.array([0, 150, 150])
    red_high = np.array([5, 255, 255])
    white_mask = cv2.inRange(img_hsv, white_low, white_high)
    red_mask = cv2.inRange(img_hsv, red_low, red_high)
    mask = cv2.bitwise_or(white_mask, red_mask)
    edges = cv2.Canny(mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_x, center_y, name = None, None, None
    for contour in contours: # polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 3: # <-- three sides
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            return (center_x, center_y, 'Yield')


def identify_construction(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'construction')
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_low = np.array([15, 150, 150])
    yellow_high = np.array([19, 255, 255])
    mask = cv2.inRange(img_hsv, yellow_low, yellow_high)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_x, center_y, name = None, None, None
    if len(contours) > 0:
        construction_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(construction_contour)
        roi = img_hsv[y:y+h, x:x+w] # Region of Interest
        name = "Construction"
        center_x = x + w // 2
        center_y = y + h // 2
    return (center_x, center_y, name)



def identify_warning(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'warning')
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_low = np.array([22, 150, 150])
    yellow_high = np.array([30, 255, 255])
    mask = cv2.inRange(img_hsv, yellow_low, yellow_high)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_x, center_y, name = None, None, None
    if len(contours) > 0:
        warning_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(warning_contour)
        roi = img_hsv[y:y+h, x:x+w] # Region of Interest
        name = "Warning"
        center_x = x + w // 2
        center_y = y + h // 2
    return (center_x, center_y, name)


def identify_rr_crossing(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'rr_crossing')
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_low = np.array([20, 100, 100])
    yellow_high = np.array([30, 255, 255])
    mask = cv2.inRange(img_hsv, yellow_low, yellow_high)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    blur = cv2.GaussianBlur(mask, (9, 9), 2)
    edges = cv2.Canny(blur, 50, 150)
    circles = cv2.HoughCircles(blur, 
                               cv2.HOUGH_GRADIENT, 
                               dp=1, minDist=20, 
                               param1=50, param2=30, 
                               minRadius=20, maxRadius=150)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center_x, center_y, radius = i[0], i[1], i[2]
            roi = img_hsv[center_y-radius:center_y+radius, center_x-radius:center_x+radius] # ROI
            if roi.size == 0:
                continue  # skip
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # cross
            lines = cv2.HoughLinesP(roi_gray, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
            if lines is not None:
                xaxis, yaxis = sign_line_axis(lines, sign="RR Crossing")
                if len(xaxis) > 2 and len(yaxis) > 2:
                    return (center_x, center_y, 'RR Crossing')


def identify_services(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'services')
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_low = np.array([100, 150, 100])
    blue_high = np.array([130, 255, 255])
    mask = cv2.inRange(img_hsv, blue_low, blue_high)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True) # polygon
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4: # <- 4 sides
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            return (center_x, center_y, 'Services')


def identify_signs(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of all signs locations and name.
    Call the other identify functions to determine where that sign is if it exists.
    :param img: Image as numpy array
    :return: Numpy array of all signs locations and name.
             [[x, y, 'stop'],
              [x, y, 'construction']]
    """
    results = []
    traffic_light = identify_traffic_light(img)
    if traffic_light is not None:
        results.append(traffic_light)

    stop_sign = identify_stop_sign(img)
    if stop_sign is not None:
        results.append(stop_sign)

    yield_sign = identify_yield(img)
    if yield_sign is not None:
        results.append(yield_sign)

    construction_sign = identify_construction(img)
    if construction_sign is not None:
        results.append(construction_sign)

    warning_sign = identify_warning(img)
    if warning_sign is not None:
        results.append(warning_sign)

    rr_crossing_sign = identify_rr_crossing(img)
    if rr_crossing_sign is not None:
        results.append(rr_crossing_sign)

    services_sign = identify_services(img)
    if services_sign is not None:
        results.append(services_sign)

    if results:
        return np.array(results, dtype=object)
    else:
        return np.array([], dtype=object) # else return empty


def identify_signs_noisy(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of all signs locations and name.
    Call the other identify functions to determine where that sign is if it exists.

    The images will have gaussian noise applied to them so you will need to do some blurring before detection.
    :param img: Image as numpy array
    :return: Numpy array of all signs locations and name.
             [[x, y, 'stop'],
              [x, y, 'construction']]
    """
    img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    img_median = cv2.medianBlur(img_gaussian, 5) # for salt and pepper
    img_filtered = cv2.bilateralFilter(img_median, d=9, sigmaColor=75, sigmaSpace=75) # chat helped with this line
    results = []
    traffic_light = identify_traffic_light(img_filtered)
    if traffic_light is not None:
        results.append(traffic_light)

    stop_sign = identify_stop_sign(img_filtered)
    if stop_sign is not None:
        results.append(stop_sign)

    yield_sign = identify_yield(img_filtered)
    if yield_sign is not None:
        results.append(yield_sign)

    construction_sign = identify_construction(img_filtered)
    if construction_sign is not None:
        results.append(construction_sign)

    warning_sign = identify_warning(img_filtered)
    if warning_sign is not None:
        results.append(warning_sign)

    rr_crossing_sign = identify_rr_crossing(img_filtered)
    if rr_crossing_sign is not None:
        results.append(rr_crossing_sign)

    services_sign = identify_services(img_filtered)
    if services_sign is not None:
        results.append(services_sign)

    if results:
        return np.array(results, dtype=object)
    else:
        return np.array([], dtype=object) # again, else return empty


def identify_signs_real(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of all signs locations and name.
    Call the other identify functions to determine where that sign is if it exists.

    The images will be real images so you will need to do some preprocessing before detection.
    You may also need to adjust existing functions to detect better with real images through named parameters
    and other code paths

    :param img: Image as numpy array
    :return: Numpy array of all signs locations and name.
             [[x, y, 'stop'],
              [x, y, 'construction']]
    """
    raise NotImplementedError # Part 5 (not doing)