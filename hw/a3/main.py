# references: 
# - https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
# - in class example
# - A bit of chat for video stuff

import cv2
from cv2 import aruco
import numpy as np

def detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids

def sort_markers(corners, ids):
    centers = {}
    for corner, id_ in zip(corners, ids.flatten()):
        center_x = int(np.mean(corner[0][:, 0]))
        center_y = int(np.mean(corner[0][:, 1]))
        centers[id_] = (center_x, center_y)
    sorted_ids = sorted(centers.keys())
    return [centers[id_] for id_ in sorted_ids]

def overlay_image(source_img, overlay_img, marker_positions):
    h, w = overlay_img.shape[:2]
    dst_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    src_points = np.float32(marker_positions)
    matrix, _ = cv2.findHomography(dst_points, src_points)

    # warp
    warped_img = cv2.warpPerspective(overlay_img, matrix, (source_img.shape[1], source_img.shape[0]))
    mask = cv2.warpPerspective(np.ones_like(overlay_img, dtype=np.uint8) * 255, matrix, (source_img.shape[1], source_img.shape[0]))

    background = cv2.bitwise_and(source_img, cv2.bitwise_not(mask))
    return cv2.add(background, warped_img)

def process_images(source_images, overlay_image_path, output_dir):
    overlay_img = cv2.imread(overlay_image_path)

    for idx, source_image_path in enumerate(source_images):
        source_img = cv2.imread(source_image_path)
        corners, ids = detect(source_img)
        if ids is not None and len(ids) >= 4:
            marker_positions = sort_markers(corners, ids)
            result_img = overlay_image(source_img, overlay_img, marker_positions)
            output_path = f"{output_dir}/result_image_{idx + 1}.jpg"
            cv2.imwrite(output_path, result_img)

def process_video(input_video, overlay_video, output_video):
    cap_source = cv2.VideoCapture(input_video)
    cap_overlay = cv2.VideoCapture(overlay_video)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap_source.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap_source.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    while cap_source.isOpened() and cap_overlay.isOpened():
        ret_source, source_frame = cap_source.read()
        ret_overlay, overlay_frame = cap_overlay.read()

        if not ret_source or not ret_overlay:
            break

        corners, ids = detect(source_frame)
        if ids is not None and len(ids) >= 4:
            marker_positions = sort_markers(corners, ids)
            result_frame = overlay_image(source_frame, overlay_frame, marker_positions)
        else:
            result_frame = source_frame

        out.write(result_frame)

    cap_source.release()
    cap_overlay.release()
    out.release()

if __name__ == "__main__":
    source_images = ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]
    overlay_image_path = "imgoverlay.jpeg"
    output_image_dir = "output_images"
    process_images(source_images, overlay_image_path, output_image_dir)
    input_video = "vid1.mp4"
    overlay_video = "vidoverlay.mp4"
    output_video = "result_video.mp4"
    process_video(input_video, overlay_video, output_video)
