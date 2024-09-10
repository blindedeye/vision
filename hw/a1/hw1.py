import cv2
import numpy as np


def read_image(image_path: str) -> np.ndarray:
    """
    This function reads an image and returns it as a numpy array
    :param image_path: String of path to file
    :return img: Image array as ndarray
    """
    img = cv2.imread(image_path)
    return img


def extract_green(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the green channel
    :param img: Image array as ndarray
    :return: Image array as ndarray of just green channel
    """
    green_chan = img[:, :, 1]
    return green_chan


def extract_red(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the red channel
    :param img: Image array as ndarray
    :return: Image array as ndarray of just red channel
    """
    red_chan = img[:, :, 2]
    return red_chan


def extract_blue(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the blue channel
    :param img: Image array as ndarray
    :return: Image array as ndarray of just blue channel
    """
    blue_chan = img[:, :, 0]
    return blue_chan


def swap_red_green_channel(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the image with the red and green channel
    :param img: Image array as ndarray
    :return: Image array as ndarray of red and green channels swapped
    """
    swapped_img = img.copy()
    swapped_img[:, :, [1, 2]] = img[:, :, [2, 1]]  # Swap red and green
    return swapped_img


def embed_middle(img1: np.ndarray, img2: np.ndarray, embed_size: (int, int)) -> np.ndarray:
    """
    This function takes two images and embeds the embed_size pixels from img2 onto img1
    :param img1: Image array as ndarray
    :param img2: Image array as ndarray
    :param embed_size: Tuple of size (width, height)
    :return: Image array as ndarray of img1 with img2 embedded in the middle
    """
    h, w = img1.shape[:2]
    embed_w, embed_h = embed_size
    start_w = (w - embed_w) // 2
    start_h = (h - embed_h) // 2

    # Resize img2 to embed_size
    resized_img2 = cv2.resize(img2, (embed_w, embed_h))

    # Embed in the middle of img1
    img1[start_h:start_h + embed_h, start_w:start_w + embed_w] = resized_img2
    return img1


def calc_stats(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the mean and standard deviation
    :param img: Image array as ndarray
    :return: Numpy array with mean and standard deviation in that order
    """
    mean, std_dev = cv2.meanStdDev(img)
    return np.array([mean, std_dev]).flatten()


def shift_image(img: np.ndarray, shift_val: int) -> np.ndarray:
    """
    This function takes an image and returns the image shifted by shift_val pixels to the right.
    Should have an appropriate border for the shifted area:
    https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html

    Returned image should be the same size as the input image.
    :param img: Image array as ndarray
    :param shift_val: Value to shift the image
    :return: Shifted image as ndarray
    """
    raise NotImplementedError()


def difference_image(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    This function takes two images and returns the first subtracted from the second

    Make sure the image to return is normalized:
    https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga87eef7ee3970f86906d69a92cbf064bd

    :param img1: Image array as ndarray
    :param img2: Image array as ndarray
    :return: Image array as ndarray
    """
    diff = cv2.subtract(img2, img1)
    normalized_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return normalized_diff


def add_channel_noise(img: np.ndarray, channel: int, sigma: int) -> np.ndarray:
    """
    This function takes an image and adds noise to the specified channel.

    Should probably look at randn from numpy

    Make sure the image to return is normalized:
    https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga87eef7ee3970f86906d69a92cbf064bd

    :param img: Image array as ndarray
    :param channel: Channel to add noise to
    :param sigma: Gaussian noise standard deviation
    :return: Image array with gaussian noise added
    """
    noise = np.random.randn(*img.shape[:2]) * sigma
    noisy_img = img.copy()
    noisy_img[:, :, channel] = cv2.add(noisy_img[:, :, channel], noise.astype(np.uint8))
    return noisy_img


def add_salt_pepper(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and adds salt and pepper noise.

    Must only work with grayscale images
    :param img: Image array as ndarray
    :return: Image array with salt and pepper noise
    """
    salt_pepper_img = img.copy()
    prob = 0.05  # Probability of noise
    rnd = np.random.rand(*img.shape[:2])

    salt_pepper_img[rnd < prob / 2] = 0  # Pepper noise
    salt_pepper_img[rnd > 1 - prob / 2] = 255  # Salt noise

    return salt_pepper_img


def blur_image(img: np.ndarray, ksize: int) -> np.ndarray:
    """
    This function takes an image and returns the blurred image

    https://docs.opencv.org/4.x/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
    :param img: Image array as ndarray
    :param ksize: Kernel Size for medianBlur
    :return: Image array with blurred image
    """
    return cv2.medianBlur(img, ksize)