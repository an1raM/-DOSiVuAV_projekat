import numpy as np
import cv2 as cv

from undistort import undistort


def sobel_x_mask(img, thresh=40, ksize=3):
    """Binary mask of vertical edges (Sobel along the X axis)."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    sobelX = cv.Sobel(gray, ddepth=cv.CV_32F, dx=1, dy=0, ksize=ksize)
    absX = np.absolute(sobelX)
    scaled = np.uint8(255 * absX / np.max(absX))
    _, mask = cv.threshold(scaled, thresh, 255, cv.THRESH_BINARY)
    return mask


def color_mask(img):
    """Binary mask of yellow and white lane markings in HSV."""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    yellow = cv.inRange(hsv, np.array([15, 80, 100]), np.array([35, 255, 255]))
    white = cv.inRange(hsv, np.array([0, 0, 200]), np.array([179, 30, 255]))
    return cv.bitwise_or(yellow, white)


def threshold(img):
    """Combine gradient and color into one binary image and remove small specks."""
    combined = cv.bitwise_or(sobel_x_mask(img), color_mask(img))
    kernel = np.ones((3, 3), np.uint8)
    return cv.morphologyEx(combined, cv.MORPH_OPEN, kernel)

if __name__ == '__main__':
    img = undistort(cv.imread('test_images/test6.jpg'))

    sobel = sobel_x_mask(img)
    color = color_mask(img)
    combined = threshold(img)

    cv.imwrite('docs/binary_combo_example.png', np.hstack((sobel, color, combined)))

    cv.imshow('Sobel X | Color | Combined', np.hstack((sobel, color, combined)))
    cv.waitKey(0)
    cv.destroyAllWindows()