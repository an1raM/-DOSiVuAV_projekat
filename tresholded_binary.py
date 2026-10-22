import numpy as np
import cv2 as cv

from undistort import undistort


def sobel_x_mask(img, thresh=40, ksize=3):
    """Binarna maska vertikalnih ivica (Sobel po X osi)."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobelX = cv.Sobel(gray, ddepth=cv.CV_32F, dx=1, dy=0, ksize=ksize)
    absX = np.absolute(sobelX)
    scaled = np.uint8(255 * absX / np.max(absX))
    _, mask = cv.threshold(scaled, thresh, 255, cv.THRESH_BINARY)
    return mask


def color_mask(img):
    """Binarna maska žute i bijele boje u HSV prostoru."""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    yellow = cv.inRange(hsv, np.array([15, 80, 100]), np.array([35, 255, 255]))
    white = cv.inRange(hsv, np.array([0, 0, 200]), np.array([179, 40, 255]))
    return cv.bitwise_or(yellow, white)


def threshold(img):
    """Kombinuje gradijent i boju u jednu binarnu sliku."""
    return cv.bitwise_or(sobel_x_mask(img), color_mask(img))


if __name__ == '__main__':
    img = undistort(cv.imread('test_images/test1.jpg'))

    sobel = sobel_x_mask(img)
    color = color_mask(img)
    combined = threshold(img)

    cv.imwrite('docs/binary_combo_example.png', np.hstack((sobel, color, combined)))

    cv.imshow('Sobel X | Boja | Kombinovano', np.hstack((sobel, color, combined)))
    cv.waitKey(0)
    cv.destroyAllWindows()