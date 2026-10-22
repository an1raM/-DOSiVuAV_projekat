import numpy as np
import cv2 as cv

from undistort import undistort
from thresholded_binary import threshold

SRC = np.float32([[598, 460], [706, 460], [1071, 718], [247, 719]])

DST = np.float32([[320, 0], [960, 0], [960, 720], [320, 720]])

M = cv.getPerspectiveTransform(SRC, DST)
M_INV = cv.getPerspectiveTransform(DST, SRC)


def warp(img):
    """Transform an image to the bird's-eye view."""
    h, w = img.shape[:2]
    return cv.warpPerspective(img, M, (w, h), flags=cv.INTER_LINEAR)


if __name__ == '__main__':
    img = undistort(cv.imread('test_images/straight_lines2.jpg'))

    annotated = img.copy()
    cv.polylines(annotated, [SRC.astype(np.int32)], True, (0, 0, 255), 2)

    warped = warp(img)
    cv.polylines(warped, [DST.astype(np.int32)], True, (0, 0, 255), 2)

    cv.imwrite('docs/warped_straight_lines.png', np.hstack((annotated, warped)))

    binary_warped = warp(threshold(img))
    cv.imwrite('docs/binary_warped.png', binary_warped)

    cv.imshow('Source | Warped', np.hstack((annotated, warped)))
    cv.imshow('Binary warped', binary_warped)
    cv.waitKey(0)
    cv.destroyAllWindows()