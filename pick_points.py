import cv2 as cv

from undistort import undistort

img = undistort(cv.imread('test_images/straight_lines1.jpg'))


def on_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'Clicked point: [{x}, {y}]')
        cv.circle(img, (x, y), 4, (0, 0, 255), -1)
        cv.imshow('Pick points', img)


cv.imshow('Pick points', img)
cv.setMouseCallback('Pick points', on_click)
cv.waitKey(0)
cv.destroyAllWindows()