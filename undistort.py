import numpy as np
import cv2


calibration = np.load('camera_cal/calib.npz')
mtx = calibration['mtx']
dist = calibration['dist']



def undistort(img):
    """Undistort an image using the lecture approach (getOptimalNewCameraMatrix)."""
    h, w = img.shape[:2]
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    return cv2.undistort(img, mtx, dist, None, newCameraMtx)

if __name__ == '__main__':
    img = cv2.imread('test_images/test1.jpg')
    undistortedImg = undistort(img)

    cv2.imwrite('docs/test1_undistorted.png', np.hstack((img, undistortedImg)))

    cv2.imshow('Original | Undistorted', np.hstack((img, undistortedImg)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()