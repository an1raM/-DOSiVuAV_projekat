import numpy as np
import cv2
import glob

cols = 9
rows = 6

criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)


objectPoints = np.zeros((rows * cols, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

objectPointsArray = []
imgPointsArray = []

for path in glob.glob('camera_cal/calibration*.jpg'):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objectPointsArray.append(objectPoints)
        imgPointsArray.append(corners)

        cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
        cv2.imshow('chess board', img)
        cv2.waitKey(200)
    else:
        print('Table not found on:', path)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
np.savez('camera_cal/calib.npz', mtx=mtx, dist=dist)

error = 0
for i in range(len(objectPointsArray)):
    imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
    detected = imgPointsArray[i].reshape(-1, 2).astype(np.float32)
    projected = imgPoints.reshape(-1, 2).astype(np.float32)
    error += cv2.norm(detected, projected, cv2.NORM_L2) / len(projected)

print("Total error:", error / len(objectPointsArray))

img = cv2.imread('camera_cal/calibration1.jpg')
undist = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('docs/undistort_output.png', np.hstack([img, undist]))