import os
import runpy

import cv2 as cv
import numpy as np

CALIB_EXAMPLE = 'camera_cal/calibration1.jpg'
ROAD_EXAMPLE = 'test_images/test1.jpg'
BINARY_EXAMPLE = 'test_images/test6.jpg' 
STRAIGHT_EXAMPLE = 'test_images/straight_lines2.jpg'
RESULT_EXAMPLE = 'test_images/test2.jpg'        

CALIB_FILE = 'camera_cal/calib.npz'
DOCS = 'docs'


def save(name, img):
    path = os.path.join(DOCS, name)
    if not cv.imwrite(path, img):
        raise SystemExit(f'Could not write {path}')
    print('Saved', path)


def read(path):
    img = cv.imread(path)
    if img is None:
        raise SystemExit(f'Could not read image: {os.path.abspath(path)}')
    return img


def main():
    os.makedirs(DOCS, exist_ok=True)

    if not os.path.exists(CALIB_FILE):
        print('calib.npz not found, running calibration.py ...')
        runpy.run_path('calibration.py', run_name='__main__')
        cv.destroyAllWindows()

    from undistort import undistort
    from thresholded_binary import threshold
    from perspective import warp, SRC, DST
    from lane_detection import process_image

    calibration = np.load(CALIB_FILE)
    mtx, dist = calibration['mtx'], calibration['dist']
    img = read(CALIB_EXAMPLE)
    save('undistort_output.png', np.hstack((img, cv.undistort(img, mtx, dist, None, mtx))))

    img = read(ROAD_EXAMPLE)
    save('test1_undistorted.png', np.hstack((img, undistort(img))))

    save('binary_combo_example.png', threshold(undistort(read(BINARY_EXAMPLE))))
    
    img = undistort(read(STRAIGHT_EXAMPLE))
    annotated = img.copy()
    cv.polylines(annotated, [SRC.astype(np.int32)], True, (0, 0, 255), 2)
    warped = warp(img)
    cv.polylines(warped, [DST.astype(np.int32)], True, (0, 0, 255), 2)
    save('warped_straight_lines.png', np.hstack((annotated, warped)))
    save('binary_warped.png', warp(threshold(img)))

    result, debug = process_image(read(RESULT_EXAMPLE))
    save('color_fit_lines.png', debug)
    save('example_output.png', result)

    print('Done. All images are in the "docs" folder.')


if __name__ == '__main__':
    main()