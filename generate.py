import os
import runpy

import cv2 as cv
import numpy as np

# Which test image is used for each result (change if you want a different example)
CALIB_EXAMPLE = 'camera_cal/calibration1.jpg'
ROAD_EXAMPLE = 'test_images/test1.jpg'               # undistortion example
BINARY_EXAMPLE = 'test_images/test6.jpg'             # thresholded binary example
STRAIGHT_EXAMPLE = 'test_images/straight_lines2.jpg' # perspective transform example
RESULT_EXAMPLE = 'test_images/test2.jpg'             # fit + final output (a curved road shows the radius best)

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

    # Step 0: calibration (only if it has not been done yet).
    # calibration.py has no __main__ guard, so it is executed with runpy, and it must run
    # BEFORE undistort.py is imported, because undistort.py loads calib.npz at import time.
    if not os.path.exists(CALIB_FILE):
        print('calib.npz not found, running calibration.py ...')
        runpy.run_path('calibration.py', run_name='__main__')
        cv.destroyAllWindows()

    # These imports are done here on purpose (see above)
    from undistort import undistort
    from thresholded_binary import threshold
    from perspective import warp, SRC, DST
    from lane_detection import process_image

    # 1. Calibration example: original | undistorted (plain mtx, as in calibration.py)
    calibration = np.load(CALIB_FILE)
    mtx, dist = calibration['mtx'], calibration['dist']
    img = read(CALIB_EXAMPLE)
    save('undistort_output.png', np.hstack((img, cv.undistort(img, mtx, dist, None, mtx))))

    # 2. Road image: original | undistorted (with the pipeline's undistort function)
    img = read(ROAD_EXAMPLE)
    save('test1_undistorted.png', np.hstack((img, undistort(img))))

    # 3. Thresholded binary image (final result only)
    save('binary_combo_example.png', threshold(undistort(read(BINARY_EXAMPLE))))

    # 4. Perspective transform: source trapezoid | warped image with destination rectangle
    img = undistort(read(STRAIGHT_EXAMPLE))
    annotated = img.copy()
    cv.polylines(annotated, [SRC.astype(np.int32)], True, (0, 0, 255), 2)
    warped = warp(img)
    cv.polylines(warped, [DST.astype(np.int32)], True, (0, 0, 255), 2)
    save('warped_straight_lines.png', np.hstack((annotated, warped)))
    save('binary_warped.png', warp(threshold(img)))

    # 5 and 6. Lane pixels + polynomial fit, and the final result on the road
    result, debug = process_image(read(RESULT_EXAMPLE))
    save('color_fit_lines.png', debug)
    save('example_output.png', result)

    print('Done. All images are in the "docs" folder.')


if __name__ == '__main__':
    main()