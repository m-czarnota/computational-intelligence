import cv2
import numpy as np


def mean_shift(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_name = video_path.replace('.', '_mean-shift.')
    out = cv2.VideoWriter(out_name, fourcc, 20.0, size)

    ret, frame = cap.read()  # take first frame of the video

    # setup initial location of window
    x, y, w, h = cv2.selectROI(frame, False)  # simply hardcoded the values
    track_window = (x, y, w, h)

    # set up the ROI for tracking
    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    # cv2.term_criteria

    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        
        # Draw it on image
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

        cv2.imshow('img2', img2)
        out.write(img2)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
            
            
def cam_shift(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_name = video_path.replace('.', '_cam-shift.')
    out = cv2.VideoWriter(out_name, fourcc, 20.0, size)

    ret, frame = cap.read()  # take first frame of the video

    # setup initial location of window
    x, y, w, h = cv2.selectROI(frame, False)  # select
    # x, y, w, h = 100, 50, 100, 100 # written
    track_window = (x, y, w, h)

    # set up the ROI for tracking
    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if ret is not True:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply camshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, 255, 2)

        cv2.imshow('img2', img2)
        out.write(img2)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # mean_shift("moving.mp4")
    cam_shift("moving.mp4")

    # mean_shift("scaling.mp4")
    cam_shift("scaling.mp4")
