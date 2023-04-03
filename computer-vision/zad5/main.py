import cv2
import numpy as np
from matplotlib import pyplot as plt

video = 'video.mp4'
cap = cv2.VideoCapture(video)


def plot_motion_vectors(step=16):
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    # Obliczenie wymiarów obrazu
    h, w = frame1.shape[:2]

    # Tworzenie siatki regularnej
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)

    # Obliczenie przesunięć punktów pomiędzy dwoma obrazami
    prev_pts = np.float32(np.column_stack((x, y)))
    next_pts, status, errors = cv2.calcOpticalFlowPyrLK(frame1, frame2, prev_pts, None)

    # Wyodrębnienie wektorów przesunięć
    vectors = next_pts - prev_pts

    # Tworzenie obrazka w tle
    vis = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    for i in range(len(x)):
        cv2.arrowedLine(vis, tuple(prev_pts[i].astype(int)), tuple(next_pts[i].astype(int)), (0, 255, 0), 1, tipLength=0.3)

    # Wizualizacja
    plt.imshow(vis)
    plt.show()


def sparse_flow_detection(color_channel: str = None) -> None:
    # params for ShiTomasi corner detection
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()

        if color_channel is None:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif color_channel == 'R':
            frame_gray = frame[:, :, 2]
        elif color_channel == 'G':
            frame_gray = frame[:, :, 1]
        else:
            frame_gray = frame[:, :, 0]

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    cap.release()


def dense_flow_detection(color_channel: str = None) -> None:
    ret, frame1 = cap.read()
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    if color_channel is None:
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    elif color_channel == 'R':
        prvs = frame1[:, :, 2]
    elif color_channel == 'G':
        prvs = frame1[:, :, 1]
    else:
        prvs = frame1[:, :, 0]

    while True:
        ret, frame2 = cap.read()

        if color_channel is None:
            next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        elif color_channel == 'R':
            next_frame = frame2[:, :, 2]
        elif color_channel == 'G':
            next_frame = frame2[:, :, 1]
        else:
            next_frame = frame2[:, :, 0]

        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2', rgb)
        k = cv2.waitKey(30) & 0xff

        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', rgb)

        prvs = next_frame

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sparse_flow_detection('R')
    # dense_flow_detection('G')

    # plot_motion_vectors()
