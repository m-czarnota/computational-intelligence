import cv2
import numpy as np
import os
import random


def plot_motion_vectors(step=16):
    ret, frame1 = cap.read()
    for i in range(4):
        ret, frame2 = cap.read()

    # Obliczenie wymiarów obrazu
    h, w = frame1.shape[:2]

    # Tworzenie siatki regularnej
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)

    # Obliczenie przesunięć punktów pomiędzy dwoma obrazami
    prev_pts = np.float32(np.column_stack((x, y)))
    next_pts, status, errors = cv2.calcOpticalFlowPyrLK(
        frame1, frame2, prev_pts, None)

    # Wyodrębnienie wektorów przesunięć
    vectors = next_pts - prev_pts

    # Tworzenie obrazka w tle
    vis = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    for i in range(len(x)):
        cv2.arrowedLine(
            frame1,
            tuple(prev_pts[i].astype(int)),
            tuple(next_pts[i].astype(int)),
            (0, 255, 255),
            1,
            tipLength=0.3
        )

    # Wizualizacja
    cv2.imshow('Motion vectors', frame1)
    cv2.waitKey()


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
        if not ret:
            print('No frames grabbed!')
            break
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

    cv2.waitKey()  # to delete for testing
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


def getRandom(n, x, y):
    return random.sample(range(x, y+1), n)


def getNotRandom(x, y):
    return np.linspace(x, y, num=y-x+1)


def getFrames(videuo, frames):
    images = []
    for i in frames:
        videuo.set(1, i)
        ret, frame = videuo.read()
        images.append(frame)
    return images


def saveImagesTo(images, path):
    i = 0
    for image in images:
        i += 1
        cv2.imwrite(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), path, f"{i}.jpg"), image)


def generateImages(count):
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frameCount = count

    # generate random frames
    frameNumbers = getRandom(frameCount, 1, int(nframes))
    images = getFrames(cap, frameNumbers)
    saveImagesTo(images, "images/random")

    # generate specific frames
    frameNumbers = getNotRandom(1, frameCount)
    images = getFrames(cap, frameNumbers)
    saveImagesTo(images, "images/not")


def generateVideoFromFolder(path, count, filename):
    directory = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), path)

    images = []
    for i in range(1, count):
        images.append(cv2.imread(os.path.join(directory, str(i)+'.jpg')))
    h, w, ch = images[0].shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), filename), fourcc, 1, (w, h))

    for image in images:
        video.write(image)

    video.release()


def generate_images_for_count(count: int) -> None:
    generateImages(count)

    # Generate Videos from the images
    generateVideoFromFolder("images/not", count, "not.mp4")
    generateVideoFromFolder("images/random", count, "random.mp4")


if __name__ == '__main__':
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    video = 'video.mp4'
    filePath = os.path.join(scriptDir, video)
    cap = cv2.VideoCapture(filePath)

    plot_motion_vectors()
    # sparse_flow_detection('B')
    dense_flow_detection('R')

    # Generate Images for the videos
    # generate_images_for_count(1000)

    scriptDir = os.path.dirname(os.path.abspath(__file__))
    video = 'not.mp4'
    filePath = os.path.join(scriptDir, video)
    cap = cv2.VideoCapture(filePath)

    sparse_flow_detection()
