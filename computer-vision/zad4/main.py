import cv2
import numpy as np

plik = 'video.mp4'
cap = cv2.VideoCapture(plik)


def zad1():
    fgbgAdaptiveGaussain = cv2.createBackgroundSubtractorMOG2()
    _, frame = cap.read()

    avg_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    alpha = 10

    while True:
        ret, frame = cap.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = cv2.absdiff(avg_frame, gray_frame)
        _, fgmask = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)

        # noisefld=np.random.randn(frame.shape[0],frame.shape[1])
        # frame[:,:,0]=(frame[:,:,0]+10*noisefld).astype('int')
        # frame[:,:,1]=(frame[:,:,1]+10*noisefld).astype('int')
        # frame[:,:,2]=(frame[:,:,2]+10*noisefld).astype('int')

        fgbgAdaptiveGaussainmask = fgbgAdaptiveGaussain.apply(frame)

        cv2.namedWindow('Background Subtraction', 0)
        cv2.namedWindow('Background Subtraction Adaptive Gaussian', 0)
        cv2.namedWindow('Original', 0)

        cv2.imshow('Background Subtraction', fgmask)
        cv2.imshow('Background Subtraction Adaptive Gaussian', fgbgAdaptiveGaussainmask)
        cv2.imshow('Original', frame)

        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break

        avg_frame = cv2.addWeighted(avg_frame, 1 - alpha, gray_frame, alpha, 0)

    cap.release()
    cv2.destroyAllWindows()


def zad2():
    fgbgAdaptiveGaussian = cv2.createBackgroundSubtractorMOG2()
    _, frame = cap.read()

    first_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)
    n_frames = 15
    background_frames = [first_gray] * n_frames

    # shadow removal parameters
    alpha_shadow = 0.4
    beta_shadow = 100
    tau_h = 10
    tau_s = 50

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for i in range(n_frames - 1):
            background_frames[i] = background_frames[i + 1]
        background_frames[-1] = gray_frame.copy()

        bg_model = np.mean(background_frames, axis=0)
        bg_model = cv2.GaussianBlur(bg_model, (5, 5), 0)

        diff = cv2.absdiff(bg_model.astype(np.uint8), gray_frame)

        _, fgmask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # shadow removal
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_shadow = np.array([0, 0, 0], dtype=np.uint8)
        upper_shadow = np.array([180, 255, 50], dtype=np.uint8)
        shadow_mask = cv2.inRange(hsv_frame, lower_shadow, upper_shadow)

        hue_diff = cv2.absdiff(hsv_frame[:, :, 0], int(lower_shadow[0]))

        shadow_mask[hue_diff < tau_h] = 0
        shadow_mask[hsv_frame[:, :, 1] < tau_s] = 0

        shadow_mask = cv2.dilate(shadow_mask, None, iterations=2)
        shadow_mask = cv2.medianBlur(shadow_mask, 7)

        alpha_shadow_mask = alpha_shadow * shadow_mask
        beta_shadow_mask = beta_shadow * shadow_mask

        fgmask = cv2.bitwise_and(fgmask, cv2.bitwise_not(shadow_mask))
        fgmask = cv2.subtract(fgmask, alpha_shadow_mask.astype(np.uint8))
        fgmask = cv2.subtract(fgmask, beta_shadow_mask.astype(np.uint8))

        fgbgAdaptiveGaussainmask = fgbgAdaptiveGaussian.apply(frame)

        cv2.namedWindow('Background Subtraction', 0)
        cv2.namedWindow('Background Subtraction Moving Average', 0)
        cv2.namedWindow('Background Subtraction Adaptive Gaussian', 0)
        cv2.namedWindow('Shadow Mask', 0)
        cv2.namedWindow('Foreground Mask', 0)
        cv2.namedWindow('Original', 0)

        cv2.imshow('Background Subtraction', fgmask)
        cv2.imshow('Background Subtraction Moving Average', diff)
        cv2.imshow('Background Subtraction Adaptive Gaussian', fgbgAdaptiveGaussainmask)
        cv2.imshow('Shadow Mask', shadow_mask)
        cv2.imshow('Foreground Mask', fgmask)
        cv2.imshow('Original', frame)

        k = cv2.waitKey(1) & 0xff

        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    zad2()
