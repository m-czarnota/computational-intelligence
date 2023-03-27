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
    fgbgAdaptiveGaussain = cv2.createBackgroundSubtractorMOG2()
    _, frame = cap.read()

    # ustalenie parametrów dla usuwania cieni
    low_H = 0
    low_S = 0
    low_V = 0
    high_H = 179
    high_S = 50
    high_V = 255
    alpha = 0.6

    first_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = cv2.absdiff(first_gray, gray_frame)
        _, fgmask = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)

        fgbgAdaptiveGaussainmask = fgbgAdaptiveGaussain.apply(frame)

        # utworzenie maski HSV dla usuwania cieni
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_shadow = np.array([low_H, low_S, low_V])
        high_shadow = np.array([high_H, high_S, high_V])
        mask_shadow = cv2.inRange(hsv, low_shadow, high_shadow)
        mask_shadow = cv2.bitwise_not(mask_shadow)
        mask_shadow = cv2.GaussianBlur(mask_shadow, (5, 5), 0)
        mask_shadow = mask_shadow.astype('float32') / 255
        mask_shadow = cv2.merge([mask_shadow, mask_shadow, mask_shadow])

        # usunięcie cieni z oryginalnego obrazu
        frame = alpha * frame + (1 - alpha) * frame * mask_shadow

        cv2.namedWindow('Background Subtraction', 0)
        cv2.namedWindow('Background Subtraction Adaptive Gaussian', 0)
        cv2.namedWindow('Original', 0)
        cv2.namedWindow('HSV Mask', 0)

        cv2.imshow('Background Subtraction', fgmask)
        cv2.imshow('Background Subtraction Adaptive Gaussian', fgbgAdaptiveGaussainmask)
        cv2.imshow('Original', frame)
        cv2.imshow('HSV Mask', mask_shadow[:, :, 0])

        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    zad2()
