import cv2

# https://chev.me/arucogen/

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

resizeShape = (100, 100)

idDict = {
    10: cv2.resize(cv2.imread("./images/kekw.png"), resizeShape),
    # 11: cv2.resize(cv2.imread("img/frankerz.png"), resizeShape)
}


vid = cv2.VideoCapture(0)  # define a video capture object

while True:

    # Capture the video frame by frame
    ret, frame = vid.read()
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

    if len(markerCorners) > 0:
        markerIds = markerIds.flatten()

        for (corner, id) in zip(markerCorners, markerIds):
            corners = corner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # middle
            cX = int((topLeft[0] + bottomRight[0])/2)
            cY = int((topLeft[1] + bottomRight[1])/2)

            # image
            if id in idDict.keys():
                X = int(cX - resizeShape[0]/2)
                Y = int(cY - resizeShape[1]/2)

                frame[Y:Y+resizeShape[1], X:X+resizeShape[0]] = idDict[id]

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the quitting button you may use any desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()  # After the loop release the cap object
cv2.destroyAllWindows()  # Destroy all the windows
