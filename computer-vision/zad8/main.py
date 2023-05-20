import cv2

# https://chev.me/arucogen/

plik = 'video.mp4'

if __name__ == '__main__':
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    resizeShape = (100, 100)

    idDict = {
        11: cv2.resize(cv2.imread("./images/frog.png"), resizeShape),
        18: cv2.resize(cv2.imread("./images/terraria.png"), resizeShape),
    }

    vid = cv2.VideoCapture(plik)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mp4', fourcc, 60.0, size)


    while True:
        # Capture the video frame by frame
        ret, frame = vid.read()
        marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(frame)

        if len(marker_corners) > 0:
            marker_ids = marker_ids.flatten()

            for (corner, id) in zip(marker_corners, marker_ids):
                corners = corner.reshape((4, 2))
                top_left, top_right, bottom_right, bottom_left = corners

                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))

                # middle
                cx = (top_left[0] + bottom_right[0]) // 2
                cy = (top_left[1] + bottom_right[1]) // 2

                # image
                if id in idDict.keys():
                    x = cx - resizeShape[0] // 2
                    y = cy - resizeShape[1] // 2

                    frame[y:y + resizeShape[1], x:x + resizeShape[0]] = idDict[id]

        # Display the resulting frame
        cv2.imshow('frame', frame)
        out.write(frame)

        # the 'q' button is set as the quitting button you may use any desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()  # After the loop release the cap object
    out.release()
    cv2.destroyAllWindows()  # Destroy all the windows
