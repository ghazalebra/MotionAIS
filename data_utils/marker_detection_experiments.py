import cv2
import numpy as np
import os


# input is an image (one frame of movement sequence)
def annotate_single_frame(frame):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(frame.shape)
    src_img = frame
    frame_display, preprocessed_frame = preprocess_test(frame)
    key_points = detect_markers(preprocessed_frame)
    # print(key_points)
    im_with_key_points = cv2.drawKeypoints(frame_display, key_points, np.array([]), (0, 255, 0),
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    key_points = [key_points[j] for j in range(len(key_points))]
    # key_points_x = [key_points[j].pt[0] for j in range(len(key_points))]
    # key_points_y = [key_points[j].pt[1] for j in range(len(key_points))]
    # key_points_y.sort()
    # key_points_x.sort()
    # print(key_points_x)
    # print(key_points_y)
    # cv2.imwrite(save_path + 'frame%d.jpg' % i, src_img)
    # cv2.imwrite(save_path + 'annotated_frame%d.jpg' % i, im_with_key_points)
    return key_points, im_with_key_points


def annotate_frames(folder_path, save_path):
    i = 0
    for filename in os.listdir(folder_path):
        frame = cv2.imread(os.path.join(folder_path, filename))
        key_points, frame_with_key_points = annotate_single_frame(frame)
        cv2.imwrite(save_path + 'annotated_frame%d.jpg' % i, frame_with_key_points)
        file = open('frame%d_landmarks.txt' % i, 'w+')
        for point in key_points:
            file.write(str(point.pt[0]) + ' ' + str(point.pt[1]))
        file.close()
        i += 1


# input is the path to a video
def annotate_video(save_path, video_path):
    cap = cv2.VideoCapture(video_path)

    i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(frame.shape)
        src_img = frame
        frame = preprocess(frame)
        key_points = detect_markers(frame)
        # print(key_points)
        im_with_key_points = cv2.drawKeypoints(frame, key_points, np.array([]), (0, 0, 255),
                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        key_points_x = [key_points[j].pt[0] for j in range(len(key_points))]
        key_points_y = [key_points[j].pt[1] for j in range(len(key_points))]
        key_points_y.sort()
        key_points_x.sort()
        print(key_points_x)
        print(key_points_y)
        cv2.imwrite(save_path + 'frame%d.jpg' % i, src_img)
        cv2.imwrite(save_path + 'annotated_frame%d.jpg' % i, im_with_key_points)
        break

        i += 1

    cap.release()
    cv2.destroyAllWindows()


def preprocess(frame, kernel=None):
    # converting RGB to gray-scale
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cropping
    img = img[800:1400, 350:800]
    # vertical Sobel filter
    sobel_y = np.array([[2, 2, 2],
                        [0, 0, 0],
                        [-2, -2, -2]])
    sobel_x = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]])
    # img = cv2.filter2D(img, -1, sobel_y)
    # filter
    kernel = np.array([
        [1, 1, 1, 1, 1],
        [1, 1, -4, 1, 1],
        [1, -4, -4, -4, 1],
        [1, 1, -4, 1, 1],
        [1, 1, 1, 1, 1]
    ])
    # img = cv2.filter2D(img, -1, kernel)

    # Blurring
    # blurred = cv2.medianBlur(img, 5)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # blurred = img

    # Threshold
    ret1, thr1 = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY_INV)
    thr2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                 cv2.THRESH_BINARY, 11, 2)
    thr3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                 cv2.THRESH_BINARY, 11, 2)
    ret2, thr4 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thr2 = 255 - cv2.filter2D(thr2, -1, kernel)
    # thr4 = blurred

    # thr4 = cv2.GaussianBlur(thr4, (5, 5), 0)

    # thr1 = cv2.filter2D(thr1, -1, kernel)
    # thr1 = cv2.GaussianBlur(thr1, (5, 5), 0)

    return thr4


def preprocess_test(frame):
    # converting to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cropping the image
    gray = gray[800:1400, 350:850]
    clahe = cv2.createCLAHE(clipLimit=5)
    clahe_img = clahe.apply(gray) + 30
    # blurring
    blurred = cv2.medianBlur(gray, 5)
    # threshold
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # threshold = cv2.threshold(threshold, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    threshold = cv2.bitwise_not(threshold)
    # removing the horizontal lines
    # cols = threshold.shape[1]

    # horizontal_size = int(cols)
    # horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT,
    #                                                  (10, 1))
    # vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT,
    #                                                (1, 10))
    circle_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # circle_structure = np.array([[0, 0, 0, 0, 0],
    #                              [0, 1, 1, 1, 0],
    #                              [0, 1, 1, 1, 0],
    #                              [0, 1, 1, 1, 0],
    #                              [0, 0, 0, 0, 0],
    #                              ], np.uint8)
    # lateral_structure_right = np.diag(np.full(8, 1, np.uint8))
    # lateral_structure_left = np.flip(lateral_structure_right, axis=1)
    # print(line_structure)
    # Apply morphology operations
    # horizontal = cv2.erode(threshold, horizontal_structure)
    # horizontal = cv2.dilate(horizontal, horizontal_structure, iterations=1)

    # vertical = cv2.erode(threshold, vertical_structure)
    # vertical = cv2.dilate(vertical, vertical_structure, iterations=1)
    #
    # lateral_right = cv2.erode(threshold, lateral_structure_right)
    # lateral_right = cv2.dilate(lateral_right, lateral_structure_right, iterations=1)
    #
    # lateral_left = cv2.erode(threshold, lateral_structure_left)
    # lateral_left = cv2.dilate(lateral_left, lateral_structure_left, iterations=1)
    #
    # # opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, circle_structure, iterations=1)
    #
    # frame_updated = threshold - horizontal - vertical
    #
    frame_updated = threshold
    circles = cv2.erode(frame_updated, circle_structure, iterations=1)
    circles = cv2.dilate(circles, circle_structure, iterations=2)

    return clahe_img, 255 - circles



def detect_markers(frame, params=None):
    params = cv2.SimpleBlobDetector_Params()

    # params.minThreshold = 230
    # params.maxThreshold = 255
    #
    # # Filter by Area.
    params.filterByArea = True
    params.minArea = 50
    # # params.maxArea = 40
    #
    # # Filter by Circularity
    # params.filterByCircularity = True
    # params.minCircularity = 0.8
    #
    # # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.8
    #
    # # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.5
    #
    # params.filterByDistance = True
    params.minDistBetweenBlobs = 50
    # params.maxDistBetweenBlobs = 100

    # print(params.minThreshold, params.maxThreshold)
    detector = cv2.SimpleBlobDetector_create(params)
    key_points = detector.detect(frame)
    return key_points


def refine_markers(key_points):
    np.sort(key_points)


if __name__ == '__main__':
    file_path = 'Data/Participant01/autocorrection/take1/frames/auto_01_014869_I_76.jpg'
    file_path1 = 'Data/Participant01/autocorrection/take1/frames/auto_01_014786_I_17.jpg'
    file_path2 = 'Data/Participant01/autocorrection/take1/frames/auto_01_014854_I_66.jpg'
    file_path3 = 'Data/Participant01/autocorrection/take1/frames/auto_01_014805_I_31.jpg'
    file_path4 = 'Data/Participant01/autocorrection/take1/frames/auto_01_014812_I_36.jpg'
    file_path5 = 'Data/Participant01/autocorrection/take1/frames/auto_01_014854_I_66.jpg'
    frame = cv2.imread(file_path3)
    # thr1, thr2, thr3, thr4 = preprocess(frame)
    # output = frame
    # circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1.3, 50,
    #                            param1=50, param2=30, minRadius=0, maxRadius=0)
    # if circles is not None:
    #     # Get the (x, y, r) as integers
    #     circles = np.round(circles[0, :]).astype("int")
    #     print(circles)
    #     # loop over the circles
    #     for (x, y, r) in circles:
    #         cv2.circle(output, (x, y), r, (0, 255, 0), 2)
    # annotated_frame1 = annotate_single_frame(thr1)
    # annotated_frame2 = annotate_single_frame(thr2)
    # annotated_frame3 = annotate_single_frame(thr3)
    # annotated_frame4 = annotate_single_frame(thr4)
    # annotated1 = np.concatenate((annotated_frame1, annotated_frame2), axis=1)
    # annotated2 = np.concatenate((annotated_frame3, annotated_frame4), axis=1)
    # annotated = np.concatenate((annotated1, annotated2), axis=0)
    # new_size = (int(annotated.shape[0]/2), int(annotated.shape[1]/1.5))
    # print(new_size)
    # annotated_resized = cv2.resize(annotated, new_size)

    # frame_preprocessed = preprocess_test(frame)
    # new_size = (int(frame_preprocessed.shape[0] / 2), int(frame_preprocessed.shape[1] / 1.5))
    # print(new_size)
    landmarks, annotated_frame = annotate_single_frame(frame)
    # frame_preprocessed_resized = cv2.resize(frame_preprocessed, new_size)

    cv2.imshow('Annotated Frame', annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # save_path = os.path.join(os.getcwd(), 'annotated/')
    # folder_path = 'Data/Participant01/autocorrection/take1'
    # annotate_frames(folder_path, save_path)
