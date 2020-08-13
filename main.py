# import necessary packages
import cv2
import os
import argparse
import numpy as np
import statistics


def display_lines(image, lines, flip=False):
    line_image = np.zeros_like(image)
    right = []
    left = []
    middle = []

    if flip:
        line_image = cv2.flip(line_image, 1)
        lines[:, 0] = image.shape[1] - lines[:, 0]
        lines[:, 2] = image.shape[1] - lines[:, 2]
    if lines is not None:
        for x1, y1, x2, y2, inter in lines:
            if (x2 > image.shape[1]/2 + image.shape[1]/4 + image.shape[1]/8):
                right.append([x1, y1, x2, y2, inter])
            elif (x1 < image.shape[1]/5):
                left.append([x1, y1, x2, y2, inter])
            else:
                middle.append(inter)

        # avg_max = max(middle)
        # avg_min = min(middle)

        """
        Using median is a better option, since avg_median would'nt fluctuate
        much, even if cars/bikes and other vehicles come in the frame of
        the video, max and min deviate too much - due to presence of erroneous
        edges.
        """
        try:
            avg_median = statistics.median(middle)
        except:
            avg_median = image.shape[1]/2

        try:

            inter_left = np.argmax(np.array(left)[:, 4])
            inter_right = np.argmin(np.array(right)[:, 4])
            a = right[inter_right]
            b = left[inter_left]

            # Line for right lane
            cv2.line(line_image, (int(a[0]), int(a[1])),
                     (int(avg_median), int(0)), (255, 0, 0), 10)

            # Line for left edge
            cv2.line(line_image, (int(b[0]), int(b[1])),
                     (int((avg_median + b[4])/2), int(0)), (255, 0, 0), 10)

            # top frame line of polygon encompassing road.
            cv2.line(line_image, (int((avg_median + b[4])/2),
                    int(20)), (int(image.shape[1]),
                    int(20)), (255, 0, 0), 10)
            # Bottom frame Line
            cv2.line(line_image, (int(b[0]), int(image.shape[0] - 10)),
                     (int(a[0]), int(image.shape[0]-10)), (255, 0, 0), 10)
        except:
            pass
    return line_image


def create_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2, intercept])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    my_lines = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        # It will fit the polynomial and the intercept and slope
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            right_fit.append((slope, intercept))
        else:
            left_fit.append((slope, intercept))

    flip = False

    if len(right_fit) > len(left_fit):
        for right in right_fit:
            right_line = create_coordinates(image, right)
            my_lines.append(right_line)
    else:
        for left in left_fit:
            left_line = create_coordinates(image, left)
            my_lines.append(left_line)
        flip = True

    return (np.array(my_lines), flip)


def canny_edge_detector(image):

    # Convert the image color to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Reduce noise from the image
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    cv2.imwrite('canny_edge1.jpg', canny)
    return canny


def region_of_interest(image):
    """
    For our usecase it can be 3 options
    slope being both positive/negative or
    one positive and one negative and
    accordingly, we must check for the
    polygon

    If image has both positive slopes(see majority, to 
    combat the case of false edges due to cars/plants,etc
    ) then, camera is left facing, if it has both
    positive slopes then it is right facing,
    if the camera has both negative and
    positive slopes, then it can be front facing- which 
    isn't a case we'd require, necessarily, since camera
    is on pole.
    """
    """
    # For self driving car with camera in front of vehicle
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
        ])
    """
    imshape = image.shape
    two = [0, imshape[0]/2 + imshape[0]/8]  # bottom second
    one = [0, imshape[0]]  # bottom left
    three = [imshape[1]/2 + imshape[1]/8, 0]  # top first
    four = [imshape[1], 0]  # top right corner
    five = [imshape[1], imshape[0]/2 + imshape[0]/8]
    six = [imshape[1]/2 + imshape[1]/4 + imshape[1]/8, imshape[0]]

    polygons = np.array([one, two, three,
                          four, five, six], dtype=np.int32)

    mask = np.zeros_like(image)

    # Fill poly-function deals with multiple polygon
    cv2.fillPoly(mask, [polygons], 255)

    # Bitwise operation between canny image and mask image
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def main(media_path, video=False):
    # get the frames from the media
    if video:
        cap = cv2.VideoCapture(media_path)

        # Get the Default resolutions required
        # to write the images using cv2.VideoWriter

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # Define the codec and filename.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output1.avi', fourcc, 10, (frame_width,frame_height))

        # Process the frames in a loop and display the output
        while(cap.isOpened()):
            ret_value, frame = cap.read()

            if ret_value:
                # Canny edge detection, after conversion to grayscale and
                # using Gaussian filter to reduce false edges.
                canny_image = canny_edge_detector(frame)

                # mask the canny image with our own region of interest
                cropped_image = region_of_interest(canny_image)

                # Get the straightlines from probabilistic houghlines transform
                lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100,
                                    np.array([]), minLineLength=40,
                                    maxLineGap=5)

                averaged_lines, flip = average_slope_intercept(frame, lines)
                line_image = display_lines(frame, averaged_lines, flip)
                combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
                cv2.imshow('output', combo_image); cv2.waitKey(1)
                out.write(combo_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        out.release()

        # destroy all the windows that is currently on
        cv2.destroyAllWindows()
    else:
        frame = cv2.imread(media_path)
        canny_image = canny_edge_detector(frame)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100,
                            np.array([]), minLineLength=60,
                            maxLineGap=5)
        averaged_lines, flip = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines, flip)
        
        if flip:
            frame = cv2.flip(frame, 1)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("output", combo_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", action="store_true",
                        help="by default media is assumed to \
                        be an image, pass this option for video")
    parser.add_argument("-o", "--media_path", required=True,
                        help="video/image file path relative to current directory \
                        or full path")
    args = parser.parse_args()
    kwargs = vars(args)

    # RaiseError if the filepath does not exist.
    if os.path.exists(kwargs['media_path']):
        kwargs['media_path'] = os.path.abspath(kwargs['media_path'])
        main(**kwargs)
    else:
        raise FileNotFoundError("media filepath does not exist")
