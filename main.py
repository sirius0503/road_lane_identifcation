"""
Steps:
1. Capture the video or image from disk
2. Grayscale image conversion
3. Noise reduction using Gaussian Filter
4. Edge Detection using Canny Edge Detection
5. Masking the canny edge image using ROI
6. Find Coordinates of Road Lanes.
7. Fit the coordinates into the canny image
8. Edge Detection Done.
"""

# import necessary packages
import cv2
import os
import argparse


def canny_edge_detector(image):

    # Convert the image color to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Reduce noise from the image
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def main(media_path, video=False):
    # get the frames from the media
    cap = cv2.VideoCapture(media_path)
    while(cap.isOpened()):
        _, frame = cap.read()
        canny_image = canny_edge_detector(frame)
        cv2.imshow("myframe", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # cv2.destroyAllWindows()
        """
        canny_edge_detector
        Region of Interest - 3 different types based on line slopes
        HoughLines Transform
        Display line images
        Weighted addition to original frames
        """
    # close the video file
    cap.release()

    # destroy all the windows that is currently on
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
    if os.path.exists(kwargs['media_path']):
        kwargs['media_path'] = os.path.abspath(kwargs['media_path'])
        main(**kwargs)
    else:
        raise FileNotFoundError("media filepath does not exist")
