from imutils import face_utils
import numpy as np
import dlib
import cv2
import random
import time


cap = cv2.VideoCapture(0)
time.sleep(3)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Point to Image
def draw_point(img, p, color):
    cv2.circle(img, p, 1, color, -1, cv2.LINE_AA, 0)


# draw a delaunay triangle
def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


if __name__ == '__main__' :

    while (cap.isOpened()):
        # read webcam
        ret, image = cap.read()
        # flip to image to remove mirror effect
        image = np.flip(image, axis=1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            img_orig = image.copy();#using copy of image(If we do not this line, Then it occur error
            size = image.shape
            rect = (0, 0, size[1], size[0])
            subdiv = cv2.Subdiv2D(rect);

            for p in shape:
                subdiv.insert((p[0], p[1]))

            draw_delaunay(img_orig, subdiv, (255, 255, 255));

            for p in shape:
                draw_point(img_orig, (p[0], p[1]), (0, 0, 255))


            cv2.imshow("Delaunay Triangle", img_orig)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()