import cv2
import dlib
from google.colab.patches import cv2_imshow

# 모듈생성
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor('/content/drive/MyDrive/Colab Notebooks/shape_predictor_68_face_landmarks.dat')

img = cv2.imread("/content/drive/MyDrive/Colab Notebooks/man_3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 얼굴 검출
faces = detect(gray)
for rect in faces:
    # 얼굴 영역을 좌표로 변환
    x,y = rect.left(), rect.top()
    w,h = rect.right()-x, rect.bottom()-y
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # 랜드마크 검출
    shape = predict(gray, rect)
    for i in range(68):
        # 좌표 추출
        part = shape.part(i)
        cv2.circle(img, (part.x, part.y), 2, (0, 0, 255), -1)
        cv2.putText(img, str(i), (part.x, part.y), cv2.FONT_HERSHEY_PLAIN, \
                                         0.5,(255,255,255), 1, cv2.LINE_AA)

cv2_imshow(img)
cv2.waitKey(0)
