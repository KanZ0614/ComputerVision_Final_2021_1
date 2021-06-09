import numpy as np


#모듈 생성
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor('/content/drive/MyDrive/Colab Notebooks/shape_predictor_68_face_landmarks.dat')

img = cv2.imread("/content/drive/MyDrive/Colab Notebooks/man_3.jpg")
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = faces = detect(gray)    # 얼굴 영역

points = []
for rect in rects:
    shape = predict(gray, rect)    # 랜드마크
    for i in range(68):
        part = shape.part(i)
        points.append((part.x, part.y))
        

x,y,w,h = cv2.boundingRect(np.float32(points))   # 들로네분할 객체 생성
subdiv2d = cv2.Subdiv2D((x,y,x+w,y+h))

subdiv2d.insert(points)    # 좌표
triangleList = subdiv2d.getTriangleList()

# 들로네 삼각형
h, w = img.shape[:2]
cnt = 0
for t in triangleList :
    pts = t.reshape(-1,2).astype(np.int32)
    if (pts < 0).sum() or (pts[:, 0] > w).sum() or (pts[:, 1] > h).sum():   # 이미지 영역 벗어나는 것 제외(음수)
        print(pts) 
        continue
    cv2.polylines(img, [pts], True, (255, 255,255), 1, cv2.LINE_AA)
    cnt+=1
print(cnt)


cv2_imshow(img)
cv2.waitKey(0)
