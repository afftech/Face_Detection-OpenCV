## захват лица для создания обучающей выборки
import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id = "Andrei"
print("\n[INFO] Инициализация захвата лица. Смотрите в камеру...")
count=0
camera = cv2.VideoCapture(0)

while(True): 
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
                                            gray,
                                            scaleFactor= 1.1,
                                            minNeighbors= 7,
                                            minSize=(10, 10))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        count+= 1
        cv2.imwrite(".\DataSet\\user." + str(face_id)+'.'+str(count)+'.jpg', gray[y:y+h, x:x+w])
        cv2.imshow('image',frame)
    cv2.imshow('Video', frame)
    k=cv2.waitKey(100) & 0xFF
    if k==27:
        break
    elif count >=30: 
        break
print("\n Программа Завершена")

camera.release()
cv2.destroyAllWindows()