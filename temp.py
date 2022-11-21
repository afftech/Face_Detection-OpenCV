
import numpy as np
import cv2
#Загружаем данные личности
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_Andrei.yml')
#тип шрифта
font = cv2.FONT_HERSHEY_SIMPLEX
#Список имени для ID
names= ['None','Andrey']

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#Подключаем видео камеру
cap = cv2.VideoCapture(0)

while(True): 
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
                                            gray,
                                            scaleFactor= 1.2,
                                            minNeighbors= 5,
                                            minSize=(10, 10))
    for (x, y, w, h) in faces:#Выделяем лицо в рамочку
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        id, confidance = recognizer.predict(gray[y:y+h, x:x+w])
        
        #Проверяем,что лицо разпознано
        if(confidance<40):
            is_obj=names[1]
            confidance = " {0}%".format(round(100-confidance))
            
        else:
            is_obj=names[0]
            confidance = " {0}%".format(round(100-confidance))
        print(gray)
        cv2.putText(frame, str(is_obj), (x+5,y-5), font,1, (255,255,255),2)
        cv2.putText(frame, str(confidance),(x+5,y+h-5),font,1,(255,255,0),1)
    cv2.imshow('Video', frame)
    k=cv2.waitKey(30) & 0xFF
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()