import cv2
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import time


video_obj=cv2.VideoCapture(0)

face_detect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

frame_width = int(video_obj.get(3))
frame_height = int(video_obj.get(4))
size = (frame_width, frame_height)

result = cv2.VideoWriter('result/output_video.avi', 
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        10, size)


mask_nomask_model=load_model("artifacts/mask_nomask_model.h5")
human_nohuman_model=load_model("artifacts/human_nohuman_model.h5")
print("Model loaded")

if video_obj.isOpened()==False:
    print("Error opening camera")
    status=False

status=True
while status:
    ret,frame=video_obj.read()
    frame1= Image.fromarray(frame, 'RGB')
    frame1= frame1.resize((224,224))
    frame1= np.array(frame1)
    frame1 = np.expand_dims(frame1, axis=0)
    print(np.array(frame1).shape)
    mask_outt=mask_nomask_model.predict(frame1)
    # hn_outt=human_nohuman_model.predict(frame1)
    gray_scale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray_scale,1.3,5)

    frame = cv2.flip(frame, 1)
    for index,(x,y,w,h) in enumerate(faces):
        print(x,y,w,h)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame,f"Person{index+1}",(x+w+2,y),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0, 0, 255))
    if len(faces)!=0:
        if np.argmax(mask_outt)==0:
            cv2.putText(frame,"No Mask",(0,400),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0, 0, 255))
            
        elif np.argmax(mask_outt)==1:
            cv2.putText(frame,"Masked",(0,400),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0, 255, 0))

    if len(faces)==0:
        cv2.putText(frame,"No Person",(400,400),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0, 0, 255))
        
    # if np.argmax(hn_outt)==0:
    #     cv2.putText(frame,"No Person",(400,400),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0, 0, 255))
        
    # elif np.argmax(hn_outt)==1:
    #     cv2.putText(frame,"Person_Found",(400,400),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0, 255, 0))
        

    result.write(frame)
    cv2.imshow("Face Mask Detection",frame)

    k=cv2.waitKey(30)
    if k==ord("q"):
        break

result.release()
video_obj.release()
cv2.destroyAllWindows()