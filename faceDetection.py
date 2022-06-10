import cv2                          #對影像上傳、改色用的
import mediapipe as mp              #視覺辨識本人
import matplotlib.pyplot as plt     #拿來顯示單張圖片用的
import os                           #找路徑用

#把該載入的工具載一載
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

#先決定素材要是甚麼
a = input("輸入代號")

if a == "video":
    #讀取影片路徑
    dir = os.path.dirname(os.path.abspath(__file__))
    filename = 'test.mp4'
    full_path = os.path.join(dir, filename)
    cap = cv2.VideoCapture(full_path)
elif a == "cam":
    #抓鏡頭
    cap = cv2.VideoCapture(0)
elif a == "pic":
    #讀取圖片路徑
    dir = os.path.dirname(os.path.abspath(__file__))
    filename = 'face.jpg'
    full_path = os.path.join(dir, filename)
    #print(full_path)
    image2 = cv2.imread(full_path)
    #把圖片修一下
    image2 = cv2.cvtColor(cv2.flip(image2, 1), cv2.COLOR_BGR2RGB)
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        image2.flags.writeable = False
        #上面的with不是很熟  下面的process是偵測重點
        results = face_detection.process(image2)
        #再對影像處理
        image = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
        image2.flags.writeable = True
        if results.detections:
            #抓影像的資料
            for face_no, face in enumerate(results.detections):
                print(f'FACE NUMBER: {face_no+1} ')
                print('===============================\n')

                print(f'FACE CONFIDENCE: {round(face.score[0], 2)} \n')

                face_data = face.location_data

                print(f'【FACE BOUNDING BOX】\n{face_data.relative_bounding_box}')
                for i in range(6):
                    print(f'{mp_face_detection.FaceKeyPoint(i).name}:')
                    print(f'{face_data.relative_keypoints[mp_face_detection.FaceKeyPoint(i).value]}')
            #把影像拿出來畫點點
            for detection in results.detections:
                #mp_drawing.draw_detection(image2, detection)
                mp_drawing.draw_detection(image=image2, detection=detection, 
                                 keypoint_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0),
                                                                              thickness=5,
                                                                              circle_radius=5))
        plt.axis('off');plt.imshow(image2);plt.show()
    exit()
else:
    exit()


with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    #抓攝影機和影片
    while cap.isOpened():
        #把影像從影片抓出來
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            if a == "video":
                break
            else:
                continue
        
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
        cv2.imshow('MediaPipe Face Detection 11', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()