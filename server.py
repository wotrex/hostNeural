import cv2
import os
import numpy as np
import time
import pickle
import socket
import imagezmq

configPath = os.path.join("ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
modelPath =  os.path.join("frozen_inference_graph.pb")
classesPath = os.path.join("coco.names")

ip = socket.gethostbyname(socket.gethostname())
port = 7777
print('Running on IP: '+ ip)
print('Running on port: '+ str(port))
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((ip, port))
connections = []
s.listen(1)
image_hub = imagezmq.ImageHub()
net = cv2.dnn_DetectionModel(modelPath, configPath)
net.setInputSize(200,200)
net.setInputScale(1.0/300)
net.setInputMean((300, 300, 300))
net.setInputSwapRB(True)

with open(classesPath, 'r') as f:        
        classesList = f.read().splitlines()

print(classesList)
lower_hsv_red = np.array([157,177,122]) 
upper_hsv_red = np .array([179,255,255]) 
c = 0
addr = 0
startTime = 0
while True:
        if c == 0:
              c, addr = s.accept()
        if c != 0 and len(connections) == 0:
                connections.append(c)
                print(c)
                print(str(addr) + " connected.")
        if len(connections) == 0:
                continue
        rpi_name, image = image_hub.recv_image()
        print(image)
        print(rpi_name)
        image_hub.send_reply(b'OK')
        
        currentTime = time.time()
        fps = 1/(currentTime - startTime)
        startTime = currentTime 
        classLabelIDs, confidences, bboxs = net.detect(image, confThreshold = 0.5)

        bboxs = list(bboxs)
        confidences = list(np.array(confidences).reshape(1,-1)[0])
        confidences = list(map(float, confidences))
        
        bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold = 0.5, nms_threshold = 0.2)
        objects = []
        if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):
                        bbox = bboxs[np.squeeze(bboxIdx[i])]
                        classConfidence = confidences[np.squeeze(bboxIdx[i])]
                        classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                        classLabel = classesList[classLabelID - 1]

                        x, y, w, h = bbox

                        objects.append([bbox,classLabel])
                        coord = "X:" + str(x) + " Y:" + str(y)
                        displayText = "{}:{:.4f}".format(classLabel, classConfidence)
                        if classLabel == 'traffic light':
                                traffic_light = np.zeros((image.shape[0],image.shape[1],image.shape[2]), dtype='uint8')
                                traffic_light = image[y:y+h, x:x+w]
                                traffic_light = cv2.cvtColor(traffic_light, cv2.COLOR_BGR2HSV)
                                mask_red = cv2.inRange(traffic_light,lower_hsv_red,upper_hsv_red) 
                                red_blur = cv2.medianBlur(mask_red, 7)
                                red_color = np.max(red_blur) 
                                if red_color == 255:
                                    print("red")
                                cv2.imshow("Traffic", traffic_light)
                        cv2.rectangle(image, (x,y), (x+w, y+h), color=(255,255,255), thickness=2)
                        cv2.putText(image, displayText, (x + (w-10), y), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
                        cv2.putText(image, coord, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
        cv2.putText(image, str(int(fps)) + " fps", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)                
        cv2.imshow("Result", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
                break
        objects = pickle.dumps(objects)
        try:
                c.sendall(objects)
        except:
                connections = []
                c.close()
                c = 0
                print(str(addr) + " disconnected.")
                cv2.destroyAllWindows()
                continue

cv2.destroyAllWindows()
                                                  
