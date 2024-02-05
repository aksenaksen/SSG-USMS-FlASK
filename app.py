
from flask import Flask, request

import time
import numpy as np
import requests
import re
import logging
from keras.models import load_model
import threading
import cv2


np.set_printoptions(suppress=True)

model = load_model("keras_model.h5", compile = False)

class_names = open("labels.txt", "r",encoding='UTF8').readlines()

def extract_korean(text):
    korean_pattern = re.compile('[가-힣]+')
    return ''.join(korean_pattern.findall(text))

class_names = [extract_korean(text) for text in class_names]

print(class_names)


logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)

def generate_frames(rtmp_url):
    time.sleep(5)
    rtmp_url1 = 'rtmp://usms-media.serveftp.com:1935/live/'+rtmp_url
    spring_server_url = 'https://usms.serveftp.com/live-streaming/accidents'
    cap = cv2.VideoCapture(rtmp_url1, cv2.CAP_FFMPEG)

    last_message_time = time.time()
    last_prediction_time = time.time()
    prediction_interval = 1

    app.logger.info('check')

    while cap.isOpened():
        ret, frame = cap.read()


        if not ret:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        
        current_time = time.time()
        if current_time - last_prediction_time >= prediction_interval:

#####################################################################################################  Ai 모델

            image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)


            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

            image = (image /127.5) - 1

            prediction = model.predict(image)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            

            

            last_prediction_time = current_time
            
#######################################################################################################


####################################################################################################### 
            if index != 3 and current_time >= last_message_time:                
                if ((index ==0 or index == 1) and int(np.round(confidence_score * 100)) < 94):
                    continue
                index+=3
                app.logger.info(f"Class: {class_name}, streamKey = {rtmp_url}")
                app.logger.info(f"Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%")
                timestamp = int(time.time())
                payload = {
                "streamKey": rtmp_url,
                "behavior": str(index),
                "startTimestamp": timestamp
                }
                headers = {'Content-Type': 'application/json'}
                response = requests.post(spring_server_url, json=payload, headers=headers)
                last_message_time = current_time+300 


    app.logger.info('exit')
    cap.release()
    return


@app.route('/video/streamkey',methods = ['POST'])
def video_feed():
    data = request.form.get('name')
    app.logger.info(data)
    
    thread = threading.Thread(target=generate_frames, args=(data,), daemon=True)
    thread.start()
    return ""
    

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
