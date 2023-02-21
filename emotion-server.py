from classify_emotion_pb2 import RespondImages, RespondEmotions
import cv2, os, numpy as np
import onnx, configparser
from onnx import backend
import onnxruntime as ort
import grpc 
from concurrent import futures
import random 
import classify_emotion_pb2_grpc

class EmotionClassifier(classify_emotion_pb2_grpc.AsillaServiceServicer):

    __instance = None 
    
    def getInstance(config_file):
        if EmotionClassifier.__instance is None:
            EmotionClassifier.__instance = EmotionClassifier(config_file)
        return EmotionClassifier.__instance 

    def __init__(self, config_file) -> None:
        cfg = configparser.ConfigParser()
        cfg.read(config_file)

        self.model_emotion = ort.InferenceSession(cfg['OPENFACE'].get('emotion'))
        self.emotion_table = {0:'neutral', 1:'happiness', 2:'surprise', 3:'sadness', 4: 'anger', 
                    5: 'disgust', 6: 'fear', 7: 'contempt'}

    def preporcess_emotion(self, orig_image):
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (64, 64))
        image=  np.array([image])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        return image

    def classify_emotion(self, orig_image):
        image = self.preporcess_emotion(orig_image)
        input_name =  self.model_emotion.get_inputs()[0].name
        emotions =  self.model_emotion.run(None, {input_name: image})
        emotions = self.emotion_table[emotions[0].argmax()]
        return emotions 

def serve():
    print("service GRPC is running...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    classify_emotion_pb2_grpc.add_AsillaServiceServicer_to_server(
        EmotionClassifier('D:\IEC_LAB\emodetection\config.ini'), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()