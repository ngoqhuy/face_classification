from classify_emotion_pb2_grpc import AsillaServiceStub
from classify_emotion_pb2 import RequestImage, RequestEmotions
import grpc
import cv2


class PersonGrpcClient:
    def __init__(self, ip, port, client_id="app"):
        self.channel = grpc.insecure_channel("{}:{}".format(ip, port))
        self.person_service_stub = AsillaServiceStub(self.channel)
        self.client_id = client_id
        print("Connect oke")

    def preporcess_emotion(self,_image):
        
        request = RequestImage(image = _image)
        response = self.person_service_stub.preporcess_emotion(request)
        print(response)
    
    def classify_emotion(self, _image):
        request = RequestEmotions(image = _image)
        response = self.person_service_stub.classify_emotion(request)
        print(response)

if __name__ == '__main__':
    
    import argparse
   
    parser = argparse.ArgumentParser(description='AsillaPose Client')

    parser.add_argument('--ip', default="localhost", type=str,
                      help='Ip address of the server')
    parser.add_argument('--port', default=50051, type=int,
                      help='expose port of gRPC server')
  
    args = parser.parse_args()
   
    client = PersonGrpcClient(args.ip, args.port)
    image_path = "D:\IEC_LAB\emodetection\config.ini"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    client.preporcess_emotion(image_bytes)
    client.classify_emotion(image_bytes)