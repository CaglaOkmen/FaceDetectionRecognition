from ultralytics import YOLO
import numpy as np
import torch
import cv2

from face_recognition import FaceRecognitionCode

class FaceDetectionCode:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("usin device: ", self.device)
        self.model = YOLO('human3.pt') # model yüklenmesi
        
        self.recognition = FaceRecognitionCode()

    # Model 
    def predict(self, frame):
        results = self.model(frame)
        return results
    
    # Bulunan yüzlere kutu ve isin yazdırır
    def plot_boxes(self, results, frame):
        xyxys = []
        confidences = []
        detections = [] 

        for result in results:
            boxes = result.boxes.cpu().numpy()
            xyxys.append(boxes.xyxy)
            confidences.append(boxes.conf)

        for i, box_set in enumerate(xyxys):
            # Her kutu (xyxy) seti için döngü
            for j, xyxy in enumerate(box_set):
                x1, y1, x2, y2 = xyxy

                # Yüz bölgesini al
                img = frame[int(y1):int(y2), int(x1):int(x2)]
                img = cv2.resize(img, (160,160)) 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                img = np.expand_dims(img, axis=0)  

                # Vektör karşılaştırma ile isim alma
                name = self.recognition.compare_vector(img)

                # Algılanan yüz için kutu çizimi ve id yazımı
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                cv2.putText(frame, name, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                detections.append([xyxy, confidences[i][j], name])  
            
        return frame, detections
