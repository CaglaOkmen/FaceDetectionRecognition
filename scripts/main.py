import cv2
from time import time
from face_detection import FaceDetectionCode
from deep_sort_realtime.deepsort_tracker import DeepSort

class Main:
    def __init__(self, caputure_index):
        self.caputure_index = caputure_index
        self.detection = FaceDetectionCode()
        self.customer = 0
        self.personel = 0

        self.traker = DeepSort()

    def __call__(self):
        cap = cv2.VideoCapture(self.caputure_index)
        assert cap.isOpened(), "Kamera açılamadı!"
        
        # Kameranın çözünürlüğünü ayarlama
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        countID = []
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Görüntü yakalanamadı!")
                break

            # Başlangıç zamanını kaydet (FPS hesaplamak için)
            start_time = time()
            
            # Model ile tahmin yap
            results = self.detection.predict(frame)

            # Tahmin sonuçlarına göre sınır kutuları ve diğer bilgileri icerir
            frame, detections = self.detection.plot_boxes(results, frame)
            
            
            traker_person = self.traker.update_tracks(detections, frame=frame)
            
            for res in traker_person:
                if not res.is_confirmed():
                    break
                track_id = res.track_id
                bbox = res.to_ltrb()
                    
                # Kutuyu çiz ve ID'yi yaz
                cv2.putText(frame, str(track_id), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                print(countID)
                print(res.get_det_class())
                if  track_id not in countID:
                    countID.append(track_id)
                    if res.get_det_class() == 'bilinmeyen':
                        self.customer += 1
                    else:
                        self.personel += 1
                
                cv2.putText(frame, "musteri sayisi: " + str(self.customer), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) ,2)
                cv2.putText(frame, "personel sayisi: " + str(self.personel), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Ekranda görüntüyü göster
            cv2.imshow("YOLO Tahmin", frame)

            # FPS hesapla ve ekrana yazdır
            end_time = time()
            fps = 1 / (end_time - start_time)
            print(f"FPS: {fps:.2f}")

            # 'q' tuşuna basıldığında döngüden çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Kaynakları serbest bırak
        cap.release()
        cv2.destroyAllWindows()

# Kodun çalıştırılması
if __name__ == "__main__":
    Main = Main(0)  # 0, varsayılan kamera
    Main()