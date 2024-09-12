import numpy as np
import os
import cv2

class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)  # Hedef boyut
        self.X = [] # Yüz resimleri
        self.Y = [] # Etiketler (kişiler)

    def resize_face(self, image): # Yeniden boyutlandırır
        return cv2.resize(image, self.target_size)

    # Yüz resimleri alır, düzenler ve döndürür
    def load_faces(self, dir):
        FACES = [] 
        for im_name in os.listdir(dir):
            try:
                path = os.path.join(dir, im_name)
                face = cv2.imread(path)
                if face is not None:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Renk dönüşümü
                    face_resized = self.resize_face(face)  # Boyutlandırma
                    FACES.append(face_resized)
            except Exception as e:
                print(f"Error loading image {im_name}: {e}")
        return FACES
    
    # Sınıfları ve yüz görüntülerini yükler
    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            if os.path.isdir(path):
                FACES = self.load_faces(path)
                labels = [sub_dir for _ in range(len(FACES))]
                print(f"Loaded successfully: {len(labels)} images from class {sub_dir}")
                self.X.extend(FACES) # Yüzleri ekler
                self.Y.extend(labels) # Etiketleri ekler
        
        return np.asarray(self.X), np.asarray(self.Y)
