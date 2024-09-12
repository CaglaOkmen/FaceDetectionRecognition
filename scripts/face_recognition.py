import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

class FaceRecognitionCode:
    def __init__(self):
        self.facenet = FaceNet()
        self.faces_embeddings = np.load("faces_embeddings_done_4classes.npz")

    def compare_vector(self, new_face_img):
        # Yüz vektörleri ve kişi etiketlerini yükle
        Y = self.faces_embeddings['arr_1']  # Etiketler (kişiler)
        X = self.faces_embeddings['arr_0']  # Vektörler (yüz özellikleri)
        
        # Etiketleri dönüştürmek için LabelEncoder kullanıyoruz
        encoder = LabelEncoder()
        Y_encoder = encoder.fit_transform(Y)

        # KNN modelini oluştur
        knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
        knn.fit(X, Y_encoder)

        # Yeni yüz vektörünü hesapla
        new_face_embedding = self.facenet.embeddings(new_face_img)

        # KNN modelinden tahmin al
        distances, indices = knn.kneighbors(new_face_embedding)

        # Eşik değeri: Eğer en yakın komşunun mesafesi bu değerden büyükse, 
        # tanınmayan kişi olarak kabul edelim
        threshold = 0.9  # Mesafe eşik değeri
        if distances[0][0] > threshold:
            print("Bilinmeyen kişi tespit edildi!")   
            return "bilinmeyen"
        else:
            # Eşik altında ise kişi biliniyor
            prediction = knn.predict(new_face_embedding)
            predicted_name = encoder.inverse_transform(prediction)[0]
            print(f"Tahmin edilen kişi: {predicted_name}")
            return predicted_name 