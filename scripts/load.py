import numpy as np
from keras_facenet import FaceNet
from faceloading import FACELOADING

# npz dosyasını yükleyen ve embedding yapan sınıf
class Load_NPZ:
    def __init__(self):
        self.X = [] # Yüz resimleri
        self.Y = [] # Etietler (kişiler)
        self.embeder = FaceNet()
        faceDataset = FACELOADING("facenetData")
        self.X, self.Y = faceDataset.load_classes()

    # Resimleri uygun hale getirip donusturur
    def get_embedding(self, face_img):
        face_img = face_img.astype('float32')
        face_img = np.expand_dims(face_img, axis=0)

        yhat = self.embeder.embeddings(face_img)
        return yhat[0]
    
    def __call__(self):
        EMBEDDED_X = [] # Embeddingleri tutar

        # Resimlerin embeddinglerini alır ve listeye ekler
        for img in self.X:
            EMBEDDED_X.append(self.get_embedding(img))

        EMBEDDED_X = np.asarray(EMBEDDED_X)
        # Yüz embeddinglerini ve etiketlerini dosyalar
        np.savez_compressed('faces_embeddings_done_4classes.npz', EMBEDDED_X, self.Y)
if __name__ == "__main__":
    load_npz = Load_NPZ() 
    load_npz()