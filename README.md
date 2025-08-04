# 👤 İnsan Yüzü Algılama ve Tanıma Projesi 
## 🎯 Projenin Hedefi, Benden Istenenler
- Personelleri yüz tanıma yoluyla algılayıp izlemek.
- Müşterilerin varlığını algılayıp, bulundukları süreyi takip etmek.
---
## ⚙️ Karşılaşılan Zorluklar ve Çözümleri
### 1. Veri Seti Yetersizliği
Başlangıçta veri setinin azlığı ve çeşitliliği düşük olmasından dolayı personeller arası karışıklıklar yaşandı

**Çözümü**
* 📈 Veri setindeki resimlerin sayısı arttırıldı.
* 🖼️ Bazı resimler üzeri ise renk, eğim gibi çeşitli düzenlemeler yapılarak çeşitlilik sağlandı.

Bu sayede personelleri birbirine karıştırması büyük oranda azaldı.

### 2. Model Eğitim Süresi
Veri seti eğitimlerinde Google colob üzerinden modeli eğitirken GPU kullanım süresinin dolması ve yerel bilgisayar üzerinden devam ederken ise eğitimin çok uzun sürmesi 

**Çözümü**
* 🔄 Başka Google hesaplarıyla devam edilmeye çalışıldı. Ancak bu geçici bir çözüm oldu ve bu sorun tam olarak çözülemedi.

### 3. Personel ve Müsteri Karışıklığı
YOLOv8 modeli ile yapılan eğitimlerde, müsterilerin personel olarak algılaması ya da bir kişininhem müşteri hem personel olarak sınıflandırması gibi sorunlar yaşandı.

**Çözümü**
* 🟦 **YOLOv8** modeli ile yanlızca yüz algılama yapıldı.
* 💡 Yüz tanıma aşamasında **FaceNet** ve **KNN** algoritmaları kullanıldı. Bu, karışıklığı büyük ölçüde giderdi.

### 4. FPS Değerinin Düşük Olması
yolov8n.pt nin FPS değerin 20 lerde iken kendi eğittiğim modelin FPS değeri 10 larda idi. Ayrıca diğer işlemlerde eklenince bu değer 3 lere düşüyordu.

**Çözümü**
* 🚀 Model için kullanılan veriler gözden geçirildi ekleme yapılıp yeniden eğitildi. Bu sayede Modelin FPS değeri 20 lere çıktı.
* 🛠️ Traker ve recognition üzerinde düzeltmeler yapılmaya çalışıldı ama yinede değeri yükseltilemedi.
---

## 🔨 Geliştirme Aşamaları
1. **Veri Seti Toplama:**  
   Kaggle'dan insan yüzü veri seti toplandı.  
   [Kaggle Veri Seti](https://www.kaggle.com/datasets/ashwingupta3012/human-faces/data)

2. **Veri Seti Oluşturma:**  
   Roboflow platformunda 200'e yakın yeni veri eklenerek eğitim seti oluşturuldu.  
   [Roboflow](https://roboflow.com/)

3. **Model Eğitimi:**  
   Google Colab üzerinde **YOLOv8n.pt** modeli kullanılarak eğitim yapıldı.  
   [Colab Eğitim Not Defteri](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb#scrollTo=D2YkphuiaE7_)

4. **Model Değerlendirme:**  
   En iyi sonuç veren modeller seçildi, yetersiz kalanlar için ek eğitimler yapıldı.

5. **Gerçek Zamanlı Yüz Tanıma:**  
   Önceden toplanan personel yüz resimleri ve anlık kamerada tespit edilen yüzler **FaceNet** ile embedding formatına dönüştürüldü ve **KNN** algoritması ile tanıma yapılarak id atandı.

6. **Kişi Takibi:**  
   **DeepSORT** algoritması kullanılarak kişilerin takibi yapıldı, ID atamaları gerçekleştirildi ve sayaçlar güncellendi.
---
## 🎉 Projenin Sonuçları

📌 YOLO modeli ile yüz algılama başarıyla gerçekleştirilmiştir.  
📌 FaceNet ve KNN algoritmaları ile yüz tanıma, DeepSORT ile kişi takibi sağlanmıştır.

## 📈 İyileştirme Önerileri

- **KNN** algoritması büyük veri setlerinde yavaşlama yaratabileceğinden, performansı artırmak için **SVM** algoritması tercih edilebilirdi.
- Yüz tanıma ve takip için yazılan kodlar daha uygun bir şekilde düzenlenebilirdi.

---
## 📸 Ekran Görüntüleri
### Model Eğitim Sonuçları
<img src="https://github.com/user-attachments/assets/ca488491-7517-4e4e-a2e1-dda51e5f8da0" alt="F1_curve" width="250" />
<img src="https://github.com/user-attachments/assets/e7950335-6de8-4f14-86d3-b59f5de7ca21" alt="PR_curve" width="250" />
<img src="https://github.com/user-attachments/assets/e96b3fac-6748-4e21-a1b0-1fbce1ef74bf" alt="P_curve" width="250" />
<img src="https://github.com/user-attachments/assets/b5a1e5a1-25ae-4a61-a0d9-5ce5b1a69502" alt="R_curve" width="250" />
<img src="https://github.com/user-attachments/assets/85b30208-b6a2-4b67-bcec-e1233647e54f" alt="confusion_matrix" width="255" />
<img src="https://github.com/user-attachments/assets/b09ab62e-6686-4ff3-9326-87c673f29a86" alt="val_batch0_pred" width="200" />

---
