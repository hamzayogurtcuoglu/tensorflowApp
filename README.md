1)linear regression , nonlinear regression,logistic regression , activation library
2)bildiğin convolution neural network object detection
3)recurrent neural network -nlp
4)unsupervised learning - movie recommendation system
5)autoencoder

1 - 
tensorflow defines computations as Graph bu graph nodelarıda bir operation ve graphda bir operation serisi
tensorflow computation farklı cihazlarda hesaplama olanağıda sunmakta clusterda mesela

2--
if you want to feed data to TensorFlow from outside a model, you will need to use placeholder.

3---
Linear Regression
Simple Linear model is :
Y = aX + b
Y is dependent variable X is independent variable a and b is paramater
a is slope or gradient b is intercept
Matplotlib, Python programlama dili ve sayısal matematik uzantısı olan NumPy için bir çizim kütüphanesidir.

plt.rcParams['figure.figsize'] = (10,6) # bu plotun size ayarlamak için en başta yapabilirsin her zaman
X = np.arange(0.0,5.0,0.1) ---> 0.0 ile 5.0 arasındaki ondalık sayıları list eder.

4----
Logistic Regression
L.R. provide the formala that predicts the likelihood that a given input belongs to a certain class
model makes these predictions by analayzing the data features which are set of independent variables that describe the data

takes in series of inputs and passes them through a set of three functions

1) Weight matrix multiplication set of values known as the weights. the weight values represent the relative amount of emphasis
to put on each feature of the input on each feature of input the model improves its classification ability by gradually updating these weights
throughtout thre training process

2)Bias Addition is updated according to model.

3)Fitting sigmoid probability Curve will map the sum onto a curve that represents the probability of the input belonging to certain class

5-----
Activation functions(En Çok Kullanılan RELU)-ReLU(Rectified Linear Unit)-Doğrultulmuş Doğrusal Birim

i = tf.constant([1.0,2.0,3.0],shape=[1,3]) ->  shape : Ortaya çıkan tensörün isteğe bağlı boyutları.
act = func(tf.matmul(i,w)+b) #Multiplies matrix a by matrix b, producing a * b.

Sinir ağının evet ya da hayır benzeri bir çıktı vermesini sağlar. Elde edilen değerin fonksiyona bağlı olarak 0 ya da 1 yahut -1 ya da 1 arasına sıkıştırır.

Aktivasyon fonksiyonları ikiye ayrılır:

    Doğrusal Aktivasyon Fonksiyonları
    Doğrusal Olmayan Aktivasyon fonksiyonları


Eğer aktivasyon fonksiyonu uygulanmazsa çıkış sinyali basit bir doğrusal fonksiyon olur. Doğrusal fonksiyonlar yalnızca tek dereceli polinomlardır. Aktivasyon fonksiyonu kullanılmayan bir sinir ağı sınırlı öğrenme gücüne sahip bir doğrusal bağlanım (linear regression) gibi davranacaktır. Ama biz sinir ağımızın doğrusal olmayan durumları da öğrenmesini istiyoruz. Çünkü sinir ağımıza öğrenmesi için görüntü, video, yazı ve ses gibi karmaşık gerçek dünya bilgileri vereceğiz. Çok katmanlı derin sinir ağları bu sayede verilerden anlamlı özellikleri öğrenebilir.

Ağırlıklar ile ilgili hata değerlerini hesaplamak için yapay sinir ağında hatanın "geriye yayılımı"(backpropagation) algoritması uygulanmaktadır. Optimizasyon stratejisini belirlemek ve hata oranını minimize etmek gerekmektedir. Uygun optimizasyon algoritmasını seçmek de ayrı bir konudur.

Softmax Fonksiyonu
Sigmoid fonksiyonuna çok benzer bir yapıya sahiptir. Aynı Sigmoid’te olduğu gibi sınıflayıcı olarak kullanıldığında oldukça iyi bir performans sergiler. En önemli farkı sigmoid fonksiyonu gibi ikiden fazla sınıflamak gereken durumlarda özellikle derin öğrenme modellerinin çıkış katmanında tercih edilmektedir. Girdinin belirli sınıfa ait olma olasılığını 0–1 aralığında değerler üreterek belirlenmesini sağlamaktadır. Yani olasılıksal bir yorumlama gerçekleştirir.

6------
CNN
7-------

Convolution and Feature Learning
Bildiğin convolution uyguluyorsun değişik filtre ve boyutlarda ve convolved feature matrix elde ediyorsun.

8--------
Convolution with Python and Tensorflow

9---------
MNIST Database

12-multilayer perceptron
Çok katmanlı algılayıcı

13-Evrişimsel Sinir Ağlarının Yapısı

    Convolutional Layer — Özellikleri saptamak için kullanılır
    Non-Linearity Layer — Sisteme doğrusal olmayanlığın (non-linearity) tanıtılması
    Pooling (Downsampling) Layer — Ağırlık sayısını azaltır ve uygunluğu kontrol eder
    Flattening Layer — Klasik Sinir Ağı için verileri hazırlar(Tek boyutluya çevirir.)
    Fully-Connected Layer — Sınıflamada kullanılan Standart Sinir Ağı

Giriş Resmi -> [[Conv -> ReLU]*N -> Pool?]*M -> Flattening -> [FC -> ReLU]*K -> FC

Dropout = Kısaca özetlemek gerekirse eğitim sırasında aşırı öğrenmeyi(overfitting) engellemek için bazı nöronları unutmak için kullanılanılır diyebiliriz. Eğer ağınız çok büyükse, çok uzun süre eğitim yapıyorsanız veya veri sayınız çok az ise aşırı öğrenme riski taşıdığınızı unutmamanız gerekir.

Batch Normalization

Batch normalization evrişimsel sinir ağını daha düzenli hale getirmek için kullanılan başka bir yöntemdir.
Düzenleyici bir etkinin yanı sıra, batch norm aynı zamanda evrişimsel sinir ağının eğitim sırasında yok olma gradyanına bir direnç de verir. Bu da eğitim süresini azaltabilir ve modelin daha iyi performans göstermesini sağlayabilir.

Sequential Problem
Sequential datalar bildiğin random samplingden farkı olmuyor model için.

The Recurrent Neural Network Model
Bu greet bir tool modeling sequential datalar için gerekli dataları tutmak istiyorsak bunu kullanıyoruz. 
Long Short Term Memory Model ile daha iyi bir sonuç alınabiliyor

Recursive Neural Tensor Networks
NLP problems that they are able to solve. In order to classify sentences into diffrent sentiment classes, we'll need a dataset to use for training.
sentiment:duygusallık




