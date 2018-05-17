# 最详细深度学习入门：用tensorflow实现人脸检测及识别

## 0-前言
本教程主要是对[人脸检测及识别python实现系列](http://www.cnblogs.com/neo-T/p/6426029.html) 及[碉堡了！程序员用深度学习写了个老板探测器（付源码）](https://blog.csdn.net/jxw167/article/details/54375336) 的实现。主要实现的功能是用网络摄像头自动识别在工位通道走过的人脸，如果确认是老板的话，就用一张图片覆盖到整个屏幕上。虽然原教程已经写的很好，但是我们在实现的时候仍然踩了很多坑（花了三天时间）。本着让后来者少走点弯路的原则，我们特将最详细的步骤记录如下，以期读者朋友只花最少的时间便能体验用tensorflow实现人脸检测及识别。硬件要求：一台电脑和一个普通的USB摄像头（或电脑自带摄像头），U盘（用来制作Ubuntu安装盘）。

## 1-环境搭建（Ubuntu18.04+Python3.6+Tensorflow1.8+Opencv3.3.0）
### Ubuntu18.04LTS安装
操作系统大家应该都会装。但是为了力求最详细，尽可能节省大家时间，我们将Ubuntu的安装方法也列在这里。我们是在装有Windows8.1的系统上另外安装Ubuntu18.04LTS,实现双系统，
1. 首先在Windows系统下从[Ubuntu官方网站](https://www.ubuntu.com/download)下载最新的64位(tensorflow暂时只支持64位)操作系统（本文采用Ubuntu18.04LTS版本).
2. 下载[ultraiso](https://cn.ultraiso.net/xiazai.html)制作U盘启动盘。ultraiso选择试用模式即可。用ultra打开下载好的Ubuntu镜像文件，依次点击启动光盘-写入硬盘映像-写入即可(路径选择要制作的u盘，写入模式默认)。附[ultraiso制作u盘启动盘教程图文详解](http://www.upantool.com/jiaocheng/boot/4221.html)
3. 查看系统的启动方式，是传统（Legacy BIOS）还是UEFI格式。不对应会出现grub安装时崩溃。方式：win按键+R键组合键之后输入msinfo32后回车，查看电脑BIOS模式。确认之后，在用U盘启动时，开机按esc键进行选择启动模式，会出现几个选项，有SanDisk（SanDisk是U盘名）、UEFI：SanDisk等选项，选择的方式是要保持和win8方式一致：**如果之前的win8是UEFI启动的，选择UEFI:SanDisk选项进入，如果是传统的启动模式，则选择SanDisk选项。**。
4. 进入安装流程以后，大部分选项根据提示选择即可，但__选择安装类型__一项，选择__其他选项__。之后，选择_空闲_磁盘，添加(点击+)四个linux分区：/,/home/,/boot,/swap。分区大小依次建议为：10～15G，20～100G（多多益善），200M，1～2G）。附[Windows + Ubuntu 16.04 双系统安装详细教程](https://blog.csdn.net/flyyufenfei/article/details/79187656)
5. 如果出现Failed to load ldlinux.c32，则在刻录U盘的时候，选择RAW的写入模式。

### ubuntu基本配置
文本编辑器vim及中文输入法安装
1. 安装文本编辑器vim. 打开终端（快捷键：Ctrl+Alt+T），输入: `<sudo apt-get install vim>`。Ubuntu默认编辑器是Nano，换成Vim的方式如下，在终端输入：`<update-alternatives --config editor>`然后选择vim.basic这项即可。
2. 中文输入法安装google拼音。附[Ubuntu16.04安装google输入法](https://blog.csdn.net/baobao3456810/article/details/52055938)
  * 终端输入:`<sudo apt-get install fcitx-googlepinyin>`
  * 在settings->Language Support里将keyboard input method system设置为fcitx
  * 注销系统，再登录。在settings->Text Entry里，添加输入源，搜索google-pinyin添加即可
3. 强烈推荐git及github。本教程不用git也可完全实现，所以不详述，但会使用git将对进一步的学习大有帮助。附[廖雪峰的git教程](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)

### python安装（python3.6）
ubuntu自带python的版本为2.7。但python3的普及程度及重要性越来越高, 这里我们选择安装python3.6版。
1. 安装Anaconda(一个开源的python发行版)。
  * 首先[下载](https://www.anaconda.com/download/#linux)相应版本Anaconda(这里使用的是Python3.6版64位anaconda版本)。
  * 安装。终端输入: `<bash ./Anaconda3-5.1.0-Linux-x86_64.sh>`进行安装，全程回车和Yes。
  * 配置环境变量。首先输入`<sudo gedit ~/.bashrc>` 检查末尾是否已经添加如下变量，没有则添加：`<export PATH="/home/用户名/anaconda3/bin:(dollar sign)PATH">`。
  * 重启系统：注销或重启。
  * 更换清华源，提升安装速度。在终端输入：
```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```
  * Anaconda创建名为your_env_name(建议命名为tensorflow)的python虚拟环境:`<conda create -n tensorflow python=3.6>`
  * 激活环境：`<source activate tensorflow>`
  * 关闭环境：`<source deactivate>`
  * 删除环境(需要时使用，甚重操作)：`<conda remove -n your_env_name --all>`
  * 删除环境中的某个包(需要时使用，甚重操作)：`<conda remove --name your_env_name package_name>`

### tensorflow安装
在conda的tensorflow环境下安装，使用[清华的源](https://mirrors.tuna.tsinghua.edu.cn/help/tensorflow/)。此处可选择CPU或GPU版本，本文选的是CPU版本。
  * 激活conda的tensorflow环境：`<source activate tensorflow>`
  * 安装
```
pip install --ignore-installed --upgrade   https://mirrors.tuna.tsinghua.edu.cn/tensorflow/linux/cpu/tensorflow-1.7.0-cp36-cp36m-linux_x86_64.whl
```
其中, 网址部分可以换成自己在清华源选择之后生成的版本。
  * 测试是否安装成功。在shell中输入：python，进入python交互式环境，输入：
```
import tensorflow as tf
hello = tf.constant('Hello, Tensorflow!')
sess = tf.Session()
print(sess.run(hello).decode())
```
如果出现Hello, Tensorflow!说明安装成功！

### opencv安装
opencv可以从摄像头读取视频流，识别人脸。如果其他方式安装可能会出现opencv和python版本兼容问题。采用anaconda安装便不会出现这种问题。
  * 安装PIL包：`<conda install PIL>`
  * 安装。在shell输入：`<pip install opencv-python>`
  * 测试是否安装成功。opencv分为c/c++版本和python版本，注意这里选择python版本的测试程序。将下述代码放入python交互式窗口中，回车，如果没有出现错误，说明opencv安装成功。
  
```
import cv2
import sys
from PIL import Image

def CatchUsbVideo(window_name, camera_idx):
    cv2.namedWindow(window_name)
    
    #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)        
        
    while cap.isOpened():
        ok, frame = cap.read() #读取一帧数据
        if not ok:            
            break                    
                        
        #显示图像并等待10毫秒按键输入，输入‘q’退出程序
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break        
    
    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows() 
```

### 安装keras、sklearn等
    conda install keras
    conda install scikit-learn
    
### 安装Jupyter Notebook
推荐，非必须。anaconda自带jupyter，如果不能用（出现No module named tensorflow），则在终端输入如下命令，重新安装一遍。

    source activate tensorflow #激活tensorflow环境
    conda install ipython
    conda install jupyter
    jupyter notebook #打开jupyter notebook
    
## 2-获取并显示摄像头视频

### 检查摄像头是否正常工作
  * 安装cheese：`<sudo apt-get install cheese >`
  * shell输入cheese,如果弹出摄像头窗口，说明摄像头正常。
  * 检查摄像头id：`<ls  /dev/video*>`,一般id为0。
### 利用OpenCV获取摄像头的视频流并展示
  * 激活tensorflow环境：`<source activate tensorflow>`，接下来如不做特别说明，所有shell操作都是在此环境下进行。
  * 创建文件夹，接下来如不做特别说明，所有shell操作都是在此文件夹目录下，我们暂时称为主目录。
  * 创建名为catch_usb_video.py的python文件，将下列代码copy进去，保存，退出。


    import cv2
    import sys
    from PIL import Image

    def CatchUsbVideo(window_name, camera_idx):
        cv2.namedWindow(window_name)
    
        #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
        cap = cv2.VideoCapture(camera_idx)        
        
        while cap.isOpened():
            ok, frame = cap.read() #读取一帧数据
            if not ok:            
                break                    
                            
            #显示图像并等待10毫秒按键输入，输入‘q’退出程序
            cv2.imshow(window_name, frame)
            c = cv2.waitKey(10)
            if c & 0xFF == ord('q'):
                break        
    
        #释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows() 
        
    if __name__ == '__main__':
        if len(sys.argv) != 2:
            print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        else:
            CatchUsbVideo("Capturing Video", int(sys.argv[1]))
  * 在shell窗口输入：`<python catch_usb_video.py 0>`运行上述程序。
  * 按q关闭Capturing Video窗口

## 3-识别出人脸
  * 创建名为recognise_face.py的文件夹，存入下述代码：

    #-*- coding: utf-8 -*-
    
    import cv2
    import sys
    from PIL import Image
    
    def CatchUsbVideo(window_name, camera_idx):
        cv2.namedWindow(window_name)
        
        #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
        cap = cv2.VideoCapture(camera_idx)                
        
        #告诉OpenCV使用人脸识别分类器
        classfier = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml")
        
        #识别出人脸后要画的边框的颜色，RGB格式
        color = (0, 255, 0)
            
        while cap.isOpened():
            ok, frame = cap.read() #读取一帧数据
            if not ok:            
                break  
    
            #将当前帧转换成灰度图像
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                 
            
            #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
            faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
            if len(faceRects) > 0:            #大于0则检测到人脸                                   
                for faceRect in faceRects:  #单独框出每一张人脸
                    x, y, w, h = faceRect        
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                            
            #显示图像
            cv2.imshow(window_name, frame)        
            c = cv2.waitKey(10)
            if c & 0xFF == ord('q'):
                break        
        
        #释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows() 
        
    if __name__ == '__main__':
        if len(sys.argv) != 2:
            print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        else:
            CatchUsbVideo("Recognising Face", int(sys.argv[1]))
  * 在shell输入命令`<python recognise_face.py 0>`运行上述代码，跳出的视频窗口中，人脸区域会被方框圈起来。
  
## 4-为模型训练准备人脸数据
这部分采集两个人脸数据，分别存到data/me和data/other 文件夹里。
  * 在主目录下，创建data文件夹，data文件夹里创建me和other文件夹爱。
  * 在主目录下，创建face_data.py文件，存入下述代码
  
    #-*- coding: utf-8 -*-

    import cv2
    import sys

    from PIL import Image

    def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
        cv2.namedWindow(window_name)

        #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
        cap = cv2.VideoCapture(camera_idx)                

        #告诉OpenCV使用人脸识别分类器
        classfier = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml")

        #识别出人脸后要画的边框的颜色，RGB格式
        color = (0, 255, 0)

        num = 0    
        while cap.isOpened():
            ok, frame = cap.read() #读取一帧数据
            if not ok:            
                break                

            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #将当前桢图像转换成灰度图像            

            #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
            faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
            if len(faceRects) > 0:          #大于0则检测到人脸                                   
                for faceRect in faceRects:  #单独框出每一张人脸
                    x, y, w, h = faceRect                        

                    #将当前帧保存为图片
                    img_name = '%s/%d.jpg'%(path_name, num)                
                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    cv2.imwrite(img_name, image)                                

                    num += 1                
                    if num > (catch_pic_num):   #如果超过指定最大保存数量退出循环
                        break

                    #画出矩形框
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                    #显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)                

            #超过指定最大保存数量结束程序
            if num > (catch_pic_num): break                

            #显示图像
            cv2.imshow(window_name, frame)        
            c = cv2.waitKey(10)
            if c & 0xFF == ord('q'):
                break        

        #释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows() 

    if __name__ == '__main__':
        if len(sys.argv) != 4:
            print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
        else:
            CatchPICFromVideo("Collecting face data", int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
  * 在shell先后运行：`<python face_data.py 0 1000 data/me>` 和`<python face_data.py 0 1000 data/other>`分别采集两个人各1000个人脸数据存入相应文件夹。
  * 手动剔除data文件夹中不是图片文件的数据。

## 5-利用keras库训练人脸识别模型
利用keras库来建立模型和评估模型
  * 主目录下建立model文件夹:`<mkdir model>`
  * 主目录下建立face_train_keras.py文件，存入下述代码：
  
    #-*- coding: utf-8 -*-
    import random

    import numpy as np
    from sklearn.cross_validation import train_test_split
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.optimizers import SGD
    from keras.utils import np_utils
    from keras.models import load_model
    from keras import backend as K

    from load_face_dataset import load_dataset, resize_image, IMAGE_SIZE

    class Dataset:
        def __init__(self, path_name):
            #训练集
            self.train_images = None
            self.train_labels = None

            #验证集
            self.valid_images = None
            self.valid_labels = None

            #测试集
            self.test_images  = None            
            self.test_labels  = None

            #数据集加载路径
            self.path_name    = path_name

            #当前库采用的维度顺序
            self.input_shape = None

        #加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
        def load(self, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE, 
                 img_channels = 3, nb_classes = 2):
            #加载数据集到内存
            images, labels = load_dataset(self.path_name)        

            train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size = 0.3, random_state = random.randint(0, 100))        
            _, test_images, _, test_labels = train_test_split(images, labels, test_size = 0.5, random_state = random.randint(0, 100))                

            #当前的维度顺序如果为'th'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
            #这部分代码就是根据keras库要求的维度顺序重组训练数据集
            if K.image_dim_ordering() == 'th':
                train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
                valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
                test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
                self.input_shape = (img_channels, img_rows, img_cols)            
            else:
                train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
                valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
                test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
                self.input_shape = (img_rows, img_cols, img_channels)            

                #输出训练集、验证集、测试集的数量
                print(train_images.shape[0], 'train samples')
                print(valid_images.shape[0], 'valid samples')
                print(test_images.shape[0], 'test samples')

                #我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
                #类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
                train_labels = np_utils.to_categorical(train_labels, nb_classes)                        
                valid_labels = np_utils.to_categorical(valid_labels, nb_classes)            
                test_labels = np_utils.to_categorical(test_labels, nb_classes)                        

                #像素数据浮点化以便归一化
                train_images = train_images.astype('float32')            
                valid_images = valid_images.astype('float32')
                test_images = test_images.astype('float32')

                #将其归一化,图像的各像素值归一化到0~1区间
                train_images /= 255
                valid_images /= 255
                test_images /= 255            

                self.train_images = train_images
                self.valid_images = valid_images
                self.test_images  = test_images
                self.train_labels = train_labels
                self.valid_labels = valid_labels
                self.test_labels  = test_labels
                #CNN网络模型类
    class Model:
        def __init__(self):
            self.model = None

        #建立模型
        def build_model(self, dataset, nb_classes = 2):
            #构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
            self.model = Sequential()

            #以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
            self.model.add(Convolution2D(32, 3, 3, border_mode='same',
                                         input_shape = dataset.input_shape))    #1 2维卷积层
            self.model.add(Activation('relu'))                                  #2 激活函数层

            self.model.add(Convolution2D(32, 3, 3))                             #3 2维卷积层
            self.model.add(Activation('relu'))                                  #4 激活函数层

            self.model.add(MaxPooling2D(pool_size=(2, 2)))                      #5 池化层
            self.model.add(Dropout(0.25))                                       #6 Dropout层

            self.model.add(Convolution2D(64, 3, 3, border_mode='same'))         #7  2维卷积层
            self.model.add(Activation('relu'))                                  #8  激活函数层

            self.model.add(Convolution2D(64, 3, 3))                             #9  2维卷积层
            self.model.add(Activation('relu'))                                  #10 激活函数层

            self.model.add(MaxPooling2D(pool_size=(2, 2)))                      #11 池化层
            self.model.add(Dropout(0.25))                                       #12 Dropout层

            self.model.add(Flatten())                                           #13 Flatten层
            self.model.add(Dense(512))                                          #14 Dense层,又被称作全连接层
            self.model.add(Activation('relu'))                                  #15 激活函数层
            self.model.add(Dropout(0.5))                                        #16 Dropout层
            self.model.add(Dense(nb_classes))                                   #17 Dense层
            self.model.add(Activation('softmax'))                               #18 分类层，输出最终结果

            #输出模型概况
            self.model.summary()

    #训练模型
        def train(self, dataset, batch_size = 20, nb_epoch = 10, data_augmentation = True):
            sgd = SGD(lr = 0.01, decay = 1e-6,
                      momentum = 0.9, nesterov = True) #采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=sgd,
                               metrics=['accuracy'])   #完成实际的模型配置工作

            #不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
            #训练数据，有意识的提升训练数据规模，增加模型训练量
            if not data_augmentation:
                self.model.fit(dataset.train_images,
                               dataset.train_labels,
                               batch_size = batch_size,
                               nb_epoch = nb_epoch,
                               validation_data = (dataset.valid_images, dataset.valid_labels),
                               shuffle = True)
            #使用实时数据提升
            else:
                #定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
                #次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
                datagen = ImageDataGenerator(
                    featurewise_center = False,             #是否使输入数据去中心化（均值为0），
                    samplewise_center  = False,             #是否使输入数据的每个样本均值为0
                    featurewise_std_normalization = False,  #是否数据标准化（输入数据除以数据集的标准差）
                    samplewise_std_normalization  = False,  #是否将每个样本数据除以自身的标准差
                    zca_whitening = False,                  #是否对输入数据施以ZCA白化
                    rotation_range = 20,                    #数据提升时图片随机转动的角度(范围为0～180)
                    width_shift_range  = 0.2,               #数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                    height_shift_range = 0.2,               #同上，只不过这里是垂直
                    horizontal_flip = True,                 #是否进行随机水平翻转
                    vertical_flip = False)                  #是否进行随机垂直翻转

                #计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
                datagen.fit(dataset.train_images)

                #利用生成器开始训练模型
                self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                       batch_size = batch_size),
                                         samples_per_epoch = dataset.train_images.shape[0],
                                         nb_epoch = nb_epoch,
                                         validation_data = (dataset.valid_images, dataset.valid_labels))

        MODEL_PATH = './me.face.model.h5'
        def save_model(self, file_path = MODEL_PATH):
            self.model.save(file_path)

        def load_model(self, file_path = MODEL_PATH):
            self.model = load_model(file_path)

        def evaluate(self, dataset):
            score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose = 1)
            print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    #识别人脸
        def face_predict(self, image):    
            #依然是根据后端系统确定维度顺序
            if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
                image = resize_image(image)                             #尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
                image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))   #与模型训练不同，这次只是针对1张图片进行预测    
            elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
                image = resize_image(image)
                image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))                    

            #浮点并归一化
            image = image.astype('float32')
            image /= 255

            #给出输入属于各个类别的概率，我们是二值类别，则该函数会给出输入图像属于0和1的概率各为多少
            result = self.model.predict_proba(image)
            print('result:', result)

            #给出类别预测：0或者1
            result = self.model.predict_classes(image)        

            #返回类别预测结果
            return result[0]

    if __name__ == '__main__':
        dataset = Dataset('./data/')
        dataset.load()
        
        #训练模型
        model = Model()
        model.build_model(dataset)
        model.train(dataset)
        model.save_model(file_path = './model/me.face.model.h5')
        
  * 在shell运行：`<python face_train_use_keras.py>`保存模型，可以看到训练误差（loss)、训练准确率（acc)、验证误差（val_loass）、验证准确率（val_acc）。
  * 修改face_train_use_keras.py，将训练模型注释掉，添加评估模型，如下：
        
        
        '''
        #训练模型,注释掉
        model = Model()
        model.build_model(dataset)
        model.train(dataset)
        model.save_model(file_path = './model/me.face.model.h5')
        '''
        #评估模型
        model = Model()
        model.load_model(file_path = './model/me.face.model.h5')
        model.evaluate(dataset)

  * 重新运行`<python face_train_use_keras.py>` 可以看到准确率。
  

## 6-从实时视频流识别出自己

  * 主目录下创建face_predict_use_keras.py文件，存入下述代码：

    #-*- coding: utf-8 -*-

    import cv2
    import sys
    import gc
    from face_train_use_keras import Model

    if __name__ == '__main__':
        if len(sys.argv) != 2:
            print("Usage:%s camera_id\r\n" % (sys.argv[0]))
            sys.exit(0)

        #加载模型
        model = Model()
        model.load_model(file_path = './model/me.face.model.h5')    

        #框住人脸的矩形边框颜色       
        color = (0, 255, 0)

        #捕获指定摄像头的实时视频流
        cap = cv2.VideoCapture(int(sys.argv[1]))

        #人脸识别分类器本地存储路径
        cascade_path = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"    

        #循环检测识别人脸
        while True:
            _, frame = cap.read()   #读取一帧视频

            #图像灰化，降低计算复杂度
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #使用人脸识别分类器，读入分类器
            cascade = cv2.CascadeClassifier(cascade_path)                

            #利用分类器识别出哪个区域为人脸
            faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
            if len(faceRects) > 0:                 
                for faceRect in faceRects: 
                    x, y, w, h = faceRect

                    #截取脸部图像提交给模型识别这是谁
                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    faceID = model.face_predict(image)   

                    #如果是“我”
                    if faceID == 0:                                                        
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)

                        #文字提示是谁
                        cv2.putText(frame,'Me', 
                                    (x + 30, y + 30),                      #坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                    1,                                     #字号
                                    (255,0,255),                           #颜色
                                    2)                                     #字的线宽
                    else:
                        pass

            cv2.imshow("Recognise myself", frame)

            #等待10毫秒看是否有按键输入
            k = cv2.waitKey(10)
            #如果输入q则退出循环
            if k & 0xFF == ord('q'):
                break

        #释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows()

  * 在shell输入：`<python face_predict_use_keras.py 0>`可以看到视频窗口里自己的头像会被文字“Me”标出。
  
  
## 7-识别出自己（老板）后更改电脑桌面
调用PyQt库，使摄像头识别出自己后把电脑屏幕换成一副全屏图片。
  * 在主目录下放一张图片，本文放的是equation.jpg
  * 创建image_show.py的文件，存入下述代码：
  
    import sys

    from PyQt5 import QtGui,QtWidgets


    def show_image(image_path='equation.jpeg'):
        app = QtWidgets.QApplication(sys.argv)
        pixmap = QtGui.QPixmap(image_path)
        screen = QtWidgets.QLabel()
        screen.setPixmap(pixmap)
        screen.showFullScreen()
        sys.exit(app.exec_())


    if __name__ == '__main__':
    show_image()

  * 修改face_predict_use_keras.py文件，在开头部分添加`<from image_show import show_image>`;在“如果是我”部分，添加`<image_show()>`如下：
  
      #如果是“我”
                    if faceID == 0:                                                        
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)

                        #文字提示是谁
                        cv2.putText(frame,'Me', 
                                    (x + 30, y + 30),                      #坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                    1,                                     #字号
                                    (255,0,255),                           #颜色
                                    2)                                     #字的线宽
                        show_image()

                    else:
    pass
  * shell输入`<python face_predict_use_keras.py 0>`,将脸慢慢对准摄像头，可以看到屏幕变成预设的全屏图片。
  * 把自己头像换成老板头像进行训练、识别，就是老板检测器了。
  * 键盘win键，可以关闭全屏图片。
  * 恭喜完成！

## 8-结语
  本文实现了一个简单的深度学习项目：用tensorflow实现人脸检测及识别（即老板检测器）。本文主要专注于实际的操作过程及步骤，详细的列出了从安装Ubuntu到识别出人脸切换屏幕的整个操作细节，相信对于初次接触深度学习实践的同学会有很大帮助。本文对于代码中的原理没有过多解释。希望更进一步了解的朋友可以从开头提供的博客及寻找相关资料进行学习。感谢前辈们提供的经验，也请大家批评指正。谢谢！
  

