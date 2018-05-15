#  环境搭建

## Ubuntu双系统
在已有Win8.1的基础上安装Ubuntu18.4。
1. 首先查看系统的启动方式，是传统（Legacy BIOS）还是UEFI格式。不对应会出现grub安装时崩溃。方式：win按键+R键组合键之后输入msinfo32后回车，查看电脑BIOS模式。确认之后，在用U盘启动时，开机按esc键进行选择启动模式，会出现几个选项，有SanDisk（SanDisk是U盘名）、UEFI：SanDisk等选项，选择的方式是要保持和win8方式一致：**如果之前的win8是UEFI启动的，选择UEFI:SanDisk选项进入，如果是传统的启动模式，则选择SanDisk选项。**
2. 如果出现Failed to load ldlinux.c32，则在刻录U盘的时候，选择RAW的写入模式。

## Ubuntu配置
1. 安装vim： sudo apt-get install vim

2. 安装Anaconda(python发行版)：
首先下载相应版本Anaconda(这里使用的是Python3.6版64位anaconda版本)。之后打开shell输入如下命令: ```bash ./Anaconda3-5.1.0-Linux-x86_64.sh```
之后全称回车和Yes.
配置环境变量：查看命令：
```sudo gedit ~/.bashrc```
检查末尾是否已经添加如下变量，没有则添加：
```
export PATH="/home/用户名/anaconda3/bin:$PATH"
```
**完成之后重启系统**
打开Terminal，更换清华源
```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```
Anaconda创建名为your_env_name的python虚拟环境:
> conda create -n your_env_name python=X.X（版本号）

激活环境：
> source activate your_env_name

关闭环境：
> source deactivate

删除环境：
> conda remove -n your_env_name --all

删除环境中的某个包：
> conda remove --name your_env_name package_name

3. 安装TensorFlow：参见TensorFlow官方网站，源使用清华的镜像，速度会快一点。
```
pip install --ignore-installed --upgrade tfBinaryURL
```
其中, tfBinaryURL是tensorflow python软件包的网址（建议使用清华镜像）。安装成功以后在python环境下使用以下代码进行测试。
```
import tensorflow as tf
hello = tf.constant('Hello, Tensorflow!')
sess = tf.Session()
print(sess.run(hello))
```
如果出现Hello, Tensorflow!说明成功。

4. 安装jupyter notebook:anaconda自带jupyter，如果不能用（出现No module named tensorflow），则在终端输入如下命令，重新安装一遍。

```
source activate tensorflow #激活tensorflow环境
conda install ipython
conda install jupyter
jupyter notebook #重新打开jupyter notebook
```

5. 安装opencv: 在conda环境下，输入：pip install opencv-python,在python环境下输入import cv2查看是否成功。(如果中间缺少什么模块，按提示安装)

6. 安装keras
conda install keras

7. 安装sklearn
conda install scikit-learn


## 注意事项

1. Ubuntu默认编辑器是Nano，换成Vim的方式如下：

在终端输入：

update-alternatives --config editor

然后选择vim.basic这项即可

