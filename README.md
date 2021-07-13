# CvT-ASSD
#### including extra CvT, CvT-SSD, VGG-ASSD models

### original-code-website: 
*https://github.com/albert-jin/CvT-SSD*

### new-code-website: 
*https://github.com/albert-jin/CvT-ASSD*

### 为了符合开源号召,本项目于2021-7-12 正式开源...

### project architecture:
<img src="https://github.com/albert-jin/CvT-ASSD_private/raw/main/introduce/%E9%A1%B9%E7%9B%AE%E7%BB%93%E6%9E%84.PNG" alt="显示失败 (CvT-ASSD文件示例)" width="250px">

### Mentions

1. You may probably need to install an anaconda environment which contains all packages followed.
    - pytorch                   1.9.0           py3.7_cuda10.2_cudnn7_0    pytorch
    - cudatoolkit               10.2.89              h74a9793_1
    - opencv-python             4.5.2.54                 pypi_0    pypi
    - visdom                    0.1.8.9                  pypi_0    pypi
    - yacs                      0.1.8                    pypi_0    pypi
    - jupyter                   1.0.0                    pypi_0    pypi
    

2. For training, an NVIDIA GPU is strongly recommended for speed. we use two NVIDIA GTX-1080TI,
   but we recommend GPUs like Tesla-V100 /RTX-3090 for more memory


3. Before you run the codes for self-study or reappearance the performance in this paper **"CvT-ASSD"**,
please add the **CvT_SSD/model/** directory into sources Root caused by the reference of many codes 
inside of model directory 
   
4. you should download the pytorch parameters file postfix by ".pth" and move into **models/CvT/weights** like **项目结构.PNG**
   
5. 图像物体检测benchmark(参照论文native-SSD)一般是将VOC2007—TEST的数据作为模型的测试集,训练集可有以下搭配:
   - 1. 07:VOC2007 trainval 训练集验证集
   - 2. 02+12 VOC2007 trainval + VOC2007 trainval 训练集验证集
   - 3. 07+12+COCO 在 COCO trainval35k上预训练,然后在07+12上微调
   
6. 评价指标maP使用mxnet提供的VOC07MApMetric,将recall分成10等分,继而对所有precision取平均,在对类别去平均,具体参见 https://blog.csdn.net/u014203453/article/details/77598997
