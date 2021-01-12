##各个文件介绍
**pen_digits.py**

利用Convolutional neural network(two convolutional layers)进行手写数字的识别，数据集是MNIST，已下载存放在data目录下

包含CNN模型训练、测试

**FGSM.py**

生成对抗样本，比较在不同ε下FGSM的攻击效果

**model.ckpt**

可以直接使用，也可以运行pen_digits.py再训练一次神经网络

运行时间5分钟左右，请耐心等待！

**结论**

Figure_1![Figure_1](https://img-blog.csdnimg.cn/20210111225145672.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDUwMjAxOA==,size_16,color_FFFFFF,t_70#pic_center)
Figure_2![Figure_2](https://img-blog.csdnimg.cn/20210111225145876.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDUwMjAxOA==,size_16,color_FFFFFF,t_70#pic_center)
Figure_3![Figure_3](https://img-blog.csdnimg.cn/20210111225145713.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDUwMjAxOA==,size_16,color_FFFFFF,t_70#pic_center)
