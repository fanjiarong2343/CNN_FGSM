# encoding: utf-8
'''
Fast Gradient Sign Attack（FGSM）生成对抗样本
在现实生活中,往往改动一小部分数据将会对model的结果产生巨大的影响,那么我们需要将这种微小的变化考虑进model中,增强model的泛化能力
我们可以自主生成对抗样本,让model基于对抗样本和训练样本学习
代码解释参考：https://blog.csdn.net/hg_zhh/article/details/100155785
'''

"""
白盒与黑盒
白盒的意思为：在已经获取机器学习模型内部的所有信息和参数上进行攻击
黑盒的意思为：在神经网络结构为黑箱时,仅通过模型的输入和输出,逆推生成对抗样本
误分类和目标误分类
误分类的意思是：不关心输出的分类是否存在,只要不与原分类相同即可
目标误分类的意思是：规定检测出来的类别需要与给定的一致
"""

"""
Fast Gradient Sign Attack（FGSM）
FGSM是一种简单的对抗样本生成算法,该算法的思想直观来看就是在输入的基础上沿损失函数的梯度方向加入了一定的噪声,使目标模型产生了误判
公式如下：
perturbed_image=image+epsilon∗sign(data_grad)=x+ϵ∗sign(∇xJ(θ,x,y))
"""

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),  # https://blog.csdn.net/bigFatCat_Tom/article/details/91619977 卷积层之后添加BatchNorm2d进行数据的归一化处理,这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)  # https://blog.csdn.net/qq_28418387/article/details/95918829


def fgsm_attack(image, epsilon, data_grad):
    """
    获取扰动图片
    :param image: 原始图片
    :param epsilon: 扰动量
    :param data_grad: 损失梯度
    :return:
    """
    sign_data_grad = data_grad.sign()  # 获取梯度的符号
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # 将数值裁剪到0-1的范围内
    return perturbed_image


def test(model, device, test_loader, epsilon):
    correct = 0
    adv_examples = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True  # Set requires_grad attribute of tensor. Important for Attack
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)  # https://blog.csdn.net/lyndon0_0/article/details/104457564
        model.zero_grad()  # zero_grad()函数用于每次计算完一个batch样本后梯度清零（pytorch中的梯度反馈在节点上是累加的）
        loss.backward()

        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]

        # 对于Epsilon == 0,即原图绘制5个正确的;Epsilon != 0,即扰动图像，绘制5个错误的.
        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()  # detach阻断反向传播,返回值仍为tensor;cpu()将变量放在cpu上,仍为tensor;numpy()将tensor转换为numpy。
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    return final_acc, adv_examples


if __name__ == '__main__':
    pretrained_model = "model.ckpt"
    use_cuda = True
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=50, shuffle=True
    )
    device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load(pretrained_model))
    model.eval()

    epsilons = [0, .05, .1, .15, .2, .25, .3]
    accuracies = []
    examples = []
    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

    # Plot 标出,绘制(图表) several examples of adversarial samples at each epsilon
    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])  # https://blog.csdn.net/u011208984/article/details/90720516
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray") # imshow()接收一张图像，只是画出该图，并不会立刻显示出来
    plt.tight_layout()  # tight_layout会自动调整子图参数,使之填充整个图像区域
    plt.show()  # 进行结果显示
