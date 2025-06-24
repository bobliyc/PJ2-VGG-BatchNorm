# draw_loss_landscape.py

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. 网络定义（与训练脚本一致）
cfg_A = [64, 'M', 128, 'M', 256, 256, 'M']
class VGG_A(nn.Module):
    def __init__(self, batch_norm=False, num_classes=10):
        super().__init__()
        layers, in_ch = [], 3
        for v in cfg_A:
            if v == 'M':
                layers.append(nn.MaxPool2d(2,2))
            else:
                conv = nn.Conv2d(in_ch, v, 3, padding=1)
                if batch_norm:
                    layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv, nn.ReLU(inplace=True)]
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(in_ch*4*4, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# 2. 只提取 weight 参数并扁平化
def get_weight_params(model):
    return [p for p in model.parameters() if p.ndim > 1]

def flatten(params):
    return torch.nn.utils.parameters_to_vector(params).detach()

# 3. 评估平均 loss
def eval_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
    return total_loss / len(loader.dataset)

if __name__ == '__main__':
    # 4. 加载测试集（num_workers=0）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2023,0.1994,0.2010))
    ])
    test_ds = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256,
                             shuffle=False, num_workers=0)

    # 5. 最佳权重文件名（与 checkpoints 保持一致或换成 .pth）
    ckpts = {
        'WithoutBN': 'checkpoints/WithoutBN_best.ckpt',
        'WithBN':    'checkpoints/WithBN_best.ckpt'
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models, states = {}, {}
    for name, path in ckpts.items():
        m = VGG_A(batch_norm=(name=='WithBN')).to(device)
        ck = torch.load(path, map_location=device)
        m.load_state_dict(ck['state_dict'])
        models[name] = m
        states[name] = copy.deepcopy(m.state_dict())

    # 6. 构造随机方向
    params0   = get_weight_params(models['WithoutBN'])
    base_vec0 = flatten(params0)
    direction = torch.randn_like(base_vec0)
    direction /= direction.norm()

    # 7. 计算 loss landscape
    alphas = np.linspace(-1, 1, 21, dtype=np.float32)
    criterion = nn.CrossEntropyLoss()
    losses = {name: [] for name in models}

    for a in alphas:
        for name, model in models.items():
            model.load_state_dict(states[name])
            params = get_weight_params(model)
            flat   = flatten(params)
            perturbed = flat + a * direction
            torch.nn.utils.vector_to_parameters(perturbed, params)
            l = eval_loss(model, test_loader, criterion, device)
            losses[name].append(l)

    # 8. 绘图并保存
    plt.figure()
    for name in models:
        plt.plot(alphas, losses[name], label=name)
    plt.title("1D Loss Landscape")
    plt.xlabel("Alpha")
    plt.ylabel("Test Loss")
    plt.legend()
    os.makedirs('results', exist_ok=True)
    plt.savefig("results/loss_landscape.png", dpi=150)
    print("Saved results/loss_landscape.png")
