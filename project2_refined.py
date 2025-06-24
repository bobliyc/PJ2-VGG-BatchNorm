# project2_refined.py

import os
import argparse
import copy
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ─── 模型定义（VGG_A 简化版） ────────────────────────────────────────────────────
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


# ─── MixUp 辅助函数 ────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1-lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(crit, pred, y_a, y_b, lam):
    return lam * crit(pred, y_a) + (1-lam) * crit(pred, y_b)


# ─── 训练 & 评估 ───────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device,
                    mixup_alpha, epoch, writer):
    model.train()
    total_loss, total, correct = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)
        total += batch_size
        optimizer.zero_grad()

        if mixup_alpha > 0:
            x_m, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
            out = model(x_m)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            batch_acc = 0.0
        else:
            out = model(x)
            loss = criterion(out, y)
            preds = out.argmax(dim=1)
            correct += preds.eq(y).sum().item()
            batch_acc = correct / total

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        if mixup_alpha > 0:
            pbar.set_postfix(loss=total_loss/total)
        else:
            pbar.set_postfix(loss=total_loss/total, acc=batch_acc)

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc  = correct / total if mixup_alpha == 0 else 0.0

    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    if mixup_alpha == 0:
        writer.add_scalar('Train/Acc', epoch_acc, epoch)
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device, epoch, writer):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            batch_size = x.size(0)
            total += batch_size
            total_loss += loss.item() * batch_size
            preds = out.argmax(dim=1)
            correct += preds.eq(y).sum().item()

    loss = total_loss / len(loader.dataset)
    acc  = correct / total
    writer.add_scalar('Val/Loss', loss, epoch)
    writer.add_scalar('Val/Acc', acc, epoch)
    return loss, acc


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # 数据增强 & 加载
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4,0.4,0.4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2023,0.1994,0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2023,0.1994,0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True,
        transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size,
        shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True,
        transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size,
        shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    # 两个实验：WithBN / WithoutBN
    exps = {
        'WithoutBN': VGG_A(batch_norm=False, num_classes=args.num_classes),
        'WithBN':    VGG_A(batch_norm=True,  num_classes=args.num_classes)
    }

    for name, net in exps.items():
        print(f"\n>>> Start experiment: {name}")
        net.to(device)
        optimizer = optim.Adam(
            net.parameters(), lr=args.lr,
            weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=1e-5)

        best_acc = 0.0
        for epoch in range(1, args.num_epochs+1):
            train_loss, train_acc = train_one_epoch(
                net, trainloader, optimizer, criterion,
                device, args.mixup_alpha, epoch, writer)
            val_loss, val_acc = evaluate(
                net, testloader, criterion,
                device, epoch, writer)
            scheduler.step()

            print(f"Epoch {epoch:02d} | "
                  f"TrLoss={train_loss:.4f} TrAcc={train_acc*100:5.2f}% | "
                  f"ValLoss={val_loss:.4f} ValAcc={val_acc*100:5.2f}%")

            # 保存 checkpoint
            ckpt = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc
            }
            torch.save(ckpt,
                os.path.join(args.checkpoint_dir,
                             f"{name}_epoch{epoch:02d}.ckpt"))
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(ckpt,
                    os.path.join(args.checkpoint_dir,
                                 f"{name}_best.ckpt"))

    print("\n🏁 All experiments finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',       default='./data')
    parser.add_argument('--result_dir',     default='./results')
    parser.add_argument('--checkpoint_dir', default='./checkpoints')
    parser.add_argument('--log_dir',
        default=f'./logs/{datetime.now():%Y%m%d-%H%M%S}')
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument('--num_epochs',   type=int,   default=40)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_classes',  type=int,   default=10)
    parser.add_argument('--mixup_alpha',  type=float, default=0.2)
    args = parser.parse_args()
    main(args)
