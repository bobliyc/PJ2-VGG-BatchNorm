# PJ2: VGG on CIFAR-10 with and without BatchNorm

本项目为神经网络课程大作业，旨在探究在 CIFAR-10 数据集上，VGG-A 网络在引入 Batch Normalization 前后的训练效果与模型表现差异，并进行 Loss landscape 的可视化分析。

##  项目结构

├── data/ # 数据集（建议手动下载放入此目录）
├── logs/ # 日志输出目录
├── models/ # 存储模型结构定义
├── utils/ # 辅助工具函数
├── project2_refined.py # 主训练脚本
├── VGG_Loss_Landscape.py # Loss Landscape 可视化主程序
├── draw_loss_landscape.py # 绘图脚本
├── .gitignore

markdown
复制
编辑

##  实验说明

我们基于经典的 VGG-A 架构，分别实现了：

- **VGG-A without BatchNorm**
- **VGG-A with BatchNorm**

并在 CIFAR-10 上训练，比较其：
- 收敛速度
- 准确率
- Loss 曲面平滑度（通过 loss landscape 可视化）

##  环境配置

```bash
 创建环境（推荐使用 Python 3.8+）
pip install -r requirements.txt  # 请自行整理需要的依赖，如 torch, torchvision, matplotlib 等
 训练命令
bash
复制
编辑
# 训练无 BatchNorm 版本
python project2_refined.py --use_bn False

# 训练含 BatchNorm 版本
python project2_refined.py --use_bn True
训练完成后，模型保存在 logs/ 目录下，自动按时间戳命名。

 Loss Landscape 可视化
bash
复制
编辑
# 生成 Loss landscape 网格
python VGG_Loss_Landscape.py --ckpt_path logs/xxx/model_best.pth

绘图
python draw_loss_landscape.py --input_dir logs/xxx/
 模型与可视化文件下载
Google Drive 链接（包含模型权重与 loss surface）：
 PJ2 Results

 作者信息
姓名：李奕呈

学号：23300740001

GitHub: bobliyc

