# TruFor 训练与测试文档

本文档详细介绍TruFor模型的训练配置、训练步骤、训练结果评估以及模型预测与可视化方法。

## 1. 数据配置

### 1.1 数据集路径配置

在`TruFor_train_test/project_config.py`中设置数据集根路径：

```python
dataset_paths = {
    # 指定数据集根目录
    'IMD'      : 'dataset/data/IMD2020',
    # 'FR'       : 'path/to/FR',
    # 'CA'       : 'path/to/CA',
    # 'tampCOCO' : 'path/to/tampCOCO',
    # 'compRAISE': 'path/to/compRAISE',
}
```

### 1.2 训练数据集选择

在阶段特定的配置文件中选择使用的数据集（以IMD为例）：

```yaml
DATASET:
  TRAIN: [IMD]
  VALID: [IMD]
  NUM_CLASSES: 2
  CLASS_WEIGHTS: [0.5, 2.5]  # 类别权重，用于处理类别不平衡问题
```

### 1.3 数据增强

训练和验证过程中使用的数据增强配置位于`lib/config/aug_res_comp.yaml`，主要包含：

- 随机缩放（scale_limit=(-0.5, 0.5)）
- JPEG压缩（quality_lower=30, quality_upper=100）

### 1.4 数据加载流程

`dataset/data_core.py`中的`myDataset`类负责加载和处理数据集：

- 支持多种数据集的混合使用
- 实现了类别平衡采样策略
- 在训练模式下使用裁剪大小为512×512的图像
- 在验证模式下支持原始图像大小（最大维度可通过`VALID.MAX_SIZE`限制）

## 2. 训练步骤

TruFor模型采用两阶段训练策略：

### 2.1 阶段一：训练定位网络（trufor_ph2）

此阶段主要训练图像篡改区域定位头。

**配置文件**：`lib/config/trufor_ph2.yaml`

关键配置：
```yaml
MODEL:
  MODULES: ['NP++','backbone','loc_head']  # 只启用定位相关模块
  FIX_MODULES: ['NP++']  # 冻结Noiseprint++模块
LOSS:
  LOSSES:  # 只使用定位损失
    - ['LOC', 1.0, 'dice_entropy']  # 混合Dice损失和交叉熵损失
TRAIN:
  BATCH_SIZE_PER_GPU: 1  # 较小的batch size以适应显存
  END_EPOCH: 4  # 训练轮次
  LR: 0.005  # 学习率
VALID:
  BEST_KEY: 'avg_p-F1_smooth'  # 以平滑后的F1分数作为最佳模型指标
```

**训练命令**：
```bash
python train.py -exp trufor_ph2 -g 0
```

**训练输出**：
- **模型文件**：
  - `checkpoint.pth.tar`：每个epoch结束时保存，包含当前epoch、最佳值、最佳指标键、模型状态字典和优化器状态
  - `best.pth.tar`：当验证集上的`avg_p-F1_smooth`指标达到最佳时保存
- **日志输出**：
  - 训练和验证损失
  - IoU数组
  - 混淆矩阵
  - 各评估指标值
- **可视化输出**：
  - TensorBoard日志，记录训练损失、学习率和各评估指标

**输出目录**：所有输出默认保存在`output/trufor_ph2`目录下。

### 2.2 阶段二：训练检测网络和置信度估计器（trufor_ph3）

此阶段使用阶段一的权重作为预训练模型，训练置信度估计头和检测头。

**配置文件**：`lib/config/trufor_ph3.yaml`

关键配置：
```yaml
MODEL:
  MODULES: ['NP++','backbone','loc_head','conf_head','det_head']  # 启用所有模块
  FIX_MODULES: ['NP++','backbone','loc_head']  # 冻结前面已训练好的模块
LOSS:
  LOSSES:  # 使用置信度损失和检测损失
    - ['CONF', 1.0, 'mse']  # 置信度回归使用MSE损失
    - ['DET', 0.5, 'cross_entropy']  # 检测分类使用交叉熵损失
TRAIN:
  PRETRAINING: 'weights/trufor_ph2/best.pth.tar'  # 使用阶段一的最佳模型
  BATCH_SIZE_PER_GPU: 18  # 更大的batch size以稳定训练
  END_EPOCH: 100  # 更多的训练轮次
VALID:
  BEST_KEY: 'avg_det_bacc'  # 以检测的平衡准确率作为最佳模型指标
```

**训练命令**：
```bash
python train.py -exp trufor_ph3 TRAIN.PRETRAINING "weights/trufor_ph2/best.pth.tar"
```

**训练输出**：
- **模型文件**：
  - `checkpoint.pth.tar`：每个epoch结束时保存，包含当前epoch、最佳值、最佳指标键、模型状态字典和优化器状态
  - `best.pth.tar`：当验证集上的`avg_det_bacc`指标达到最佳时保存
- **日志输出**：
  - 训练和验证损失
  - 定位指标（mIoU、F1等）
  - 置信度指标（MSE、mIoU_CONF等）
  - 检测指标（tpr、tnr、bacc等）
  - 混淆矩阵
- **可视化输出**：
  - TensorBoard日志，记录所有训练和验证指标

**输出目录**：所有输出默认保存在`output/trufor_ph3`目录下。

## 3. 训练过程详解

### 3.1 训练循环

训练循环在`train.py`中实现，主要步骤包括：

1. **参数解析**：解析命令行参数，设置GPU设备
2. **配置更新**：根据命令行参数更新配置
3. **日志初始化**：创建日志记录器和TensorBoard写入器
4. **数据加载**：创建训练和验证数据集和数据加载器
5. **模型初始化**：加载模型、设置多GPU并行
6. **优化器初始化**：根据配置创建优化器
7. **预训练加载**：加载预训练模型（如需要）
8. **训练迭代**：
   - 遍历每个epoch
   - 在每个epoch中，先打乱数据集（用于类别平衡采样）
   - 执行训练函数，更新模型参数
   - 保存检查点
   - 执行验证函数，评估模型性能
   - 根据验证指标更新最佳模型

### 3.2 训练函数

`lib/core/function.py`中的`train()`函数实现了模型训练逻辑：

- 设置模型为训练模式
- 使用平均计算器记录损失和时间
- 遍历训练数据批次
- 前向传播计算损失
- 反向传播更新参数
- 调整学习率
- 记录训练损失和学习率到TensorBoard

### 3.3 验证函数

`lib/core/function.py`中的`validate()`函数实现了模型验证逻辑：

- 设置模型为评估模式
- 遍历验证数据批次
- 前向传播计算预测结果
- 计算各种评估指标
- 记录所有指标到TensorBoard
- 返回指标字典、IoU数组和混淆矩阵

## 4. 评估指标

TruFor模型使用多种评估指标来衡量性能：

### 4.1 定位指标

- **mIoU**：平均交并比
- **mIoU_smooth**：平滑后的平均交并比（避免分母为零）
- **avg_p-F1_smooth**：平滑后的像素级F1分数，在阶段一中作为最佳模型选择标准
- **pixel_acc**：像素级准确率

### 4.2 置信度指标

- **avg_mse_CONF**：置信度估计的平均均方误差
- **avg_mIoU_CONF**：置信度估计的平均交并比

### 4.3 检测指标

- **avg_det_tpr**：检测的真阳性率
- **avg_det_tnr**：检测的真阴性率
- **avg_det_bacc**：检测的平衡准确率，在阶段二中作为最佳模型选择标准

## 5. 模型预测

### 5.1 预测命令

使用训练好的模型进行预测：

```bash
python test.py -in /path/to/input/images -out /path/to/output/folder -exp trufor_ph3 TEST.MODEL_FILE "weights/trufor_ph3/best.pth.tar"
```

参数说明：
- `-g, --gpu`：GPU设备ID，默认为0，使用-1表示CPU模式
- `-in, --input`：输入路径，可以是单个文件、目录或glob模式（如`data/*.jpg`）
- `-out, --output`：输出结果路径，可以是目录（自动创建对应结构）或具体文件名
- `-exp, --experiment`：实验名称，默认为`trufor_ph3`
- `--save_np`：是否保存Noiseprint++特征图（可选）
- `TEST.MODEL_FILE`：模型权重文件路径（必需）

### 5.2 输入数据配置

测试数据集的构建非常简单，只需要准备图像文件即可：

- **支持的输入格式**：
  - 单个图像文件：`-in /path/to/image.jpg`
  - 图像目录：`-in /path/to/image_folder`
  - Glob模式：`-in /path/to/folder/*.jpg`

- **图像要求**：
  - 支持常见图像格式（JPG、PNG等）
  - 支持任意尺寸，模型会自动处理不同分辨率的输入
  - 无需标签文件或特殊的目录结构

### 5.3 预测流程

`test.py`的执行流程如下：

1. **参数解析与配置**：解析命令行参数并更新配置
2. **数据加载**：根据输入路径加载图像文件列表，创建`TestDataset`实例
3. **模型初始化**：加载指定的模型权重文件，准备推理环境
4. **批量预测**：
   - 以batch size=1进行推理（支持任意输入尺寸）
   - 对输入图像进行预处理并前向传播
   - 生成定位图、置信度图和检测得分
5. **结果保存**：将预测结果保存为`.npz`文件

### 5.4 预测输出

预测结果保存为`.npz`文件，包含以下内容：

- **map**：异常定位图，表示每个像素被篡改的概率，形状为[H,W]
- **conf**：置信度图，表示模型对定位结果的置信程度，形状为[H,W]
- **score**：整体篡改检测得分，范围在[0,1]之间，表示整个图像被篡改的概率
- **imgsize**：输入图像的尺寸，格式为(H,W)元组
- **np++**：Noiseprint++特征图（仅当使用`--save_np`标志时保存），形状为[H,W]

**输出目录结构**：
- 当指定输出为目录时，会保持与输入相同的目录结构
- 文件命名与输入图像相同，但扩展名为`.npz`
- 自动创建不存在的目录结构

**输出格式示例**：
```python
# 加载预测结果
import numpy as np
results = np.load('output/result.npz')

# 访问各个输出项
localization_map = results['map']  # 形状为 [H, W] 的数组
confidence_map = results['conf']   # 形状为 [H, W] 的数组
detection_score = results['score'] # 标量值，范围 [0,1]
image_size = results['imgsize']    # 元组，如 (H, W)

# 如果使用了--save_np标志
if 'np++' in results:
    noiseprint = results['np++']   # Noiseprint++特征图

# 使用预测结果的示例
print(f"图像尺寸: {image_size}")
print(f"篡改检测得分: {detection_score:.4f}")
print(f"定位图形状: {localization_map.shape}")
print(f"置信度图形状: {confidence_map.shape}")

# 获取篡改区域的平均概率
mean_tampered_prob = localization_map.mean()
print(f"篡改区域平均概率: {mean_tampered_prob:.4f}")
```

### 5.5 预测命令示例

**示例1：预测单个图像**
```bash
python test.py -in test_image.jpg -out outputs/test_result.npz -exp trufor_ph3 TEST.MODEL_FILE "weights/trufor_ph3/best.pth.tar"
```

**示例2：预测目录中的所有图像**
```bash
python test.py -in dataset/test_images/ -out outputs/ -exp trufor_ph3 TEST.MODEL_FILE "weights/trufor_ph3/best.pth.tar"
```

**示例3：使用glob模式选择特定图像**
```bash
python test.py -in "dataset/test_images/*.jpg" -out outputs/ -exp trufor_ph3 TEST.MODEL_FILE "weights/trufor_ph3/best.pth.tar"
```

**示例4：保存Noiseprint++特征**
```bash
python test.py -in dataset/test_images/ -out outputs/ -save_np -exp trufor_ph3 TEST.MODEL_FILE "weights/trufor_ph3/best.pth.tar"
```

**示例5：使用CPU进行预测**
```bash
python test.py -g -1 -in dataset/test_images/ -out outputs/ -exp trufor_ph3 TEST.MODEL_FILE "weights/trufor_ph3/best.pth.tar"
```

## 6. 结果可视化

使用`visualize.py`脚本可以直观地查看模型的预测结果：

### 6.1 可视化命令

```bash
python visualize.py --image image_path --output output_path [--mask mask_path]
```

参数说明：
- `--image`：原始输入图像路径（必需）
- `--output`：预测结果的`.npz`文件路径（必需）
- `--mask`：真实篡改掩码路径（可选，用于对比）

### 6.2 可视化原理

`visualize.py`的工作流程如下：

1. **参数解析**：解析命令行参数，获取图像路径、结果文件路径和可选的掩码路径
2. **数据加载**：
   - 加载`.npz`格式的预测结果
   - 加载原始图像
   - 加载可选的真实掩码
3. **可视化布局**：根据是否有掩码和Noiseprint++特征动态调整子图数量
4. **结果渲染**：生成包含多个子图的可视化结果
5. **显示结果**：使用matplotlib显示可视化结果

### 6.3 可视化输出内容

可视化结果包含以下子图（从左到右排列）：

1. **原始图像**：显示输入的原始图像
2. **真实掩码**（可选）：如果提供了掩码路径，则显示真实的篡改掩码（二值图像）
3. **Noiseprint++特征图**（可选）：如果预测结果中包含`np++`数据，则显示Noiseprint++特征（经过下采样处理）
4. **定位热图**：使用`RdBu_r`颜色映射显示篡改区域定位结果，红色表示高篡改概率
5. **置信度热图**：使用灰度颜色映射显示模型对定位结果的置信程度

在可视化结果的顶部，会显示模型预测的整体篡改检测得分。

### 6.4 可视化命令示例

**示例1：基本可视化（无真实掩码）**
```bash
python visualize.py --image dataset/test_images/tampered.jpg --output outputs/tampered.npz
```

**示例2：带真实掩码的可视化**
```bash
python visualize.py --image dataset/test_images/tampered.jpg --output outputs/tampered.npz --mask dataset/masks/tampered_mask.png
```

**示例3：查看带Noiseprint++特征的结果**
```bash
# 首先使用--save_np参数生成包含Noiseprint++特征的结果
python test.py -in dataset/test_images/tampered.jpg -out outputs/tampered_with_np.npz --save_np -exp trufor_ph3 TEST.MODEL_FILE "weights/trufor_ph3/best.pth.tar"
# 然后可视化
python visualize.py --image dataset/test_images/tampered.jpg --output outputs/tampered_with_np.npz
```

### 6.5 批量可视化方法

由于`visualize.py`只能处理单个文件，对于批量图像的可视化，可以创建一个简单的批处理脚本：

```python
# batch_visualize.py
import os
import subprocess
import glob

# 设置路径
image_dir = 'dataset/test_images/'
output_dir = 'outputs/'
vis_output_dir = 'visualizations/'

# 创建输出目录
os.makedirs(vis_output_dir, exist_ok=True)

# 获取所有图像文件
image_files = glob.glob(os.path.join(image_dir, '*'))

for image_path in image_files:
    # 获取图像文件名
    image_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(image_name)[0]
    
    # 对应的.npz文件路径
    npz_path = os.path.join(output_dir, name_without_ext + '.npz')
    
    if os.path.exists(npz_path):
        # 执行可视化命令
        cmd = f'python visualize.py --image "{image_path}" --output "{npz_path}"'
        print(f'Processing {image_name}...')
        subprocess.run(cmd, shell=True)
    else:
        print(f'No prediction found for {image_name}')
```

运行批处理脚本：
```bash
python batch_visualize.py
```

### 6.6 可视化结果解读

- **得分（Score）**：值越高表示图像被篡改的可能性越大
- **定位热图**：红色区域表示模型预测的篡改区域，蓝色区域表示未篡改区域
- **置信度热图**：白色区域表示模型对预测结果非常有信心，黑色区域表示低置信度
- **Noiseprint++特征图**：显示图像中的噪声特征模式，帮助理解模型如何检测篡改痕迹

通过这些可视化结果，可以直观地评估模型的性能，特别是在篡改区域定位和整体检测方面的准确性。

## 7. 配置文件详解

### 7.1 默认配置

默认配置位于`lib/config/default.py`，包含以下主要部分：

- **基本配置**：输出目录、日志目录、GPU设置等
- **模型配置**：模型名称、预训练权重、使用的模块等
- **数据集配置**：数据集根路径、训练/验证集选择、类别数等
- **训练配置**：学习率、优化器、batch size、epoch数等
- **验证配置**：验证图像大小限制、最佳模型选择指标等

### 7.2 实验特定配置

每个训练阶段都有特定的配置文件，位于`lib/config/`目录下：

- **trufor_ph2.yaml**：阶段一训练配置
- **trufor_ph3.yaml**：阶段二训练配置

这些配置文件会覆盖默认配置中的相应参数。

## 8. 注意事项

### 8.1 硬件要求

- **训练要求**：
  - 阶段一训练（定位网络）使用batch size 1，需要较小的GPU内存
  - 阶段二训练使用较大的batch size（默认18），建议使用具有至少11GB显存的GPU
  - 支持多GPU训练

- **推理要求**：
  - 支持GPU和CPU模式
  - 单个图像推理内存占用较小，适合在普通配置的机器上运行

### 8.2 数据准备

- **训练数据**：
  - 确保数据集列表文件（如`IMD_train_list.txt`）格式正确
  - 数据集列表文件包含图像路径和对应的ground truth掩码路径
  - 图像和掩码需要具有相同的尺寸

- **测试数据**：
  - 只需准备图像文件，无需标签
  - 支持任意尺寸和常见格式

### 8.3 训练结果保存

- 训练过程中会定期保存检查点（`checkpoint.pth.tar`）
- 根据验证指标保存最佳模型（`best.pth.tar`）
- 所有训练日志和TensorBoard记录都会保存在相应的目录中

### 8.4 模型兼容性

- 代码中使用了`weights_only=False`参数以增强模型加载的兼容性
- 支持不同版本的albumentations库

### 8.5 自定义配置

- 可以通过命令行参数覆盖配置文件中的设置
- 例如：`python train.py -exp trufor_ph3 TRAIN.BATCH_SIZE_PER_GPU 8`将覆盖配置文件中的batch size设置

### 8.6 预测与可视化最佳实践

- 使用`best.pth.tar`模型进行测试，而非`checkpoint.pth.tar`，因为前者是在验证集上表现最好的模型
- 对于批量图像测试，建议使用目录输入方式，保持文件结构一致性
- 保存Noiseprint++特征会增加存储空间使用，但有助于更深入分析模型行为
- 在没有真实掩码的情况下，可以使用基本可视化命令快速查看模型预测结果

### 8.7 常见问题排查

- **CUDA内存不足**：减小batch size或使用`-g -1`切换到CPU模式
- **模型文件找不到**：确保`TEST.MODEL_FILE`参数指定了正确的文件路径
- **输出文件未生成**：检查输入图像格式是否支持，或是否有权限问题
- **可视化不显示**：确保安装了matplotlib等必要的可视化库

