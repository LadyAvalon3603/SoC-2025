# 环境、权重与部署

默认权重为 `checkpoints/best_hd95_1024.pth`，复制自：

```text
Instance/Train_InternImage-L-Instance/doc_instance_seg/outputs/train_runs/
internimage_l_maskdino_1024_20260820_174342/checkpoints/best_hd95.pth
```

该权重训练时使用固定 `1024×1024` Letterbox。当前部署只进行一次模型前向，
预处理统一为固定方形 Letterbox，尺寸只读取 YAML 中的 `INPUT.IMAGE_SIZE`，使用
Pillow Lanczos 等比例缩放并以灰色居中填充。工程已删除动态矩形档位、ROI 裁剪
重推、二阶段候选验收和边界融合代码。权重来源、选模指标和校验值见
`../checkpoints/model_1024.json`。

配置中的 `MODEL.MaskDINO.TWO_STAGE: true` 属于 MaskDINO 内部解码器架构，
不是部署流水线中的第二次推理。为严格加载现有 checkpoint，该模型结构保持不变。

部署保留 `criterion.empty_weight` 兼容缓冲区以加载既有 checkpoint；模型不包含
训练损失执行路径。

已用环境基线为 Python 3.9.23、PyTorch 1.11.0、torchvision 0.12.0、CUDA 11.3、
Detectron2 0.6、MMCV 1.5.0、MMSegmentation 0.27.0、Pillow 9.5.0。
安装 Python 依赖后，需要针对目标环境编译 DCNv3 和
MultiScaleDeformableAttention：

```bash
python -m pip install ./instance_seg/models/ops_dcnv3 --no-build-isolation
python -m pip install ./instance_seg/models/maskdino/modeling/pixel_decoder/ops --no-build-isolation
```

## 输出控制

`infer.py` 顶部的 `OUTPUT_MODE` 可设为 `debug` 或 `production`：

- `debug`：仅保存最终 mask 叠加图；
- `production`：仅保存原图分辨率单通道 uint8 PNG，像素值为 0 和 255。

命令行 `--output-mode` 可以覆盖 IDE 默认设置。

## 单文档选择规则

一次整图推理最多恢复 5 个置信度排序候选。候选回到原图坐标后，每个 mask
只保留面积最大的 8 连通域，且该连通域面积必须严格大于 100 个原图像素。
最后只输出合格候选中可见面积最大的一个实例；仅当面积完全相同时，原有置信度
顺序才起到确定性仲裁作用。

## 固定输入约束

部署输入尺寸只有一个权威来源：`configs/internimage_l_maskdino.yaml` 中的
`INPUT.IMAGE_SIZE`。该值必须为 32 的正整数倍。横图、竖图和方图都进入由它定义的
同一个方形画布：原图宽高比保持不变，剩余区域对称填充。命令行不提供动态矩形
或 ROI 二阶段开关。
