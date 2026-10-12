# 分割推理交付

本目录包含四套部署推理代码：

```text
4-dewarp/
├── TEST_dewarp/                        # 所有推理入口的默认测试图片
├── Lightweight-Segmentation-Only-Inference/ # 轻量级单分割推理
├── Joint-Segmentation-Keypoint-Detection-Inference/ # 分割与关键点检测联合推理
├── internimage-l-semantic-segmentation/ # InternImage-L + UPerHead 语义分割
└── internimage-l-instance/              # InternImage-L + MaskDINO 实例分割
```

语义分割输出合并的文档前景区域；实例分割分别输出每份文档的 mask。
两套 InternImage-L 推理包均保留原图叠加分割结果的可视化。

## InternImage-L 实例分割

`internimage-l-instance/` 包含部署推理、模型、独立配置和默认权重。
默认加载原始 1024 训练权重，以 YAML 中 `INPUT.IMAGE_SIZE` 配置的固定方形
Letterbox 进行单次整图推理；
不使用动态矩形档位，也不执行 ROI 二次推理。
在 `infer.py` 设置 `OUTPUT_MODE`：`debug` 输出叠加图，
`production` 只输出 `instances/` 中的 0/255 mask。使用 `doc_instance_seg` Conda 环境，在包内运行 `python infer.py`；
待处理图片统一放在 `4-dewarp/TEST_dewarp/`。具体参数与依赖见包内 README。
该包使用 PyTorch/CUDA，需要 NVIDIA GPU。

训练工程位于 `../Instance/Train_InternImage-L-Instance/`。

## InternImage-L 语义分割

`internimage-l-semantic-segmentation/` 由原 `Test_InternImage-L/` 重命名，
保留现有推理代码、权重和输出。
在包内运行 `python infer.py`；默认输入为 `../TEST_dewarp/`，权重为
`checkpoints/best_final_before.pth`。输出包括 mask 和叠加可视化。
训练工程位于 `../Semantic-Segmentation/Train_InternImage-L/`。

## 轻量级单分割与分割-关键点联合推理

`Lightweight-Segmentation-Only-Inference/` 使用轻量级 DeepLabV3+，只执行文档语义分割，
可输出 mask、edge 和混合图，入口为 `predict.py`。训练工程位于
`../Semantic-Segmentation/Lightweight-Segmentation-Only-Training/`。

`Joint-Segmentation-Keypoint-Detection-Inference/` 使用 C++/NCNN 联合执行书本分割与中缝关键点预测，
入口为 `run.sh`。训练和验证工程位于
`../Semantic-Segmentation/Joint-Segmentation-Keypoint-Detection-Training/`。
两套工程的运行方法详见各自 README。

所有批量推理入口默认递归读取 `TEST_dewarp/` 的图片，并在各自输出目录保留输入子目录结构。
