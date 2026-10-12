# InternImage-L 实例分割推理

InternImage-L + MaskDINO 文档实例分割部署包。当前部署只执行一次整图推理：
所有图片保持宽高比缩放并居中填充到固定方形 Letterbox 画布。画布尺寸只由
`configs/internimage_l_maskdino.yaml` 的 `INPUT.IMAGE_SIZE` 定义，不再使用动态
矩形档位，也不再执行 ROI 二次推理。

## 运行

```bash
conda activate doc_instance_seg
cd 4-dewarp/internimage-l-instance
python infer.py
```

可在 IDE 中直接运行 `infer.py`。默认递归读取 `4-dewarp/TEST_dewarp/`，加载
`checkpoints/best_hd95_1024.pth`，输出到本包 `outputs/`；默认路径不依赖工作目录。

```bash
python infer.py --help
python infer.py --max-images 1
python infer.py ../TEST_dewarp/instance --output-dir outputs_instance
python infer.py /path/to/image.jpg --output-mode production
python infer.py --output-mode debug
```

权重来自固定 `1024×1024` Letterbox 训练；当前部署输入尺寸读取 YAML 的
`INPUT.IMAGE_SIZE`，插值使用 Pillow Lanczos。该推理尺寸变化不会改变权重本身。

置信度阈值默认 0.50，最多恢复 5 个候选实例。mask 恢复到原图后，每个候选
只保留面积最大的 8 连通域；该区域的原图前景面积必须严格大于 100 像素。
程序最终只输出合格候选中可见面积最大的一个实例。重复使用同一输出目录时，
会先清除该图片遗留的 `_instance_002` 等旧 mask。

在 `infer.py` 顶部可设置 IDE 默认输出模式：

```python
OUTPUT_MODE = "debug"         # 保存最终 mask 叠加图
# OUTPUT_MODE = "production"  # 保存原图尺寸 0/255 单通道 PNG mask
```

命令行 `--output-mode` 优先于该变量。常用阈值集中在
`instance_seg/settings.py`，模型结构配置位于
`configs/internimage_l_maskdino.yaml`。

## 输出

```text
outputs/
├── instances/  # production：<图名>_instance_001.png
└── overlays/   # debug：<图名>.jpg
```

| 模式 | 独立 mask | 叠加图 |
| --- | --- | --- |
| `production` | 是 | 否 |
| `debug` | 否 | 是 |

输出保留输入目录的相对子目录结构。

## 目录

```text
internimage-l-instance/
├── infer.py                  # 启动入口
├── configs/                  # 模型结构与唯一的部署输入尺寸配置
├── checkpoints/              # 部署权重及来源记录
├── instance_seg/
│   ├── settings.py           # 路径与推理默认参数
│   ├── cli.py                # 命令行参数及校验
│   ├── pipeline.py           # 单次推理、输出和耗时统计
│   ├── runtime.py            # 权重加载、候选过滤和 CUDA 计时
│   ├── config.py             # Detectron2 配置加载
│   ├── preprocessing.py      # 固定方形 Letterbox 预处理
│   ├── geometry.py           # mask 外接框计算
│   ├── visualization.py      # 最终 mask 叠加图
│   ├── io.py                 # 递归读图和文件输出
│   └── models/               # InternImage、MaskDINO、CUDA 算子
├── tests/                    # 部署行为回归检查
├── docs/deployment.md        # 环境、权重来源与编译说明
├── licenses/                 # 上游代码许可证
└── requirements.txt
```

配置中的 `MODEL.MaskDINO.TWO_STAGE: true` 是 MaskDINO 模型内部的解码器结构，
是现有 checkpoint 的组成部分；它不表示“整图一次、ROI 再推一次”，不能作为
业务二阶段开关删除。

训练工程位于 `../../Instance/Train_InternImage-L-Instance/`。部署模块使用独立
包名 `instance_seg`，与训练侧 `doc_instance_seg` 区分。

## 验证

```bash
python -m unittest discover -s tests -v
```

回归检查覆盖 YAML 配置尺寸的固定方形 Letterbox、默认路径、轻量参数解析、已删除 ROI 参数、
递归文件筛选、候选连通域/面积过滤、最终最大面积选择和两种输出模式。真实模型
冒烟验证可运行 `python infer.py --max-images 1`。

## 权重来源

`best_hd95_1024.pth` 由固定 `1024×1024` Letterbox 输入训练得到。来源、选模
指标和 SHA-256 记录在 `checkpoints/model_1024.json`。完整训练结果位于：

```text
Instance/Train_InternImage-L-Instance/doc_instance_seg/
└── outputs/train_runs/internimage_l_maskdino_1024_20260820_174342/
```

该训练完成 296,910 次迭代，默认 `best_hd95` 在第 19,794 次迭代选出，单实例
验证 `dataset_p95_hd95` 为 31.773448。该指标属于训练时统一 1024 Letterbox
评估口径，不应直接解释成当前部署尺寸推理结果的像素误差。
