# VisionTrack 本机复现记录

两阶段完整训练已完成。测试数据、全量推理和指标计算流程见 [EVALUATION.md](EVALUATION.md)。

## 工作目录与环境

项目：`/ssd8t/huyq/GMT-main`。

独立 Python 3.10 环境：`/ssd8t/huyq/.venvs/gmt`。

```bash
source /ssd8t/huyq/.venvs/gmt/bin/activate
cd /ssd8t/huyq/GMT-main
```

GPU 检查确认机器配备 6 张 RTX A6000（每张约 48 GB），驱动版本
535.309.01。Codex 文件系统沙箱内没有 NVIDIA 设备节点；GPU 检查、CUDA
编译和训练需要能够访问宿主 GPU 的执行环境。宿主机驱动无需重装。

## 环境安装

```bash
python -m pip install 'setuptools<81' wheel ninja tensorboardX
python -m pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
CUDA_HOME=/usr/local/cuda-11.7 CC=gcc-11 CXX=g++-11 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 python -m pip install --no-build-isolation /ssd8t/huyq/.deps/detectron2
python -m pip check
```

安装命令应顺序执行，避免多个 pip 进程同时修改同一个环境。
当前环境的完整版本快照位于 `requirements-lock-local.txt`，其中 Detectron2
使用本机源码路径；迁移到其他机器时需同步源码并调整该路径。

Detectron2 源码位于 `/ssd8t/huyq/.deps/detectron2`，下载时提交为
`b0fe2f0a5bb6496f65246ef35687a386a9f19202`。仓库未提供作者使用的确切
Detectron2 提交。该版本已通过实际导入、CUDA 算子及两阶段模型构建检查。

本机已有 CUDA 11.7 编译工具链，PyTorch 按 README 使用 CUDA 11.8 版本。
两者存在 CUDA 次版本差异；本机已编译成功，并通过 ROIAlign 与调制可变形
卷积的 CUDA 前向、反向验证。

原始依赖将 NumPy 固定为 1.24.1，却将 OpenCV 固定为要求 NumPy 2 的
4.12.0.88，pip 已确认无法解析。将 OpenCV 调整为 4.11.0.86，其余原始
依赖版本保留。

## 数据

代码注册的实际路径为 `datasets/VisionTrack/images/`，与 README 示例
存在差异。本地使用符号链接：

- `datasets/VisionTrack/images/train` → `/ssd8t/huyq/VisionTrack/train`
- `datasets/VisionTrack/images/annotations` → `/ssd8t/huyq/gmt_json/VisionTrack`

`VisionTrack/train.zip` 包含 57,508 张图像，与 `train.json` 的全部图像
路径一致。训练标注包含 596,749 个框、24 个视频组。
`train_stage1.json` 包含 479 个正身份；加载器会将身份重新映射为连续编号。
三个 JSON 的图像均包含加载器要求的 `view_id`、`ids_range`、`len_seq`
等字段。

解压后已检查全部图像路径与归档文件大小，并抽样解码 25 张图像，全部通过。

测试 JSON 与测试图像均已准备完成，测试归档解压并校验通过。

## 训练入口

训练与测试脚本已去除写死的 GPU 编号，使用调用方指定的
`CUDA_VISIBLE_DEVICES`。以下命令中的 `GPU_INDEX` 必须换成运行时选定的
GPU 编号；开始前通过 `nvidia-smi` 检查资源。

先进行两次迭代的小规模检查：

```bash
CUDA_VISIBLE_DEVICES=GPU_INDEX python train_net.py --num-gpus 1 --config-file configs/VISION_local_stage1.yaml SOLVER.MAX_ITER 2 DATALOADER.NUM_WORKERS 0 INPUT.TRAIN_SIZE 640 INPUT.VIDEO.TRAIN_LEN 2 OUTPUT_DIR ./output/vision_smoke_stage1
CUDA_VISIBLE_DEVICES=GPU_INDEX python train_net.py --num-gpus 1 --config-file configs/VISION_local_stage2.yaml MODEL.WEIGHTS ./output/vision_smoke_stage1/model_final.pth SOLVER.MAX_ITER 2 DATALOADER.NUM_WORKERS 0 INPUT.TRAIN_SIZE 640 INPUT.VIDEO.TRAIN_LEN 2 MODEL.ASSO_HEAD.ASSO_THRESH 0.01 OUTPUT_DIR ./output/vision_smoke_stage2
```

上述两阶段各两次迭代的检查已在 GPU 2 上完成，退出码均为 0。
第二阶段检查临时降低关联候选框阈值，确保低分辨率、未充分训练的检测器能
提供候选框；正式配置仍使用作者的 0.3 阈值。

- 第一阶段记录的 `loss_reid` 为 14.3153，总损失为 15.7197。
- 第二阶段记录的 `loss_asso` 为 141790.1992，总损失为 141792.9006。
- 两阶段均生成 `output/vision_smoke_stage*/model_final.pth`。
- 日志为 `output/reproduction/stage1-smoke-finite.log` 与
  `output/reproduction/stage2-smoke.log`。

这些检查只证明数据读取、损失计算、反向传播与保存能够执行。第二阶段
尚未收敛，初始关联损失较大，不能作为正式训练效果或论文指标复现结果。

低分辨率检查曾遇到候选框为空，原身份损失对空张量求均值产生 NaN。
`make_loss_with_center` 已对空特征返回可反传的零损失，使该批次仅训练检测器。

再运行完整训练：

```bash
CUDA_VISIBLE_DEVICES=GPU_INDEX python train_net.py --num-gpus 1 --config-file configs/VISION_local_stage1.yaml
CUDA_VISIBLE_DEVICES=GPU_INDEX python train_net.py --num-gpus 1 --config-file configs/VISION_local_stage2.yaml
```

本机配置覆盖预训练权重、输出目录和每 500 次迭代保存检查点的频率，
保留作者配置中的训练超参数。训练循环接入原已创建的
`PeriodicCheckpointer`，使检查点包含迭代信息，并在结束时生成
`model_final.pth`；原循环仅手动保存模型，没有生成该最终文件。
第一阶段使用已有的 `/ssd8t/huyq/CH_FPN_1x.pth`；第二阶段加载第一阶段的
`model_final.pth`。

本机配置设置 `MODEL.DLA.PRETRAINED: False`，关闭构建阶段隐式下载
ImageNet 权重。模型随后仍通过 `MODEL.WEIGHTS` 加载已有检测器或第一阶段
权重；检测器权重的加载日志未报告缺少 DLA 骨干参数。

`GMTDatasetMapper` 原类内重复定义了 `__init__`、`from_config` 和
`__call__`，后面的实现覆盖 RGB 路径，强制读取 `img1V_cut`、`img1T`。
已将该实现拆为 `MultiModalGMTDatasetMapper` 独立类，使默认入口使用前面的
RGB 多视角采样逻辑，保持 `MULTI_MODAL: False`。

环境检查命令：

```bash
CUDA_VISIBLE_DEVICES=GPU_INDEX TORCH_HOME=/ssd8t/huyq/.cache/torch python tools/check_environment.py
```

论文正文描述每阶段 14,000 次迭代、batch size 8、Adam；仓库 VisionTrack
配置则是每阶段 20,000 次、`IMS_PER_BATCH: 1`、AdamW。先按仓库配置跑通，
正式对齐论文数字前需确认这些差异及视频采样器对 batch 的实际解释。

## 测试与结果对照

原 README 要求将项目 `evaluator.py` 用于 Detectron2 的评估流程，但该文件
包含场景列表、视角名称和输出路径等硬编码。本机使用独立入口执行原模型推理，
按实际图像元数据导出结果，无需替换 Detectron2 安装文件。

```bash
CUDA_VISIBLE_DEVICES=GPU_INDEX python tools/evaluate_vision.py --gt-sources dataset evaluation-kit json --invalid-gt skip-scene
```

论文 VisionTrack 对照结果：CVIDF1 81.3、CVMA 75.2、HOTA 66.2、IDF1 82.1。
需用仓库 TrackEval 与跨视角评估工具生成指标；训练成功本身不能视为完成
论文数值复现。
