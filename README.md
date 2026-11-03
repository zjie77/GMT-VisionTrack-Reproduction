# GMT on VisionTrack: course reproduction

本仓库整理 GMT 在 VisionTrack 上的课程复现实验代码与已有评估记录。模型主体、第三方代码及许可证保留；本仓库不是 GMT 作者的官方实现。对应论文为 **GMT: Effective Global Framework for Multi-Camera Multi-Target Tracking**（[arXiv:2407.01007](https://arxiv.org/abs/2407.01007)，本次使用 v2，2025-11-24）。

## 目录

- `gtr/`、`layers/`、`third_party/`：模型和依赖源码。
- `configs/VISION_*.yaml`：VisionTrack 两阶段训练和测试配置。
- `tools/evaluate_vision.py`、`tools/vision_metrics.py`：本次评估入口与指标实现。
- `TrackEval/`、`MOTChallengeEvalKit_cv_test/`：评估工具源码；大体积 GT 需单独准备。
- `results/vision_evaluation/`：原始预测、指标、配置清单和标注审计。
- `results/六项指标汇总.csv`、`results/逐场景身份指标.csv`：课程报告汇总表。
- `docs/original_notes/`：原项目说明和既有机器上的实验记录，包含历史绝对路径。

## 六项指标

数值单位为 %，越高越好。

| 指标 | 论文 GMT | 原始标注（14场景） | 工具标注（14场景） | JSON标注（22场景） |
|---|---:|---:|---:|---:|
| HOTA | 66.20 | 65.72 | 69.02 | 75.05 |
| AssA | 69.40 | 71.92 | 75.00 | 73.27 |
| MOTA | 78.00 | 72.17 | 74.69 | 87.57 |
| IDF1 | 82.10 | 81.38 | 81.82 | 83.32 |
| CVIDF1 | 81.30 | 81.23 | 81.66 | 80.81 |
| CVMA / CVMA_TrackEval | 75.20 | 69.90 | 71.29 | 81.50 |

论文最后一行为 CVMA，本地为 **CVMA_TrackEval**，尚未核对与原 MATLAB 后端等价，不能作为统一协议下的直接优劣比较。原始标注和工具标注因8个身份冲突场景而仅评估14场景；JSON口径覆盖22场景、44个摄像头序列、58,038张图像。全部口径的标注、覆盖范围与论文配置存在差异，JSON数值较高不代表在相同协议下超越论文。

## 环境与数据

依照 `docs/original_notes/README.upstream.md` 和 `REPRODUCTION.md` 准备 Python 3.10、PyTorch 2.0.0、torchvision 0.15.1、CUDA 和 Detectron2，随后安装 `requirements.txt`。本次 Detectron2 记录的提交为 `b0fe2f0a5bb6496f65246ef35687a386a9f19202`。`requirements-lock-local.txt` 为历史环境快照，含本机安装路径，迁移时需修改。

数据和模型不在本仓库中。按原项目说明下载 VisionTrack 训练/测试图像、JSON 标注和检测器初始权重，并使以下实际注册路径可用：

```text
datasets/VisionTrack/images/train/
datasets/VisionTrack/images/test/
datasets/VisionTrack/images/annotations/
```

调整 `configs/VISION_local_stage1.yaml` 中的初始权重路径。第二阶段默认加载第一阶段最终权重。评估工具的 GT 文件也需要按原目录另行准备；不能用预测文件代替 GT。

## 训练和评估

在仓库根目录运行，根据机器资源选择 `CUDA_VISIBLE_DEVICES`：

```bash
python train_net.py --num-gpus 1 --config-file configs/VISION_local_stage1.yaml
python train_net.py --num-gpus 1 --config-file configs/VISION_local_stage2.yaml

python tools/evaluate_vision.py \
  --data-root /path/to/VisionTrack/test \
  --annotation-file /path/to/annotations/test.json \
  --gt-sources dataset evaluation-kit json \
  --invalid-gt skip-scene
```

推理输出默认写入 `output/vision_evaluation/`。已有产物单独存放于 `results/vision_evaluation/`，避免与新运行混用。要复查保存的六项汇总，不需要 GPU、模型或数据，可运行：

```bash
python tools/summarize_saved_metrics.py
```

重新计算检测/跟踪指标则仍需要原始 GT、对应的标注 JSON 和原实验版本；当前 `evaluate_vision.py` 即使采用 `--metrics-only` 也会校验检查点哈希，因此仍需准备相应权重。它还会校验输出 manifest，当前路径或源码与旧 manifest 不一致时应建立新实验目录，不应改写旧记录来绕过校验。

## 实验追溯与限制

- 本次 PDF 第5.1节为每阶段10,000次迭代、Adam、batch size 8；本地仓库为每阶段20,000次、AdamW、`IMS_PER_BATCH=1`，两阶段基础学习率为5e-5、2e-5。旧说明中的14,000次与本PDF不一致。
- `manifest.json` 记录当时的检查点、标注、关键源码哈希及随机种子。历史 `/ssd8t/huyq/...` 是实验位置，不是本仓库运行时强制路径。
- 当前5个关键推理源码中3个哈希与 manifest 一致；`gtr/modeling/meta_arch/gtr_rcnn.py` 和 `gtr/modeling/roi_heads/gtr_roi_heads.py` 两个文件不同。详见 `results/source_hash_verification.csv`。仓库为当前源码快照，保留的预测和指标为此前实验产物，两者不能声称逐字节对应。
- `results/source_files.csv` 记录从本地项目整理的每个文件及其SHA256；没有将原项目中其他任务的训练输出、数据集或模型权重纳入仓库。

## 许可证与引用

GMT 代码许可证保留于 `license.txt`；各第三方项目许可证保留在各自目录。请同时遵守各项目的原许可证。

```bibtex
@article{GMT2025,
  title={GMT: Effective Global Framework for Multi-Camera Multi-Target Tracking},
  author={Zhen, Yihao and Xu, Mingyue and Wang, Qiang and Fan, Baojie and Dong, Jiahua and Zhao, Tinghui and Fan, Huijie},
  journal={arXiv preprint arXiv:2407.01007},
  year={2025}
}
```
