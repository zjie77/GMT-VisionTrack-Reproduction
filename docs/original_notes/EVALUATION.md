# VisionTrack 评估

## 数据与模型

测试压缩包 `/ssd8t/huyq/VisionTrack/test.zip` 已解压，全部归档文件大小核对通过。
测试 JSON 对应 22 个场景、44 个摄像头序列、58,038 张图像。
`datasets/VisionTrack/images/test` 链接到解压后的测试目录。

评估权重为 `output/vision_local_stage2/model_final.pth`，配置继承
`configs/VISION_local_test.yaml`。使用原配置的检测阈值、输入尺寸、历史窗口和
记忆库，不使用训练期间的低分辨率检查配置。

## 推理与恢复

```bash
source /ssd8t/huyq/.venvs/gmt/bin/activate
cd /ssd8t/huyq/GMT-main
CUDA_VISIBLE_DEVICES=2 TORCH_HOME=/ssd8t/huyq/.cache/torch TMPDIR=/ssd8t/huyq/.tmp/gmt python -u tools/evaluate_vision.py --inference-only
```

场景预测写入 `output/vision_evaluation/predictions/<场景>/View1.txt` 和
`View2.txt`，格式为 MOT 的 `frame,id,x,y,width,height,score,-1,-1,-1`。
每个场景完整写入后生成 `complete.json`；重复运行会跳过已完成场景。
结果采用覆盖式原子写入，不会重复追加检测框。

输出目录中保留 `manifest.json`，记录检查点和标注 JSON 的 SHA256、实际配置、
随机种子及关键推理源码哈希。配置或模型变化时必须使用新的输出目录。
同一目录使用文件锁，避免两个评估进程同时写入。

后台任务日志：`output/reproduction/evaluation.log`。
当前全量任务按已确认的策略运行：

```bash
CUDA_VISIBLE_DEVICES=2 TORCH_HOME=/ssd8t/huyq/.cache/torch TMPDIR=/ssd8t/huyq/.tmp/gmt python -u tools/evaluate_vision.py --gt-sources dataset evaluation-kit json --invalid-gt skip-scene
```

已有任务运行时无需重复执行。全部推理完成后自动计算三组指标。

```bash
tail -f output/reproduction/evaluation.log
```

首场景每个视角前 80 帧已通过实际推理、结果导出和指标计算检查。
该短序列结果在 `output/vision_evaluation_smoke/metrics.json`，不能作为全量结果。

## 标注检查

测试压缩包原始标注、作者 JSON 和评估工具随附标注存在差异，结果必须分别报告。
原始标注有 580,178 个有效框，评估工具随附标注有 576,133 个有效框，JSON
含 564,980 个标注框。

原始标注及评估工具标注有 8 个场景出现同一视角、同一帧、同一 ID 对应多个框，
包括不同位置的目标，不能简单当作完全相同的重复行删除。JSON 标注没有这种冲突。
冲突审计保存在 `output/vision_evaluation/ground_truth_audit.json`。

原始标注不做修改。默认指标计算遇到冲突时会报错。
如选择 `--invalid-gt skip-scene`，会排除冲突场景的全部视角，并在报告中明确列出
排除原因和实际覆盖场景。这种结果属于子集结果，不能直接冒充完整测试集结果。

原始 `00012night_View2` 未提供 `gt.txt`，读取同一归档内的六列 xywh 格式
`gt/View2.txt`。评估工具标注中，场景根目录的十列文件为 xyxy，嵌套 `gt/`
中的六列文件为 xywh，分别按对应格式转换。

## 指标计算

以下为本次补充对照流程。原 README 的指标评估入口是 TrackEval 和
MOTChallengeEvalKit_cv_test，分别读取其配套 GT；本脚本的 evaluation-kit
来源仅为跨视角评估包目录，其单视角指标不是直接运行原 TrackEval 数据集入口
得到的结果。JSON 来源则是额外转换后计算的完整测试集参考口径。

预测完成后，可独立计算指标，无需重复 GPU 推理：

```bash
python tools/evaluate_vision.py --metrics-only --gt-sources dataset evaluation-kit json --invalid-gt skip-scene
```

这条命令产生原始标注无冲突子集、工具标注无冲突子集与 JSON 完整测试集三组结果，
保存为 `output/vision_evaluation/metrics.json`。

计算使用仓库自带 TrackEval 的 HOTA、CLEAR、Identity 实现，按其
`combine_sequences` 汇总序列，非简单平均各序列的百分比。
MOTA/Identity 的匹配 IoU 阈值为 0.5，HOTA 按实现的 0.05–0.95 阈值计算。

跨视角 CVIDF1 在同一场景内按视角串接时间轴，保留跨视角共享 ID；
`CVMA_TrackEval` 使用原跨视角准备脚本的交错视角时间轴，并调用 TrackEval CLEAR。
本机无 MATLAB，该字段未与原 MATLAB 后端逐项核对，不能声称与论文 CVMA 数字
严格等价，因此使用单独的指标名称。

保留当前仓库推理代码中的随机框扰动和 `radius=5*sqrt(area)` 设置，未根据
测试集调整参数；固定每个场景的随机种子。这些实现细节与论文描述存在差异，
应在解读复现结果时一并考虑。
