import importlib.metadata
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "third_party" / "CenterNet2"))

    import torch
    import torchvision
    import detectron2
    from detectron2.layers import ModulatedDeformConv, ROIAlign
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from centernet.config import add_centernet_config
    from gtr.config import add_gtr_config

    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("Detectron2:", detectron2.__version__)
    print("NumPy:", importlib.metadata.version("numpy"))
    print("OpenCV:", importlib.metadata.version("opencv-python"))
    print("PyTorch CUDA:", torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run where NVIDIA devices are accessible")

    print("GPU:", torch.cuda.get_device_name(0))
    features = torch.randn(1, 4, 8, 8, device="cuda", requires_grad=True)
    regions = torch.tensor([[0, 0, 0, 7, 7]], dtype=torch.float32, device="cuda")
    pooled = ROIAlign((2, 2), 1.0, 0, aligned=True)(features, regions)
    convolution = ModulatedDeformConv(4, 4, 3, padding=1).cuda()
    offsets = torch.zeros(1, 18, 8, 8, device="cuda")
    mask = torch.ones(1, 9, 8, 8, device="cuda")
    deformed = convolution(features, offsets, mask)
    (pooled.sum() + deformed.sum()).backward()
    torch.cuda.synchronize()
    print("CUDA ROIAlign and modulated deformable convolution forward/backward: PASS")

    for filename in ("VISION_local_stage1.yaml", "VISION_local_stage2.yaml"):
        config = get_cfg()
        add_centernet_config(config)
        add_gtr_config(config)
        config.merge_from_file(str(project_root / "configs" / filename))
        model = build_model(config)
        print(filename, "model construction: PASS", "parameters:",
              sum(parameter.numel() for parameter in model.parameters()))
        del model
    print("Environment checks: PASS")


if __name__ == "__main__":
    main()
