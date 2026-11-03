import argparse
import copy
import gc
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party/CenterNet2"))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger
from centernet.config import add_centernet_config
from gtr.config import add_gtr_config
from gtr.data.custom_build_augmentation import build_custom_augmentation
from gtr.data.gtr_dataset_dataloader import get_GMT_video_dataset_dicts
from gtr.data.gtr_dataset_mapper import GMTDatasetMapper
from vision_metrics import evaluate_predictions, write_report


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_scene(outputs, images, views, destination):
    ordered = sorted(images, key=lambda image: (image["frame_id"], image["view_id"]))
    if len(outputs) != len(ordered):
        raise ValueError("Model output/image count mismatch")
    rows = {view: [] for view in views}
    for output, image in zip(outputs, ordered):
        instances = output["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy().copy()
        boxes[:, 2:4] -= boxes[:, :2]
        for identity, box, score in zip(instances.track_ids.tolist(), boxes, instances.scores.tolist()):
            if identity < 0:
                continue
            rows[image["view_id"]].append([image["frame_id"], identity, *box, score, -1, -1, -1])
    destination.mkdir(parents=True, exist_ok=True)
    for view, detections in rows.items():
        target = destination / f"View{view}.txt"
        temporary = target.with_suffix(".tmp")
        np.savetxt(temporary, np.asarray(detections).reshape(-1, 10), delimiter=",",
                   fmt=["%d", "%d", "%.4f", "%.4f", "%.4f", "%.4f", "%.6f", "%d", "%d", "%d"])
        temporary.replace(target)
    return {str(view): len(detections) for view, detections in rows.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/VISION_local_test.yaml")
    parser.add_argument("--output-dir", default="output/vision_evaluation")
    parser.add_argument("--data-root", default="../VisionTrack/test")
    parser.add_argument("--annotation-file", default="../gmt_json/VisionTrack/test.json")
    parser.add_argument("--gt-sources", nargs="+", choices=["dataset", "evaluation-kit", "json"],
                        default=["dataset", "evaluation-kit"])
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--invalid-gt", choices=["error", "skip-scene"], default="error")
    args = parser.parse_args()
    if args.metrics_only and args.inference_only:
        parser.error("--metrics-only and --inference-only cannot be combined")
    os.chdir(PROJECT_ROOT)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_lock = (output / ".evaluation.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    setup_logger(output=str(output), name="detectron2")
    config = get_cfg()
    add_centernet_config(config)
    add_gtr_config(config)
    config.merge_from_file(args.config_file)
    config.freeze()
    annotation_path = Path(args.annotation_file).resolve()
    dataset = json.loads(annotation_path.read_text())
    if args.max_scenes:
        dataset["videos"] = dataset["videos"][:args.max_scenes]
        selected = {video["id"] for video in dataset["videos"]}
        dataset["images"] = [image for image in dataset["images"] if image["video_id"] in selected]
    manifest = {
        "checkpoint": str(Path(config.MODEL.WEIGHTS).resolve()),
        "checkpoint_sha256": file_hash(Path(config.MODEL.WEIGHTS)),
        "annotation_sha256": file_hash(annotation_path),
        "config": config.dump(),
        "seed": args.seed,
        "scene_ids": [video["id"] for video in dataset["videos"]],
        "max_frames": args.max_frames,
        "inference_source_sha256": {
            filename: file_hash(PROJECT_ROOT / filename)
            for filename in ("gtr/modeling/meta_arch/gtr_rcnn.py",
                             "gtr/modeling/meta_arch/custom_rcnn.py",
                             "gtr/modeling/roi_heads/gtr_roi_heads.py",
                             "gtr/modeling/roi_heads/transformer.py",
                             "gtr/data/gtr_dataset_mapper.py")
        },
        "box_format": "xywh",
        "inference_notes": "Repository inference retained, including random box jitter and radius=5*sqrt(area).",
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Output manifest differs; select a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    predictions = output / "predictions"
    if not args.metrics_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        seed_all_rng(args.seed)
        model = build_model(config)
        DetectionCheckpointer(model).load(config.MODEL.WEIGHTS)
        model.eval()
        mapper = GMTDatasetMapper(config, False, augmentations=build_custom_augmentation(config, False))
        videos = get_GMT_video_dataset_dicts(config.DATASETS.TEST)
        image_metadata = {image["id"]: image for image in dataset["images"]}
        selected = set(manifest["scene_ids"])
        videos = [video for video in videos if video["video_id"] in selected]
        video_names = {video["id"]: video["file_name"] for video in dataset["videos"]}
        with torch.no_grad():
            for index, video in enumerate(videos, 1):
                scene = video_names[video["video_id"]]
                destination = predictions / scene
                completion = destination / "complete.json"
                if completion.exists():
                    print(f"Skip completed scene {scene}", flush=True)
                    continue
                seed_all_rng(args.seed + video["video_id"])
                video = copy.deepcopy(video)
                for image in video["images"]:
                    image["frame_id"] = image_metadata[image["image_id"]]["frame_id"]
                video["images"].sort(key=lambda image: (image["view_id"], image["frame_id"]))
                if args.max_frames:
                    video["images"] = [image for image in video["images"] if image["frame_id"] <= args.max_frames]
                views = sorted({image["view_id"] for image in video["images"]})
                frame_ids = [[image["frame_id"] for image in video["images"] if image["view_id"] == view]
                             for view in views]
                if any(frames != frame_ids[0] for frames in frame_ids) or frame_ids[0] != list(range(1, len(frame_ids[0]) + 1)):
                    raise ValueError(f"View synchronization mismatch in {scene}")
                print(f"Scene {index}/{len(videos)}: {scene}, {len(video['images'])} images", flush=True)
                start = time.perf_counter()
                inputs = mapper(video)
                outputs, view_count = model(inputs)
                if view_count != len(views):
                    raise ValueError("Unexpected model view count")
                counts = export_scene(outputs, video["images"], views, destination)
                completion.write_text(json.dumps({"scene": scene, "frames_per_view": len(frame_ids[0]),
                                                   "detections": counts, "seconds": time.perf_counter() - start}))
                print(f"Completed {scene}: {counts}", flush=True)
                del inputs, outputs, video
                gc.collect()
                torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
    for video in dataset["videos"]:
        if not (predictions / video["file_name"] / "complete.json").is_file():
            raise ValueError(f"Incomplete predictions: {video['file_name']}")
    if args.inference_only:
        print(f"All scene predictions saved: {predictions}", flush=True)
        return
    report = evaluate_predictions(dataset, predictions, Path(args.data_root).resolve(),
                                  args.gt_sources, args.max_frames, args.invalid_gt)
    report["protocol"] = {
        "engine": "Bundled TrackEval; combined sequence aggregation; IoU=0.5 for CLEAR/Identity",
        "cross_view": "Shared IDs; concatenated views for CVIDF1, interleaved views for CVMA_TrackEval",
        "limitation": "CVMA_TrackEval follows repository view interleaving but is not validated against its MATLAB backend.",
        "partial_run": args.max_frames is not None or args.max_scenes is not None,
        "invalid_ground_truth_policy": args.invalid_gt,
        "scenes": len(dataset["videos"]),
        "manifest": str(manifest_path),
        "dataset_ground_truth": str(Path(args.data_root).resolve()),
        "evaluation_kit_ground_truth": str(PROJECT_ROOT / "MOTChallengeEvalKit_cv_test/data/eval/vision/gt"),
    }
    write_report(output / "metrics.json", report)
    for source in args.gt_sources:
        print(source, report[source]["metrics_percent"], flush=True)
    print(f"Evaluation complete: {output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
