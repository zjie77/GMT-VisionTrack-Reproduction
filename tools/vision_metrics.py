import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TrackEval"))
from trackeval.metrics import CLEAR, HOTA, Identity


def read_rows(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return np.empty((0, 10))
    first_line = path.open().readline()
    return np.loadtxt(path, delimiter="," if "," in first_line else None, ndmin=2)


def box_iou(ground_truth, predictions):
    intersection_start = np.maximum(ground_truth[:, None, :2], predictions[None, :, :2])
    intersection_end = np.minimum(
        ground_truth[:, None, :2] + ground_truth[:, None, 2:4],
        predictions[None, :, :2] + predictions[None, :, 2:4],
    )
    intersection = np.maximum(0, intersection_end - intersection_start).prod(axis=2)
    ground_truth_area = np.maximum(0, ground_truth[:, 2:4]).prod(axis=1)
    prediction_area = np.maximum(0, predictions[:, 2:4]).prod(axis=1)
    union = ground_truth_area[:, None] + prediction_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def frame_records(ground_truth, predictions, length):
    for rows in (ground_truth, predictions):
        if not np.isfinite(rows).all():
            raise ValueError("Nonfinite tracking or ground-truth values")
        if len(rows) and (rows[:, 0].min() < 1 or rows[:, 0].max() > length):
            raise ValueError("Frame index outside the evaluated image range")
        if len(rows) != len(np.unique(rows[:, :2], axis=0)):
            raise ValueError("Duplicate frame/identity pairs")
    grouped = []
    for rows in (ground_truth, predictions):
        by_frame = [[] for _ in range(length)]
        for row in rows:
            by_frame[int(row[0]) - 1].append(row)
        grouped.append([np.asarray(items).reshape(-1, rows.shape[1]) for items in by_frame])
    return [
        (gt[:, 1].astype(int), prediction[:, 1].astype(int), box_iou(gt[:, 2:6], prediction[:, 2:6]))
        for gt, prediction in zip(*grouped)
    ]


def metric_data(records):
    gt_ids = np.unique(np.concatenate([record[0] for record in records]))
    prediction_ids = np.unique(np.concatenate([record[1] for record in records]))
    return {
        "num_timesteps": len(records),
        "num_gt_ids": len(gt_ids),
        "num_tracker_ids": len(prediction_ids),
        "num_gt_dets": sum(len(record[0]) for record in records),
        "num_tracker_dets": sum(len(record[1]) for record in records),
        "gt_ids": [np.searchsorted(gt_ids, record[0]) for record in records],
        "tracker_ids": [np.searchsorted(prediction_ids, record[1]) for record in records],
        "similarity_scores": [record[2] for record in records],
    }


def ground_truth_rows(source, scene, view, images, annotations, data_root):
    if source == "dataset":
        directory = data_root / f"{scene}_View{view}" / "gt"
        mot_file = directory / "gt.txt"
        if mot_file.is_file():
            rows = read_rows(mot_file)
            return rows[(rows[:, 6] != 0) & (rows[:, 7] == 1)]
        rows = read_rows(directory / f"View{view}.txt")
        if rows.shape[1] != 6:
            raise ValueError(f"Expected 6-column xywh ground truth: {directory}")
        return rows
    if source == "evaluation-kit":
        path = PROJECT_ROOT / "MOTChallengeEvalKit_cv_test/data/eval/vision/gt"
        direct = path / scene / f"View{view}.txt"
        if direct.is_file():
            rows = read_rows(direct)
            if rows.shape[1] != 10:
                raise ValueError(f"Expected 10-column xyxy ground truth: {direct}")
            rows[:, 4:6] -= rows[:, 2:4]
            return rows[rows[:, 6] != 0]
        nested = path / scene / "gt" / f"View{view}.txt"
        rows = read_rows(nested)
        if rows.shape[1] != 6:
            raise ValueError(f"Expected 6-column xywh ground truth: {nested}")
        return rows
    rows = []
    for image in images:
        for annotation in annotations.get(image["id"], []):
            if annotation.get("iscrowd", 0) or annotation["category_id"] != 1:
                continue
            rows.append([image["frame_id"], annotation["instance_id"], *annotation["bbox"], 1, 1, 1])
    return np.asarray(rows, dtype=float).reshape(-1, 9)


def ground_truth_issue(rows, length):
    if not np.isfinite(rows).all():
        return "Nonfinite ground-truth values"
    if len(rows) and (rows[:, 0].min() < 1 or rows[:, 0].max() > length):
        return "Ground-truth frame outside the image range"
    pairs, counts = np.unique(rows[:, :2], axis=0, return_counts=True)
    duplicates = pairs[counts > 1]
    if len(duplicates):
        return {
            "reason": "Duplicate frame/identity pairs",
            "pair_count": len(duplicates),
            "examples": [rows[(rows[:, 0] == pair[0]) & (rows[:, 1] == pair[1])].tolist()
                         for pair in duplicates[:3]],
        }
    return None


def evaluate_predictions(dataset, predictions_root, data_root, sources, max_frames=None, invalid_gt="error"):
    by_video = {}
    for image in dataset["images"]:
        by_video.setdefault(image["video_id"], {}).setdefault(image["view_id"], []).append(image)
    annotations = {}
    if "json" in sources:
        for annotation in dataset["annotations"]:
            annotations.setdefault(annotation["image_id"], []).append(annotation)
    metrics = {"HOTA": HOTA(), "CLEAR": CLEAR({"PRINT_CONFIG": False}),
               "Identity": Identity({"PRINT_CONFIG": False})}
    report = {}
    for source in sources:
        single = {name: {} for name in metrics}
        cross_identity = {}
        cross_clear = {}
        skipped = {}
        for video in dataset["videos"]:
            scene = video["file_name"]
            truth_by_view = {}
            issues = {}
            for view, images in sorted(by_video[video["id"]].items()):
                length = len(images) if max_frames is None else min(len(images), max_frames)
                gt = ground_truth_rows(source, scene, view, images, annotations, data_root)
                if max_frames is not None:
                    gt = gt[gt[:, 0] <= length]
                truth_by_view[view] = gt
                issue = ground_truth_issue(gt, length)
                if issue:
                    issues[f"View{view}"] = issue
            if issues:
                if invalid_gt == "error":
                    raise ValueError(f"Invalid ground truth [{source}] {scene}: {issues}")
                skipped[scene] = issues
                print(f"Excluded complete scene [{source}]: {scene}, invalid ground truth", flush=True)
                continue
            views = []
            for view, images in sorted(by_video[video["id"]].items()):
                length = len(images) if max_frames is None else min(len(images), max_frames)
                gt = truth_by_view[view]
                predictions = read_rows(predictions_root / scene / f"View{view}.txt")
                records = frame_records(gt, predictions, length)
                data = metric_data(records)
                for name, metric in metrics.items():
                    single[name][f"{scene}_View{view}"] = metric.eval_sequence(data)
                views.append(records)
            if len({len(view) for view in views}) != 1:
                raise ValueError(f"Unsynchronized view lengths in {scene}")
            concatenated = [record for view in views for record in view]
            interleaved = [record for instant in zip(*views) for record in instant]
            cross_identity[scene] = metrics["Identity"].eval_sequence(metric_data(concatenated))
            cross_clear[scene] = metrics["CLEAR"].eval_sequence(metric_data(interleaved))
            print(f"Metrics [{source}]: {scene}", flush=True)
        if not cross_identity:
            raise ValueError(f"No valid scenes remain for ground-truth source {source}")
        combined = {name: metrics[name].combine_sequences(results) for name, results in single.items()}
        identity = metrics["Identity"].combine_sequences(cross_identity)
        clear = metrics["CLEAR"].combine_sequences(cross_clear)
        report[source] = {
            "metrics_percent": {
                "HOTA": float(np.mean(combined["HOTA"]["HOTA"]) * 100),
                "AssA": float(np.mean(combined["HOTA"]["AssA"]) * 100),
                "MOTA": float(combined["CLEAR"]["MOTA"] * 100),
                "IDF1": float(combined["Identity"]["IDF1"] * 100),
                "CVIDF1": float(identity["IDF1"] * 100),
                "CVMA_TrackEval": float(clear["MOTA"] * 100),
            },
            "gt_detections": int(combined["Identity"]["IDTP"] + combined["Identity"]["IDFN"]),
            "predicted_detections": int(combined["Identity"]["IDTP"] + combined["Identity"]["IDFP"]),
            "per_view": {sequence: {name: single[name][sequence] for name in metrics}
                         for sequence in single["HOTA"]},
            "cross_view_identity": cross_identity,
            "cross_view_clear": cross_clear,
            "evaluated_scenes": list(cross_identity),
            "excluded_scenes": skipped,
            "covers_all_selected_scenes": not skipped,
        }
    return report


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2, default=json_value, allow_nan=False))
