import argparse
import itertools
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.evaluate_tile_detections import (
    compact_number,
    is_finite_number,
    load_json,
    normalize_polygon_rings,
    require_finite_number,
    validate_bbox,
    write_json,
)


DEFAULT_IOU_THRESHOLD = 0.2
DEFAULT_CENTER_THRESHOLD = 30.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze likely duplicate detection pairs before merge or NMS."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to results_global_evaluated.json.",
    )
    parser.add_argument("--output", required=True, help="Output candidate_pairs.json path.")
    parser.add_argument(
        "--iou_thr",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="Candidate threshold for bbox IoU (strictly greater than).",
    )
    parser.add_argument(
        "--center_thr",
        type=float,
        default=DEFAULT_CENTER_THRESHOLD,
        help="Candidate threshold for center distance in pixels (strictly less than).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file.")
    return parser.parse_args()


def bbox_xyxy(bbox, detection_index):
    x, y, width, height = validate_bbox(bbox, detection_index)
    return x, y, x + width, y + height


def bbox_iou(box_a, box_b):
    intersection_width = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    intersection_height = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection_area = intersection_width * intersection_height
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def bbox_center(box):
    return (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5


def center_distance(center_a, center_b):
    return math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])


def rectangle_intersection(rect_a, rect_b):
    left = max(rect_a[0], rect_b[0])
    top = max(rect_a[1], rect_b[1])
    right = min(rect_a[2], rect_b[2])
    bottom = min(rect_a[3], rect_b[3])
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def has_positive_intersection(rect_a, rect_b):
    return rectangle_intersection(rect_a, rect_b) is not None


def polygon_array(detection, detection_index):
    if "segmentation" in detection:
        field_name = "segmentation"
    elif "polygon" in detection:
        field_name = "polygon"
    else:
        raise ValueError(
            f"Detection {detection_index} has neither segmentation nor polygon coordinates."
        )

    rings = normalize_polygon_rings(
        detection.get(field_name), detection_index, field_name
    )
    if not rings:
        raise ValueError(f"Detection {detection_index} {field_name} is empty.")
    if len(rings) != 1:
        raise ValueError(
            f"Detection {detection_index} has {len(rings)} polygon rings; "
            "the existing iou_poly utility supports one ring per detection."
        )
    ring = rings[0]
    return list(zip(ring[0::2], ring[1::2]))


def polygon_iou(polygon_a, polygon_b, det_a, det_b):
    try:
        from utils.metrics.iou import iou_poly

        value = float(iou_poly(polygon_a, polygon_b))
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Polygon IoU requires the project's Shapely, OpenCV, and NumPy dependencies. "
            "Run this script in the sampoly environment."
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"Unable to calculate polygon IoU for detections {det_a} and {det_b}: {exc}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Polygon IoU for detections {det_a} and {det_b} is not finite: {value!r}."
        )
    if value < -1e-9 or value > 1.0 + 1e-9:
        raise ValueError(
            f"Polygon IoU for detections {det_a} and {det_b} is outside [0, 1]: {value}."
        )
    return min(1.0, max(0.0, value))


def tile_rectangle(detection, detection_index):
    values = {}
    for field in ("x_offset", "y_offset", "tile_width", "tile_height"):
        values[field] = require_finite_number(
            detection.get(field), f"Detection {detection_index} field {field!r}"
        )
    if values["tile_width"] <= 0 or values["tile_height"] <= 0:
        raise ValueError(
            f"Detection {detection_index} tile_width and tile_height must be positive."
        )
    return (
        values["x_offset"],
        values["y_offset"],
        values["x_offset"] + values["tile_width"],
        values["y_offset"] + values["tile_height"],
    )


def merge_priority(detection, detection_index):
    quality = detection.get("tile_quality")
    if not isinstance(quality, dict):
        raise ValueError(f"Detection {detection_index} has no valid tile_quality object.")
    return require_finite_number(
        quality.get("merge_priority"),
        f"Detection {detection_index} tile_quality.merge_priority",
    )


def prepare_detection(detection, detection_index):
    if not isinstance(detection, dict):
        raise ValueError(f"Detection {detection_index} must be an object.")
    tile_id = detection.get("tile_id")
    if tile_id is None or str(tile_id) == "":
        raise ValueError(f"Detection {detection_index} has no valid tile_id.")
    box = bbox_xyxy(detection.get("bbox"), detection_index)
    return {
        "index": detection_index,
        "bbox": box,
        "center": bbox_center(box),
        "polygon": polygon_array(detection, detection_index),
        "priority": merge_priority(detection, detection_index),
        "tile_id": str(tile_id),
        "tile_rect": tile_rectangle(detection, detection_index),
    }


def pair_in_overlap(item_a, item_b):
    if item_a["tile_id"] == item_b["tile_id"]:
        return False
    shared_tile_area = rectangle_intersection(item_a["tile_rect"], item_b["tile_rect"])
    if shared_tile_area is None:
        return False
    return (
        has_positive_intersection(item_a["bbox"], shared_tile_area)
        and has_positive_intersection(item_b["bbox"], shared_tile_area)
    )


def analyze_pairs(results, iou_threshold, center_threshold):
    if not isinstance(results, list):
        raise ValueError("Evaluated results must contain a JSON list of detections.")
    detections = [
        prepare_detection(detection, index)
        for index, detection in enumerate(results)
    ]

    candidate_pairs = []
    for item_a, item_b in itertools.combinations(detections, 2):
        pair_bbox_iou = bbox_iou(item_a["bbox"], item_b["bbox"])
        pair_center_distance = center_distance(item_a["center"], item_b["center"])
        candidate = (
            pair_bbox_iou > iou_threshold
            or pair_center_distance < center_threshold
        )
        if not candidate:
            continue

        same_tile = item_a["tile_id"] == item_b["tile_id"]
        candidate_pairs.append({
            "det_a": item_a["index"],
            "det_b": item_b["index"],
            "bbox_iou": compact_number(pair_bbox_iou),
            "polygon_iou": compact_number(
                polygon_iou(
                    item_a["polygon"],
                    item_b["polygon"],
                    item_a["index"],
                    item_b["index"],
                )
            ),
            "center_distance": compact_number(pair_center_distance),
            "priority_a": compact_number(item_a["priority"]),
            "priority_b": compact_number(item_b["priority"]),
            "same_tile": same_tile,
            "in_overlap": pair_in_overlap(item_a, item_b),
            "candidate": True,
        })

    pair_count = len(candidate_pairs)
    same_tile_pairs = sum(pair["same_tile"] for pair in candidate_pairs)
    stats = {
        "detection_count": len(detections),
        "candidate_pair_count": pair_count,
        "average_iou": (
            sum(float(pair["bbox_iou"]) for pair in candidate_pairs) / pair_count
            if pair_count else 0.0
        ),
        "average_center_distance": (
            sum(float(pair["center_distance"]) for pair in candidate_pairs) / pair_count
            if pair_count else 0.0
        ),
        "cross_tile_pairs": pair_count - same_tile_pairs,
        "same_tile_pairs": same_tile_pairs,
    }
    return candidate_pairs, stats


def validate_thresholds(iou_threshold, center_threshold):
    if not is_finite_number(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("--iou_thr must be a finite number between 0 and 1.")
    if not is_finite_number(center_threshold) or center_threshold < 0:
        raise ValueError("--center_thr must be a finite, non-negative number.")


def run(args):
    validate_thresholds(args.iou_thr, args.center_thr)
    results_path = Path(args.results).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not results_path.is_file():
        raise FileNotFoundError(f"Evaluated results file not found: {results_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite.")

    results = load_json(results_path)
    candidate_pairs, stats = analyze_pairs(
        results,
        float(args.iou_thr),
        float(args.center_thr),
    )
    write_json(output_path, candidate_pairs)

    print(f"Detection count: {stats['detection_count']}")
    print(f"Candidate Pair count: {stats['candidate_pair_count']}")
    print(f"Average IoU: {stats['average_iou']:.6f}")
    print(f"Average center distance: {stats['average_center_distance']:.6f}")
    print(f"Cross-tile pair: {stats['cross_tile_pairs']}")
    print(f"Same-tile pair: {stats['same_tile_pairs']}")
    print(f"output path: {output_path}")
    return stats


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"Detection candidate pair analysis failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
