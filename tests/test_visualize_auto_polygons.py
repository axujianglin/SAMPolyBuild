"""Model-free tests for the single-image Auto polygon overlay utility."""

import json
import tempfile
import unittest
from pathlib import Path

from tools.visualize_auto_polygons import load_auto_results, select_polygon_rings


class VisualizeAutoPolygonsTest(unittest.TestCase):
    def test_selects_matching_predictions_and_applies_score_threshold(self):
        results = [
            {
                "image_id": "image001",
                "score": 0.91,
                "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
            },
            {
                "image_id": "image001",
                "score": 0.05,
                "segmentation": [[20, 20, 30, 20, 30, 30]],
            },
            {
                "image_id": "another_image",
                "score": 0.99,
                "segmentation": [[1, 1, 2, 1, 2, 2]],
            },
        ]

        rings, prediction_count = select_polygon_rings(results, "image001", 0.1)

        self.assertEqual(prediction_count, 1)
        self.assertEqual(len(rings), 1)
        self.assertEqual(rings[0][0], [0.0, 0.0])
        self.assertEqual(rings[0][-1], [0.0, 10.0])

    def test_accepts_flat_segmentation_and_nested_image_id(self):
        results = [{
            "image_id": "nested/image001",
            "score_cls": 0.8,
            "segmentation": [0, 0, 5, 0, 5, 5],
        }]

        rings, prediction_count = select_polygon_rings(results, "image001", 0.5)

        self.assertEqual(prediction_count, 1)
        self.assertEqual(rings, [[[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]])

    def test_rejects_invalid_polygon_coordinates(self):
        results = [{
            "image_id": "image001",
            "score": 0.9,
            "segmentation": [[0, 0, 10, 0, float("nan"), 10]],
        }]

        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            select_polygon_rings(results, "image001", 0.1)

    def test_reports_image_id_mismatch(self):
        results = [{
            "image_id": "other",
            "score": 0.9,
            "segmentation": [[0, 0, 10, 0, 10, 10]],
        }]

        with self.assertRaisesRegex(ValueError, "No Auto results match"):
            select_polygon_rings(results, "image001", 0.1)

    def test_loads_results_json_and_rejects_non_list_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "results.json"
            valid_path.write_text(
                json.dumps([{"image_id": "image001"}]), encoding="utf-8"
            )
            self.assertEqual(load_auto_results(valid_path)[0]["image_id"], "image001")

            invalid_path = root / "invalid.json"
            invalid_path.write_text(json.dumps({"image_id": "image001"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "root must be a list"):
                load_auto_results(invalid_path)


if __name__ == "__main__":
    unittest.main()
