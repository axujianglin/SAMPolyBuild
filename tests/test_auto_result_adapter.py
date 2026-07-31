"""Model-free tests for the Docker Auto response polygon adapter."""

import unittest

from services.auto_result_adapter import adapt_auto_results


class AutoResultAdapterTest(unittest.TestCase):
    def test_converts_matching_coco_result_to_original_pixel_coordinates(self):
        raw_results = [{
            "image_id": "image001",
            "score": 0.91,
            "score_cls": 0.94,
            "bbox": [10, 20, 30, 40],
            "segmentation": [[10, 20, 40, 20, 40, 60, 10, 60]],
            "category_id": 1,
            "area": 1200,
        }]

        polygons = adapt_auto_results(
            raw_results,
            image_id="image001",
            image_width=100,
            image_height=80,
            score_threshold=0.1,
        )

        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0]["id"], "image001_poly_0001")
        self.assertEqual(polygons[0]["bbox_xywh"], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(polygons[0]["points"][0], [10.0, 20.0])
        self.assertEqual(polygons[0]["metadata"]["raw_index"], 0)

    def test_uses_longest_ring_and_filters_by_classification_score(self):
        raw_results = [
            {
                "image_id": "nested/image001",
                "score": 0.8,
                "score_cls": 0.2,
                "segmentation": [[0, 0, 5, 0, 5, 5]],
            },
            {
                "image_id": "nested/image001",
                "score": 0.7,
                "score_cls": 0.9,
                "segmentation": [
                    [1, 1, 2, 1, 2, 2],
                    [10, 10, 30, 10, 30, 30, 10, 30],
                ],
            },
        ]

        polygons = adapt_auto_results(
            raw_results,
            image_id="image001",
            image_width=64,
            image_height=64,
            score_threshold=0.5,
        )

        self.assertEqual(len(polygons), 1)
        self.assertEqual(len(polygons[0]["points"]), 4)
        self.assertEqual(polygons[0]["metadata"]["raw_index"], 1)

    def test_derives_bbox_when_raw_bbox_is_missing(self):
        polygons = adapt_auto_results(
            [{
                "image_id": "image001",
                "score": 0.9,
                "segmentation": [[10, 15, 40, 15, 40, 50, 10, 50]],
            }],
            image_id="image001",
            image_width=100,
            image_height=100,
            score_threshold=0.1,
        )

        self.assertEqual(polygons[0]["bbox_xywh"], [10.0, 15.0, 30.0, 35.0])

    def test_rejects_non_finite_and_out_of_bounds_polygons(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            adapt_auto_results(
                [{
                    "image_id": "image001",
                    "score": 0.9,
                    "segmentation": [[0, 0, 10, 0, float("nan"), 10]],
                }],
                image_id="image001",
                image_width=20,
                image_height=20,
                score_threshold=0.1,
            )

        with self.assertRaisesRegex(ValueError, "exceeds image bounds"):
            adapt_auto_results(
                [{
                    "image_id": "image001",
                    "score": 0.9,
                    "segmentation": [[0, 0, 30, 0, 30, 10]],
                }],
                image_id="image001",
                image_width=20,
                image_height=20,
                score_threshold=0.1,
            )


if __name__ == "__main__":
    unittest.main()
