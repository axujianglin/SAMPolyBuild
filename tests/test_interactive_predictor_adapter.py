"""Model-free checks for cropped HTTP predictor coordinates and embedding reuse."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from interactive_api.config import InteractiveServiceConfig
from interactive_api.errors import InteractiveAPIError
from interactive_api.predictor import InteractivePredictorAdapter


class FakePredictor:
    device = "cpu"

    def __init__(self):
        self.calls = []
        self.encoded_crops = []

    def reset_image(self):
        self.is_image_set = False

    def set_image_resize(self, image, image_format="RGB"):
        self.encoded_crops.append((image.shape[:2], image_format))
        self.original_size = image.shape[:2]
        self.input_size = (224, 224)

    def get_image_embedding(self):
        return torch.zeros((1, 2, 2, 2))

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        height, width = self.original_size
        mask = np.zeros((1, height, width), dtype=np.uint8)
        mask[0, 12:80, 12:80] = 1
        logits = np.ones((1, 4, 4), dtype=np.float32)
        vertices = {"vmap": torch.zeros((1, 1, 4, 4)), "voff": torch.zeros((1, 1, 4, 4))}
        return mask, np.asarray([0.83], dtype=np.float32), logits, vertices


def fake_polygon(**kwargs):
    left, top = kwargs["pos_transform"][:2]
    return np.asarray([[left + 12, top + 12], [left + 79, top + 12],
                       [left + 79, top + 79], [left + 12, top + 79]], dtype=float), 0.9


class PredictorAdapterTest(unittest.TestCase):
    def setUp(self):
        config = InteractiveServiceConfig(api_key="test-key", model_config=None, checkpoint=None)
        self.adapter = InteractivePredictorAdapter(config)
        self.adapter.predictor = FakePredictor()
        self.adapter.settings = SimpleNamespace(multi_mask=True, max_distance=10)
        self.adapter._initialized = True
        self.prepared = self.adapter.prepare_image(np.zeros((1024, 1024, 3), dtype=np.uint8))
        self.bbox = (584.26, 432.28, 704.12, 547.47)
        self.original_polygon = ((610, 455), (670, 455), (670, 520), (610, 520))

    def predict(self, *, bbox=None, previous_logits=None, positive_points=((652, 487),)):
        with patch("interactive_api.predictor.polygon_from_prediction_mask", side_effect=fake_polygon):
            return self.adapter.predict(
                prepared_image=self.prepared, bbox=bbox or self.bbox,
                positive_points=positive_points, negative_points=((690, 500),),
                original_polygon=self.original_polygon, previous_logits=previous_logits)

    def test_crop_coordinates_and_previous_logits(self):
        previous = np.full((1, 4, 4), 2.0, dtype=np.float32)
        result = self.predict(previous_logits=previous)
        left, top, right, bottom = self.adapter._crop_bounds(self.bbox, 1024, 1024)
        self.assertEqual(self.adapter.predictor.encoded_crops[0][0], (bottom - top, right - left))
        call = self.adapter.predictor.calls[0]
        np.testing.assert_allclose(call["point_coords"][1:],
                                   [[652 - left, 487 - top], [690 - left, 500 - top]])
        np.testing.assert_allclose(call["point_coords"][0],
                                   [(self.bbox[0] + self.bbox[2]) / 2 - left,
                                    (self.bbox[1] + self.bbox[3]) / 2 - top], atol=1e-3)
        self.assertEqual(call["point_labels"].tolist(), [1, 1, 0])
        np.testing.assert_allclose(call["box"], np.asarray(self.bbox) - [left, top, left, top], atol=1e-4)
        np.testing.assert_array_equal(call["mask_input"], previous)
        self.assertEqual(result.polygon[0], [float(left + 12), float(top + 12)])

    def test_same_bbox_reuses_embedding_and_logits(self):
        first = self.predict()
        self.predict(previous_logits=first.mask_logits, positive_points=((652, 487), (654, 489)))
        self.assertEqual(len(self.adapter.predictor.encoded_crops), 1)
        self.assertIsNone(self.adapter.predictor.calls[0]["mask_input"])
        self.assertIsNotNone(self.adapter.predictor.calls[1]["mask_input"])

    def test_different_bbox_gets_separate_embedding(self):
        self.predict()
        self.predict(bbox=(590, 430, 700, 550))
        self.assertEqual(len(self.adapter.predictor.encoded_crops), 2)
        self.assertEqual(len(self.prepared.crop_cache), 2)

    def test_polygon_escaping_crop_is_rejected(self):
        with patch("interactive_api.predictor.polygon_from_prediction_mask",
                   return_value=(np.asarray([[0, 0], [10, 0], [10, 10]]), 0.9)):
            with self.assertRaisesRegex(InteractiveAPIError, "escaped the target crop"):
                self.adapter.predict(prepared_image=self.prepared, bbox=self.bbox,
                                     positive_points=((652, 487),), negative_points=(),
                                     original_polygon=self.original_polygon)


if __name__ == "__main__":
    unittest.main()
