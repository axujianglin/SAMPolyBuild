import unittest
from types import SimpleNamespace

import numpy as np

from interactive_api.config import InteractiveServiceConfig
from interactive_api.predictor import InteractivePredictorAdapter, PreparedImage


class FakePredictor:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def reset_image(self):
        self.is_image_set = False

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        mask = np.zeros((1, 16, 16), dtype=np.uint8)
        mask[3:11, 3:11] = 1
        logits = np.ones((1, 4, 4), dtype=np.float32)
        return mask, np.asarray([0.83], dtype=np.float32), logits, None


class PredictorAdapterTest(unittest.TestCase):
    def setUp(self):
        config = InteractiveServiceConfig(
            api_key="test-key",
            model_config=None,
            checkpoint=None,
        )
        self.adapter = InteractivePredictorAdapter(config)
        self.adapter.predictor = FakePredictor()
        self.adapter.settings = SimpleNamespace(multi_mask=True)
        self.adapter._initialized = True
        self.prepared = PreparedImage(np.zeros((1, 2, 2, 2)), (16, 16), (16, 16))

    def test_positive_negative_labels_order_and_previous_logits(self):
        previous = np.full((1, 4, 4), 2.0, dtype=np.float32)
        result = self.adapter.predict(
            prepared_image=self.prepared,
            bbox=(1, 1, 14, 14),
            positive_points=((5, 5), (7, 7)),
            negative_points=((13, 13),),
            original_polygon=((3, 3), (10, 3), (10, 10), (3, 10)),
            previous_logits=previous,
        )
        call = self.adapter.predictor.calls[0]
        self.assertEqual(call["point_coords"].tolist(), [[5, 5], [7, 7], [13, 13]])
        self.assertEqual(call["point_labels"].tolist(), [1, 1, 0])
        self.assertEqual(call["box"].tolist(), [1, 1, 14, 14])
        np.testing.assert_array_equal(call["mask_input"], previous)
        self.assertGreaterEqual(len(result.polygon), 3)
        self.assertEqual(len(result.bbox_xywh), 4)
        self.assertAlmostEqual(result.score, 0.83, places=5)
        self.assertEqual(result.mask_logits.shape, (1, 4, 4))

    def test_first_round_does_not_pass_previous_mask(self):
        self.adapter.predict(
            prepared_image=self.prepared,
            bbox=(1, 1, 14, 14),
            positive_points=((5, 5),),
            negative_points=(),
            original_polygon=((3, 3), (10, 3), (10, 10), (3, 10)),
        )
        self.assertIsNone(self.adapter.predictor.calls[0]["mask_input"])


if __name__ == "__main__":
    unittest.main()
