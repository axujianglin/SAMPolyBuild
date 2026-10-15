import io
import json
import unittest

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from interactive_api.app import create_app
from interactive_api.config import InteractiveServiceConfig
from interactive_api.errors import InteractiveAPIError
from interactive_api.predictor import PredictionResult


API_KEY = "unit-test-secret"


class MockPredictor:
    def __init__(self):
        self.initialized = False
        self.prepare_count = 0
        self.predict_calls = []
        self.fail_next = False

    def initialize(self):
        self.initialized = True

    def close(self):
        self.initialized = False

    def prepare_image(self, image):
        self.prepare_count += 1
        return {"embedding": np.asarray([self.prepare_count]), "shape": image.shape}

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        if self.fail_next:
            self.fail_next = False
            raise InteractiveAPIError("inference_failed", "mock failure", 500)
        return PredictionResult(
            polygon=[[3.0, 3.0], [10.0, 3.0], [10.0, 10.0], [3.0, 10.0]],
            bbox_xywh=[3.0, 3.0, 7.0, 7.0],
            score=0.9,
            mask_logits=np.full((1, 4, 4), len(self.predict_calls), dtype=np.float32),
            selection_strategy="positive_negative_prompts",
        )


def image_bytes(image_format="PNG", size=(16, 16)):
    stream = io.BytesIO()
    Image.new("RGB", size, color=(32, 64, 96)).save(stream, format=image_format)
    return stream.getvalue()


class InteractiveAPITest(unittest.TestCase):
    def setUp(self):
        self.predictor = MockPredictor()
        config = InteractiveServiceConfig(
            api_key=API_KEY,
            model_config=None,
            checkpoint=None,
            max_upload_bytes=1024 * 1024,
            session_ttl_seconds=60,
            max_sessions=2,
        )
        self.client_context = TestClient(create_app(config=config, predictor=self.predictor))
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def _create(self, *, request_id="create-1", name="tile.png", data=None, width=16, height=16):
        body = {
            "protocol_version": "2.0",
            "request_id": request_id,
            "coordinate_space": "original_image_pixel",
            "image": {"file_name": name, "width": width, "height": height},
        }
        return self.client.post(
            "/v1/interactive/sessions",
            headers={"X-API-Key": API_KEY, "X-Request-ID": request_id},
            files={
                "image": (name, data if data is not None else image_bytes(), "image/png"),
                "request": (None, json.dumps(body), "application/json"),
            },
        )

    def _refine_body(self, session_id, revision=1, token=""):
        return {
            "protocol_version": "2.0",
            "request_id": f"round-{revision}",
            "session_id": session_id,
            "polygon_id": "building_0006",
            "revision": revision,
            "coordinate_space": "original_image_pixel",
            "image": {"file_name": "tile.png", "width": 16, "height": 16},
            "bbox_xyxy": [1, 1, 14, 14],
            "positive_points": [[5, 5]],
            "negative_points": [],
            "original_polygon": [[3, 3], [10, 3], [10, 10], [3, 10], [3, 3]],
            "interaction_token": token,
        }

    def _refine(self, body):
        return self.client.post(
            f"/v1/interactive/sessions/{body['session_id']}/refine",
            headers={"X-API-Key": API_KEY, "X-Request-ID": body["request_id"]},
            json=body,
        )

    def test_create_png_jpeg_tiff_and_embedding_once_per_session(self):
        png = self._create()
        self.assertEqual(png.status_code, 201)
        payload = png.json()
        self.assertTrue(payload["session_id"])
        self.assertTrue(payload["image"]["image_id"])
        self.assertGreater(payload["expires_in_seconds"], 0)
        self.assertEqual(self.predictor.prepare_count, 1)
        self.client.delete(
            f"/v1/interactive/sessions/{payload['session_id']}",
            headers={"X-API-Key": API_KEY},
        )
        for image_format, suffix in (("JPEG", "jpg"), ("TIFF", "tif")):
            response = self._create(
                request_id=f"create-{suffix}",
                name=f"tile.{suffix}",
                data=image_bytes(image_format),
            )
            self.assertEqual(response.status_code, 201)
            self.client.delete(
                f"/v1/interactive/sessions/{response.json()['session_id']}",
                headers={"X-API-Key": API_KEY},
            )

    def test_create_validation_and_authentication(self):
        unauthorized = self.client.post("/v1/interactive/sessions")
        self.assertEqual(unauthorized.status_code, 401)
        mismatch = self._create(width=20)
        self.assertEqual(mismatch.status_code, 422)
        self.assertEqual(mismatch.json()["error"]["code"], "image_size_mismatch")
        broken = self._create(data=b"not an image")
        self.assertEqual(broken.status_code, 415)
        unsupported = self._create(name="tile.bmp")
        self.assertEqual(unsupported.status_code, 415)

    def test_create_protocol_identity_and_capacity_validation(self):
        body = {
            "protocol_version": "1.0",
            "request_id": "json-id",
            "coordinate_space": "original_image_pixel",
            "image": {"file_name": "tile.png", "width": 16, "height": 16},
        }
        response = self.client.post(
            "/v1/interactive/sessions",
            headers={"X-API-Key": API_KEY, "X-Request-ID": "json-id"},
            files={
                "image": ("tile.png", image_bytes(), "image/png"),
                "request": (None, json.dumps(body), "application/json"),
            },
        )
        self.assertEqual(response.json()["error"]["code"], "invalid_protocol_version")
        mismatch = self._create(request_id="matched")
        self.assertEqual(mismatch.status_code, 201)
        second = self._create(request_id="second")
        self.assertEqual(second.status_code, 201)
        limited = self._create(request_id="third")
        self.assertEqual(limited.status_code, 429)
        self.assertEqual(limited.json()["error"]["code"], "session_limit_reached")

    def test_two_rounds_accumulate_points_and_restore_logits(self):
        session_id = self._create().json()["session_id"]
        first_body = self._refine_body(session_id)
        first = self._refine(first_body)
        self.assertEqual(first.status_code, 200)
        token = first.json()["interaction_token"]
        self.assertTrue(token)
        self.assertIsNone(self.predictor.predict_calls[0]["previous_logits"])

        second_body = self._refine_body(session_id, revision=2, token=token)
        second_body["positive_points"].append([7, 7])
        second_body["negative_points"].append([13, 13])
        second = self._refine(second_body)
        self.assertEqual(second.status_code, 200)
        self.assertNotEqual(second.json()["interaction_token"], token)
        self.assertEqual(self.predictor.predict_calls[1]["positive_points"], ((5.0, 5.0), (7.0, 7.0)))
        self.assertEqual(self.predictor.predict_calls[1]["negative_points"], ((13.0, 13.0),))
        self.assertIsNotNone(self.predictor.predict_calls[1]["previous_logits"])
        self.assertEqual(second.json()["polygon"]["bbox_xywh"], [3.0, 3.0, 7.0, 7.0])

    def test_empty_token_recomputes_after_prompt_removal(self):
        session_id = self._create().json()["session_id"]
        first = self._refine(self._refine_body(session_id)).json()
        reset_body = self._refine_body(session_id, revision=2, token="")
        response = self._refine(reset_body)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.predictor.predict_calls[-1]["previous_logits"])
        self.assertNotEqual(response.json()["interaction_token"], first["interaction_token"])

    def test_failure_does_not_advance_revision(self):
        session_id = self._create().json()["session_id"]
        first = self._refine(self._refine_body(session_id)).json()
        second_body = self._refine_body(session_id, revision=2, token=first["interaction_token"])
        second_body["positive_points"].append([7, 7])
        self.predictor.fail_next = True
        failed = self._refine(second_body)
        self.assertEqual(failed.status_code, 500)
        retried = self._refine(second_body)
        self.assertEqual(retried.status_code, 200)

    def test_geometry_token_and_session_errors(self):
        session_id = self._create().json()["session_id"]
        body = self._refine_body(session_id)
        body["positive_points"] = []
        self.assertEqual(self._refine(body).json()["error"]["code"], "invalid_positive_points")
        body = self._refine_body(session_id)
        body["bbox_xyxy"] = [5, 5, 2, 2]
        self.assertEqual(self._refine(body).json()["error"]["code"], "invalid_bbox")
        body = self._refine_body(session_id)
        body["original_polygon"] = [[1, 1], [2, 2]]
        self.assertEqual(self._refine(body).json()["error"]["code"], "invalid_polygon")
        body = self._refine_body(session_id, revision=2, token="wrong-token")
        response = self._refine(body)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "interaction_token_invalid")

    def test_request_and_session_identity_errors(self):
        session_id = self._create().json()["session_id"]
        body = self._refine_body(session_id)
        response = self.client.post(
            f"/v1/interactive/sessions/{session_id}/refine",
            headers={"X-API-Key": API_KEY, "X-Request-ID": "different"},
            json=body,
        )
        self.assertEqual(response.json()["error"]["code"], "request_id_mismatch")
        body["session_id"] = "another-session"
        response = self.client.post(
            f"/v1/interactive/sessions/{session_id}/refine",
            headers={"X-API-Key": API_KEY, "X-Request-ID": body["request_id"]},
            json=body,
        )
        self.assertEqual(response.status_code, 400)

    def test_token_cannot_cross_sessions(self):
        first_session = self._create(request_id="create-a").json()["session_id"]
        second_session = self._create(request_id="create-b").json()["session_id"]
        first = self._refine(self._refine_body(first_session)).json()
        body = self._refine_body(second_session, revision=2, token=first["interaction_token"])
        response = self._refine(body)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "interaction_token_invalid")

    def test_delete_is_idempotent_and_refine_then_returns_not_found(self):
        session_id = self._create().json()["session_id"]
        url = f"/v1/interactive/sessions/{session_id}"
        first = self.client.delete(url, headers={"X-API-Key": API_KEY, "X-Request-ID": ""})
        second = self.client.delete(url, headers={"X-API-Key": API_KEY})
        self.assertEqual(first.status_code, 204)
        self.assertEqual(second.status_code, 204)
        response = self._refine(self._refine_body(session_id))
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "session_not_found")


if __name__ == "__main__":
    unittest.main()
