"""HTTP contract tests for the Docker Auto API using a mock model runner."""

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


API_TEST_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("fastapi", "httpx", "multipart")
)


@unittest.skipUnless(API_TEST_DEPS_AVAILABLE, "FastAPI test dependencies are not installed.")
class AutoApiContractTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from api.app import create_app
        from services.auto_inference_service import AutoInferenceService, AutoServiceConfig

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        config_path = self.root / "auto.py"
        checkpoint_path = self.root / "auto.pth"
        config_path.write_text("task_name='test'", encoding="utf-8")
        checkpoint_path.write_bytes(b"mock-checkpoint")
        self.service = AutoInferenceService(
            AutoServiceConfig(
                model_config=config_path,
                checkpoint=checkpoint_path,
                runtime_root=self.root / "runtime",
                device="cuda",
                num_workers=0,
                max_upload_bytes=1024 * 1024,
            ),
            infer_runner=self._fake_runner,
            cuda_probe=lambda: True,
        )
        # Explicitly disable authentication for the existing local contract
        # tests, independent of the developer machine's environment.
        self.client = TestClient(create_app(self.service, api_key=""))

    def tearDown(self):
        self.client.close()
        self.temporary_directory.cleanup()

    @staticmethod
    def _fake_runner(**kwargs):
        result_dir = Path(kwargs["result_dir"])
        result_dir.mkdir(parents=True, exist_ok=True)
        image_path = next(Path(kwargs["img_dir"]).glob(f"*{kwargs['img_suffix']}"))
        results_path = result_dir / "results.json"
        results_path.write_text(
            json.dumps([{
                "image_id": image_path.stem,
                "score": 0.9,
                "score_cls": 0.95,
                "bbox": [1, 2, 10, 8],
                "segmentation": [[1, 2, 11, 2, 11, 10, 1, 10]],
            }]),
            encoding="utf-8",
        )
        return {"results_path": str(results_path)}

    @staticmethod
    def _image_bytes() -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (32, 24), color=(30, 40, 50)).save(buffer, format="TIFF")
        return buffer.getvalue()

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")
        self.assertEqual(response.json()["service"], "sampoly-auto-inference")

    def test_infer_endpoint_returns_original_pixel_polygons_and_cleans_job(self):
        response = self.client.post(
            "/v1/auto/infer",
            files={"image": ("image001.tiff", self._image_bytes(), "image/tiff")},
            data={"score_threshold": "0.1"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["coordinate_space"], "original_image_pixel")
        self.assertEqual(payload["summary"]["polygon_count"], 1)
        self.assertEqual(payload["polygons"][0]["points"][0], [1.0, 2.0])
        self.assertEqual(list((self.root / "runtime").iterdir()), [])

    def test_rejects_unsupported_and_empty_uploads(self):
        unsupported = self.client.post(
            "/v1/auto/infer",
            files={"image": ("image.bmp", b"data", "image/bmp")},
        )
        self.assertEqual(unsupported.status_code, 400)
        self.assertEqual(unsupported.json()["error"]["code"], "unsupported_image_format")

        empty = self.client.post(
            "/v1/auto/infer",
            files={"image": ("image.tiff", b"", "image/tiff")},
        )
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(empty.json()["error"]["code"], "empty_upload")

    def test_request_validation_uses_stable_error_envelope(self):
        response = self.client.post(
            "/v1/auto/infer",
            files={"image": ("image001.tiff", self._image_bytes(), "image/tiff")},
            data={"score_threshold": "not-a-number"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertFalse(response.json()["success"])
        self.assertEqual(
            response.json()["error"]["code"], "request_validation_error"
        )

    def test_configured_api_key_protects_inference_but_not_health(self):
        from fastapi.testclient import TestClient

        from api.app import create_app

        with TestClient(create_app(self.service, api_key="test-secret")) as client:
            health = client.get("/health")
            unauthorized = client.post(
                "/v1/auto/infer",
                files={"image": ("image001.tiff", self._image_bytes(), "image/tiff")},
            )
            authorized = client.post(
                "/v1/auto/infer",
                headers={"X-API-Key": "test-secret"},
                files={"image": ("image001.tiff", self._image_bytes(), "image/tiff")},
            )

        self.assertEqual(health.status_code, 200)
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(unauthorized.json()["error"]["code"], "unauthorized")
        self.assertEqual(authorized.status_code, 200, authorized.text)


if __name__ == "__main__":
    unittest.main()
