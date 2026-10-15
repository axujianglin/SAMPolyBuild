"""Model-free tests for the Docker Auto inference service core."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from services.auto_inference_service import (
    AutoInferenceService,
    AutoServiceConfig,
    AutoServiceError,
)


class AutoInferenceServiceTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "auto.py"
        self.checkpoint_path = self.root / "auto.pth"
        self.config_path.write_text("task_name='test'", encoding="utf-8")
        self.checkpoint_path.write_bytes(b"mock-checkpoint")
        self.runner_calls = []

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _service(self, runner=None, cuda_available=True, max_upload_bytes=1024 * 1024):
        return AutoInferenceService(
            AutoServiceConfig(
                model_config=self.config_path,
                checkpoint=self.checkpoint_path,
                runtime_root=self.root / "runtime",
                device="cuda",
                gpu=0,
                batch_size=1,
                num_workers=0,
                default_score_threshold=0.1,
                max_upload_bytes=max_upload_bytes,
            ),
            infer_runner=runner or self._fake_runner,
            cuda_probe=lambda: cuda_available,
        )

    def _fake_runner(self, **kwargs):
        self.runner_calls.append(kwargs)
        result_dir = Path(kwargs["result_dir"])
        result_dir.mkdir(parents=True, exist_ok=True)
        image_path = next(Path(kwargs["img_dir"]).glob(f"*{kwargs['img_suffix']}"))
        results_path = result_dir / "results.json"
        results_path.write_text(
            json.dumps([{
                "image_id": image_path.stem,
                "score": 0.92,
                "score_cls": 0.95,
                "bbox": [2, 3, 10, 8],
                "segmentation": [[2, 3, 12, 3, 12, 11, 2, 11]],
                "category_id": 1,
            }]),
            encoding="utf-8",
        )
        return {
            "results_path": str(results_path),
            "results_mask_path": str(result_dir / "results_mask.json"),
        }

    def _write_image(self, parent: Path, name="image001.tiff") -> Path:
        image_path = parent / "input" / name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), color=(120, 130, 140)).save(image_path)
        return image_path

    def test_health_reports_readiness_without_running_model(self):
        service = self._service()

        health = service.health()

        self.assertEqual(health["status"], "ready")
        self.assertTrue(health["cuda_available"])
        self.assertFalse(health["busy"])
        self.assertEqual(self.runner_calls, [])

    def test_infer_calls_existing_runner_and_returns_http_contract(self):
        job_root = self.root / "runtime" / "request_test"
        image_path = self._write_image(job_root)
        service = self._service()

        response = service.infer(
            image_path,
            request_id="request-test",
            original_file_name="image001.tiff",
            score_threshold=0.5,
        )

        self.assertTrue(response["success"])
        self.assertEqual(response["coordinate_space"], "original_image_pixel")
        self.assertEqual(response["image"]["width"], 32)
        self.assertEqual(response["image"]["height"], 24)
        self.assertEqual(response["summary"]["polygon_count"], 1)
        self.assertEqual(response["polygons"][0]["id"], "image001_poly_0001")
        self.assertEqual(len(self.runner_calls), 1)
        self.assertTrue(self.runner_calls[0]["replace_predict_loader"])
        self.assertEqual(self.runner_calls[0]["device"], "cuda")

    def test_missing_checkpoint_and_cuda_have_explicit_errors(self):
        self.checkpoint_path.unlink()
        with self.assertRaisesRegex(AutoServiceError, "checkpoint not found"):
            self._service().infer(
                self._write_image(self.root / "job1"),
                request_id="one",
                original_file_name="image001.tiff",
            )

        self.checkpoint_path.write_bytes(b"mock-checkpoint")
        with self.assertRaisesRegex(AutoServiceError, "CUDA is unavailable"):
            self._service(cuda_available=False).infer(
                self._write_image(self.root / "job2"),
                request_id="two",
                original_file_name="image001.tiff",
            )

    def test_invalid_threshold_does_not_call_runner(self):
        service = self._service()
        with self.assertRaisesRegex(AutoServiceError, "score_threshold"):
            service.infer(
                self._write_image(self.root / "job"),
                request_id="invalid",
                original_file_name="image001.tiff",
                score_threshold=1.5,
            )
        self.assertEqual(self.runner_calls, [])


if __name__ == "__main__":
    unittest.main()
