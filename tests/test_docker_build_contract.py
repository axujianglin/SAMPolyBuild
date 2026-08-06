"""Static contract tests for Docker-only native extension prerequisites."""

import unittest
from pathlib import Path


class DockerBuildContractTest(unittest.TestCase):
    """Prevent regressions in the vendored pycocotools build layer."""

    @classmethod
    def setUpClass(cls):
        cls.repository_root = Path(__file__).resolve().parents[1]
        cls.dockerfile = (
            cls.repository_root / "docker" / "Dockerfile"
        ).read_text(encoding="utf-8")
        cls.api_requirements = (
            cls.repository_root / "docker" / "requirements-api.txt"
        ).read_text(encoding="utf-8")

    def test_complete_pycocotools_tree_is_available_to_native_build(self):
        """The C extension needs PythonAPI and its sibling common directory."""
        self.assertTrue(
            (self.repository_root / "pycocotools" / "common" / "maskApi.c").is_file()
        )
        self.assertIn("COPY pycocotools /tmp/pycocotools", self.dockerfile)
        self.assertNotIn(
            "COPY pycocotools/PythonAPI /tmp/pycocotools", self.dockerfile
        )

    def test_legacy_build_dependencies_are_explicit(self):
        """NumPy visibility and Cython availability must be deterministic."""
        self.assertIn("Cython==0.29.36", self.dockerfile)
        self.assertIn("--no-build-isolation", self.dockerfile)
        self.assertIn("/tmp/pycocotools/PythonAPI", self.dockerfile)

    def test_runtime_native_and_cloud_compatibility_is_build_checked(self):
        """The image must fail early if verified runtime imports regress."""
        self.assertIn("libgl1", self.dockerfile)
        self.assertIn("lightning-cloud==0.5.31", self.api_requirements)
        self.assertIn("import cv2", self.dockerfile)
        self.assertIn(
            "from lightning_cloud.openapi import AppinstancesIdBody",
            self.dockerfile,
        )
        self.assertIn("import pycocotools._mask", self.dockerfile)


if __name__ == "__main__":
    unittest.main()
