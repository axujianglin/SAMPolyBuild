from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window
except ImportError as exc:  # Keep CLI help importable when the optional dependency is absent.
    rasterio = None
    Affine = None
    Window = None
    _RASTERIO_IMPORT_ERROR = exc
else:
    _RASTERIO_IMPORT_ERROR = None


class ImageTileReader:
    """Windowed raster access for large geospatial images."""

    def __init__(self, path, bands: Sequence[int] = (1, 2, 3)):
        if rasterio is None:
            raise ImportError(
                "ImageTileReader requires rasterio. Install it in the runtime "
                "environment with: conda install -c conda-forge rasterio"
            ) from _RASTERIO_IMPORT_ERROR

        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Raster image not found: {self.path}")

        self.bands = tuple(int(band) for band in bands)
        if not self.bands:
            raise ValueError("At least one raster band must be selected.")

        self._dataset = rasterio.open(self.path, mode="r")
        try:
            self.width = int(self._dataset.width)
            self.height = int(self._dataset.height)
            self.count = int(self._dataset.count)
            invalid_bands = [band for band in self.bands if band < 1 or band > self.count]
            if invalid_bands:
                raise ValueError(
                    f"Selected bands {invalid_bands} are outside the valid range 1..{self.count}."
                )

            selected_dtypes = tuple(self._dataset.dtypes[band - 1] for band in self.bands)
            if len(set(selected_dtypes)) != 1:
                raise ValueError(
                    "Selected bands must share one dtype, got "
                    f"{selected_dtypes}."
                )
            self.dtype = selected_dtypes[0]
            self.crs = self._dataset.crs
            self.transform = self._dataset.transform
            self.band_nodata = tuple(
                self._dataset.nodatavals[band - 1] for band in self.bands
            )
            first_nodata = self.band_nodata[0]
            if not all(self._nodata_equal(first_nodata, value) for value in self.band_nodata[1:]):
                raise ValueError(
                    "Selected bands must share one nodata value so it can be preserved, got "
                    f"{self.band_nodata}."
                )
            self.nodata = first_nodata
        except Exception:
            self.close()
            raise

    @staticmethod
    def _nodata_equal(left, right):
        if left is None or right is None:
            return left is right
        return bool((np.isnan(left) and np.isnan(right)) or left == right)

    @property
    def closed(self) -> bool:
        return self._dataset is None or self._dataset.closed

    def _require_open(self):
        if self.closed:
            raise RuntimeError("ImageTileReader is closed.")

    def _validate_window(self, x_offset, y_offset, width, height) -> Tuple[int, int, int, int]:
        values = (x_offset, y_offset, width, height)
        if any(int(value) != value for value in values):
            raise ValueError("Tile offsets and dimensions must be integers.")
        x_offset, y_offset, width, height = (int(value) for value in values)
        if x_offset < 0 or y_offset < 0:
            raise ValueError("Tile offsets must be non-negative.")
        if width <= 0 or height <= 0:
            raise ValueError("Tile width and height must be positive.")
        if x_offset + width > self.width or y_offset + height > self.height:
            raise ValueError(
                "Tile window exceeds source bounds: "
                f"window=({x_offset}, {y_offset}, {width}, {height}), "
                f"source=({self.width}, {self.height})."
            )
        return x_offset, y_offset, width, height

    def read_tile(self, x_offset, y_offset, width, height) -> np.ndarray:
        """Read one bounded window and return it in HWC order."""
        self._require_open()
        x_offset, y_offset, width, height = self._validate_window(
            x_offset, y_offset, width, height
        )
        window = Window(x_offset, y_offset, width, height)
        tile_chw = self._dataset.read(indexes=self.bands, window=window)
        expected_shape = (len(self.bands), height, width)
        if tile_chw.shape != expected_shape:
            raise RuntimeError(
                f"Window read returned shape {tile_chw.shape}, expected {expected_shape}."
            )
        return np.moveaxis(tile_chw, 0, -1)

    def tile_transform(self, x_offset, y_offset):
        """Return the affine transform for a tile origin."""
        self._require_open()
        if int(x_offset) != x_offset or int(y_offset) != y_offset:
            raise ValueError("Tile offsets must be integers.")
        x_offset, y_offset = int(x_offset), int(y_offset)
        if not (0 <= x_offset < self.width and 0 <= y_offset < self.height):
            raise ValueError(
                f"Tile origin ({x_offset}, {y_offset}) is outside source bounds."
            )
        return self.transform * Affine.translation(x_offset, y_offset)

    def tile_profile(self, x_offset, y_offset, width, height) -> dict:
        """Build a Rasterio profile for writing one GeoTIFF tile."""
        self._require_open()
        x_offset, y_offset, width, height = self._validate_window(
            x_offset, y_offset, width, height
        )
        return {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": len(self.bands),
            "dtype": self.dtype,
            "crs": self.crs,
            "transform": self.tile_transform(x_offset, y_offset),
            "nodata": self.nodata,
        }

    def close(self):
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
