import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.image_tile_reader import ImageTileReader


def parse_bands(value):
    try:
        bands = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--bands must be comma-separated integers.") from exc
    if not bands:
        raise argparse.ArgumentTypeError("--bands must select at least one band.")
    return bands


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a large geospatial raster into window-read GeoTIFF tiles."
    )
    parser.add_argument("--input", required=True, help="Input GeoTIFF path.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--tile_size", type=int, default=1024, help="Tile side length in pixels.")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between adjacent tiles.")
    parser.add_argument("--prefix", default=None, help="Tile name prefix; defaults to input stem.")
    parser.add_argument("--bands", type=parse_bands, default=(1, 2, 3), help="Bands, e.g. 1,2,3.")
    parser.add_argument("--skip_empty", action="store_true", help="Skip mostly nodata/black/white tiles.")
    parser.add_argument(
        "--empty_threshold",
        type=float,
        default=0.98,
        help="Pixel ratio above which a tile is considered empty.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tiles and manifest.")
    return parser.parse_args()


def axis_offsets(length, stride):
    return list(range(0, length, stride))


def iter_tile_windows(width, height, tile_size, stride):
    for y_offset in axis_offsets(height, stride):
        tile_height = min(tile_size, height - y_offset)
        for x_offset in axis_offsets(width, stride):
            tile_width = min(tile_size, width - x_offset)
            yield x_offset, y_offset, tile_width, tile_height


def affine_to_list(transform):
    return [
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    ]


def crs_to_string(crs):
    if crs is None:
        return None
    epsg = crs.to_epsg()
    if epsg is None:
        epsg = crs.to_epsg(confidence_threshold=0)
    return f"EPSG:{epsg}" if epsg is not None else crs.to_wkt()


def empty_tile_reason(tile, nodata, threshold):
    pixel_count = tile.shape[0] * tile.shape[1]
    if pixel_count == 0:
        return "empty_window"

    if nodata is not None:
        if isinstance(nodata, float) and math.isnan(nodata):
            nodata_pixels = np.all(np.isnan(tile), axis=2)
        else:
            nodata_pixels = np.all(tile == nodata, axis=2)
        if np.count_nonzero(nodata_pixels) / pixel_count > threshold:
            return "nodata"

    zero_pixels = np.all(tile == 0, axis=2)
    if np.count_nonzero(zero_pixels) / pixel_count > threshold:
        return "all_zero"

    white_pixels = np.all(tile == 255, axis=2)
    if np.count_nonzero(white_pixels) / pixel_count > threshold:
        return "all_255"
    return None


def remove_existing_prefix_tiles(tiles_dir, prefix):
    name_prefix = f"{prefix}_x"
    for tile_path in tiles_dir.glob("*.tif"):
        if tile_path.name.startswith(name_prefix):
            tile_path.unlink()


def write_tile(tile_path, tile, profile):
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "tile_large_image.py requires rasterio. Install it with: "
            "conda install -c conda-forge rasterio"
        ) from exc

    write_profile = dict(profile)
    write_profile.update(compress="deflate")
    with rasterio.open(tile_path, mode="w", **write_profile) as dataset:
        dataset.write(np.moveaxis(tile, -1, 0))


def validate_args(args):
    if args.tile_size <= 0:
        raise ValueError("--tile_size must be positive.")
    if args.overlap < 0 or args.overlap >= args.tile_size:
        raise ValueError("--overlap must satisfy 0 <= overlap < tile_size.")
    if not 0.0 <= args.empty_threshold <= 1.0:
        raise ValueError("--empty_threshold must be between 0 and 1.")
    if not args.prefix:
        args.prefix = Path(args.input).stem
    if Path(args.prefix).name != args.prefix or args.prefix in (".", ".."):
        raise ValueError("--prefix must be a file-name-safe value without path separators.")


def run(args):
    validate_args(args)
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    tiles_dir = output_dir / "tiles"
    manifest_path = output_dir / "tiles_manifest.json"
    stride = args.tile_size - args.overlap

    existing_tiles = [
        path for path in tiles_dir.glob("*.tif")
        if path.name.startswith(f"{args.prefix}_x")
    ] if tiles_dir.exists() else []
    if not args.overwrite and (manifest_path.exists() or existing_tiles):
        raise FileExistsError(
            f"Output already exists in {output_dir}. Use --overwrite to replace it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        remove_existing_prefix_tiles(tiles_dir, args.prefix)

    tiles = []
    skipped_count = 0
    edge_tile_count = 0
    with ImageTileReader(input_path, bands=args.bands) as reader:
        planned_windows = list(
            iter_tile_windows(reader.width, reader.height, args.tile_size, stride)
        )
        if not planned_windows:
            raise RuntimeError("No tile windows were generated.")
        right_coverage = max(x + width for x, _, width, _ in planned_windows)
        bottom_coverage = max(y + height for _, y, _, height in planned_windows)
        if right_coverage != reader.width or bottom_coverage != reader.height:
            raise RuntimeError(
                "Tile grid does not cover the source bounds: "
                f"coverage=({right_coverage}, {bottom_coverage}), "
                f"source=({reader.width}, {reader.height})."
            )

        source_crs = crs_to_string(reader.crs)
        source = {
            "path": str(input_path),
            "image_id": input_path.stem,
            "width": reader.width,
            "height": reader.height,
            "count": reader.count,
            "bands": list(reader.bands),
            "dtype": reader.dtype,
            "crs": source_crs,
            "transform": affine_to_list(reader.transform),
            "nodata": reader.nodata,
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "stride": stride,
        }

        print(f"source size: {reader.width} x {reader.height}")
        print(f"dtype: {reader.dtype}")
        print(f"crs: {source_crs}")
        print(f"tile_size: {args.tile_size}")
        print(f"overlap: {args.overlap}")
        print(f"stride: {stride}")

        for x_offset, y_offset, width, height in planned_windows:
            tile = reader.read_tile(x_offset, y_offset, width, height)
            if args.skip_empty:
                reason = empty_tile_reason(tile, reader.nodata, args.empty_threshold)
                if reason is not None:
                    skipped_count += 1
                    continue

            tile_id = f"{args.prefix}_x{x_offset:06d}_y{y_offset:06d}"
            file_name = f"{tile_id}.tif"
            tile_path = tiles_dir / file_name
            profile = reader.tile_profile(x_offset, y_offset, width, height)
            write_tile(tile_path, tile, profile)
            is_edge = width < args.tile_size or height < args.tile_size
            edge_tile_count += int(is_edge)
            tiles.append({
                "tile_id": tile_id,
                "file_name": file_name,
                "path": (Path("tiles") / file_name).as_posix(),
                "x_offset": x_offset,
                "y_offset": y_offset,
                "width": width,
                "height": height,
                "is_edge": is_edge,
                "crs": source_crs,
                "transform": affine_to_list(profile["transform"]),
            })

    if not tiles:
        raise RuntimeError("No tiles were written; check --skip_empty and --empty_threshold.")

    actual_tiles = [
        path for path in tiles_dir.glob("*.tif")
        if path.name.startswith(f"{args.prefix}_x")
    ]
    if len(actual_tiles) != len(tiles):
        raise RuntimeError(
            f"Manifest has {len(tiles)} tiles but found {len(actual_tiles)} tile files."
        )

    manifest = {"source": source, "tiles": tiles}
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    with temporary_manifest.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    temporary_manifest.replace(manifest_path)

    print(f"tile count: {len(tiles)}")
    print(f"edge tile count: {edge_tile_count}")
    print(f"skipped empty tile count: {skipped_count}")
    print(f"manifest path: {manifest_path}")
    return manifest


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"Tiling failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
