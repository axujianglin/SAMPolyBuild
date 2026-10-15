"""Optional real-service three-round verification for protocol 2.0."""

from __future__ import annotations

import argparse
import json
import mimetypes
import uuid
from pathlib import Path

import httpx
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--bbox", type=json.loads, required=True, help="JSON [x1,y1,x2,y2]")
    parser.add_argument("--positive", type=json.loads, required=True, help="JSON with at least two points")
    parser.add_argument("--negative", type=json.loads, required=True, help="JSON with at least one point")
    parser.add_argument("--polygon", type=json.loads, required=True, help="Original polygon JSON")
    return parser.parse_args()


def checked(response: httpx.Response, expected: int) -> dict:
    if response.status_code != expected:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json() if response.content else {}


def main() -> None:
    args = parse_args()
    if len(args.positive) < 2 or len(args.negative) < 1:
        raise ValueError("Verification needs at least two positive points and one negative point.")
    image_path = args.image.expanduser().resolve()
    with Image.open(image_path) as image:
        width, height = image.size
    headers = {"X-API-Key": args.api_key}
    create_id = str(uuid.uuid4())
    create_request = {
        "protocol_version": "2.0",
        "request_id": create_id,
        "coordinate_space": "original_image_pixel",
        "image": {"file_name": image_path.name, "width": width, "height": height},
    }
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=120.0) as client:
        with image_path.open("rb") as stream:
            created = checked(
                client.post(
                    "/v1/interactive/sessions",
                    headers={**headers, "X-Request-ID": create_id},
                    files={
                        "image": (
                            image_path.name,
                            stream,
                            mimetypes.guess_type(image_path.name)[0] or "application/octet-stream",
                        ),
                        "request": (None, json.dumps(create_request), "application/json"),
                    },
                ),
                201,
            )
        session_id = created["session_id"]
        token = ""
        try:
            rounds = [
                (args.positive[:1], []),
                (args.positive, []),
                (args.positive, args.negative),
            ]
            observed_tokens = []
            for revision, (positive, negative) in enumerate(rounds, start=1):
                request_id = str(uuid.uuid4())
                body = {
                    "protocol_version": "2.0",
                    "request_id": request_id,
                    "session_id": session_id,
                    "polygon_id": "verification-building",
                    "revision": revision,
                    "coordinate_space": "original_image_pixel",
                    "image": {
                        "file_name": image_path.name,
                        "width": width,
                        "height": height,
                    },
                    "bbox_xyxy": args.bbox,
                    "positive_points": positive,
                    "negative_points": negative,
                    "original_polygon": args.polygon,
                    "interaction_token": token,
                }
                result = checked(
                    client.post(
                        f"/v1/interactive/sessions/{session_id}/refine",
                        headers={**headers, "X-Request-ID": request_id},
                        json=body,
                    ),
                    200,
                )
                token = result["interaction_token"]
                observed_tokens.append(token)
                print(
                    f"revision={revision} score={result['polygon']['score']:.4f} "
                    f"points={len(result['polygon']['points'])} token_changed=yes"
                )
            if len(set(observed_tokens)) != 3:
                raise RuntimeError("The server did not return a new token for every round.")
        finally:
            checked(client.delete(f"/v1/interactive/sessions/{session_id}", headers=headers), 204)
        request_id = str(uuid.uuid4())
        deleted_body = {
            "protocol_version": "2.0",
            "request_id": request_id,
            "session_id": session_id,
            "polygon_id": "verification-building",
            "revision": 4,
            "coordinate_space": "original_image_pixel",
            "image": {"file_name": image_path.name, "width": width, "height": height},
            "bbox_xyxy": args.bbox,
            "positive_points": args.positive,
            "negative_points": args.negative,
            "original_polygon": args.polygon,
            "interaction_token": token,
        }
        deleted = client.post(
            f"/v1/interactive/sessions/{session_id}/refine",
            headers={**headers, "X-Request-ID": request_id},
            json=deleted_body,
        )
        if deleted.status_code != 404 or deleted.json()["error"]["code"] != "session_not_found":
            raise RuntimeError(f"Deleted session remained usable: {deleted.text}")
    print("Interactive protocol 2.0 real-service verification passed.")


if __name__ == "__main__":
    main()
