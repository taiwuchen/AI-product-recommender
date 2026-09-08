import hashlib
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps


def load_image(url, cache_dir):
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("Product image URL must use HTTP or HTTPS.")
    path = Path(cache_dir) / (hashlib.sha256(url.encode()).hexdigest() + ".jpg")
    if path.exists():
        with Image.open(path) as image:
            return image.convert("RGB")
    with requests.get(url, timeout=(5, 15), stream=True) as response:
        response.raise_for_status()
        chunks = bytearray()
        for chunk in response.iter_content(65536):
            chunks.extend(chunk)
            if len(chunks) > 15 * 1024 * 1024:
                raise ValueError("Image exceeds 15 MB.")
    with Image.open(BytesIO(chunks)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=95)
    return image
