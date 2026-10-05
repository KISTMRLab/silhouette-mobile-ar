"""Fetch a pinned onnxruntime-web (WASM build) into ignored static/vendor/onnxruntime/.

The browser demo loads it from the local server only (no CDN at runtime) to run an
exported compressed U-Net in the page. Run once with internet access.
"""
from pathlib import Path
from urllib.request import urlopen
import argparse

VERSION = "1.22.0"
FILES = ("ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm")


def prepare(destination: str | Path) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = destination / name
        if target.is_file() and target.stat().st_size > 0:
            continue
        with urlopen(f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{VERSION}/dist/{name}", timeout=120) as response:
            target.write_bytes(response.read())
    (destination / "VERSION").write_text(f"onnxruntime-web {VERSION} (MIT licence, Microsoft)\n", encoding="utf-8")
    print(f"onnxruntime-web {VERSION} prepared in {destination}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "static" / "vendor" / "onnxruntime"))
    prepare(parser.parse_args().out)
