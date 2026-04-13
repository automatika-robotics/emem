from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np

log = logging.getLogger("harness")


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """POST JSON and return the parsed response.

    :param url: Target URL.
    :param payload: JSON-serialisable request body.
    :param headers: Extra HTTP headers.
    :param timeout: Request timeout in seconds.
    :returns: Parsed JSON response.
    :raises RuntimeError: On network, timeout or JSON parse failure.
    """
    data = json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Service unavailable at {url}: {e}") from e
    except TimeoutError as e:
        raise RuntimeError(f"Request timed out ({timeout}s): {url}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from {url}: {e}") from e


def post_json_with_retry(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: int = 120,
    max_retries: int = 3,
    backoff: float = 2.0,
) -> dict[str, Any]:
    """Like :func:`post_json` but retries on HTTP 429 with exponential backoff.

    :param url: Target URL.
    :param payload: JSON-serialisable request body.
    :param headers: Extra HTTP headers.
    :param timeout: Per-request timeout in seconds.
    :param max_retries: Maximum number of attempts.
    :param backoff: Base delay in seconds (doubled each retry).
    :returns: Parsed JSON response.
    :raises RuntimeError: After all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            return post_json(url, payload, headers=headers, timeout=timeout)
        except RuntimeError as e:
            if "HTTP 429" in str(e) and attempt < max_retries - 1:
                wait = backoff * (2**attempt)
                log.warning(
                    "Rate limited (429), retrying in %.0fs (%d/%d)",
                    wait,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Unreachable")  # pragma: no cover


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models.

    Also strips orphaned ``</think>`` tags and trailing ``<think>`` blocks
    that were never closed (model cut off mid-thought).

    :param text: Raw model output.
    :returns: Text with thinking blocks stripped.
    """
    # Full <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Unclosed <think> at end of output
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    # Orphaned </think> tag
    text = text.replace("</think>", "")
    return text.strip()


def encode_image_b64(image: np.ndarray) -> str:
    """Encode an RGB numpy array as a base64 PNG string.

    :param image: ``(H, W, 3)`` uint8 numpy array.
    :returns: Base64-encoded image string.
    """
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
