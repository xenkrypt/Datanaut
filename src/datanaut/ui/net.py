"""The one place the app talks to the internet: a small, restricted JSON fetch for Open-Meteo.

It is injected into `core.weather.fetch_open_meteo`, so the core never opens a connection and
tests use a fake. Only the two public Open-Meteo hosts over HTTPS are allowed. Nothing about
the user's data is sent except the coordinates and dates in the request URL.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from datanaut.errors import WeatherError

ALLOWED_HOSTS = frozenset({"archive-api.open-meteo.com", "api.open-meteo.com"})
TIMEOUT_SECONDS = 20
MAX_BYTES = 5_000_000


def fetch_json(url: str, timeout: float = TIMEOUT_SECONDS) -> Any:
    """GET `url` and return the parsed JSON. Raises WeatherError with a friendly message."""
    parts = urlparse(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise WeatherError("Weather can only be requested from Open-Meteo.")
    request = urllib.request.Request(url, headers={"User-Agent": "datanaut"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = response.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise WeatherError(
            f"The weather service refused the request (HTTP {exc.code}). "
            "Check the coordinates and dates, or upload a weather file instead."
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise WeatherError(
            "The weather service could not be reached. You may be offline. "
            "You can upload a weather CSV file instead."
        ) from exc
    if len(body) > MAX_BYTES:
        raise WeatherError("The weather reply was unexpectedly large and was rejected.")
    try:
        return json.loads(body)
    except ValueError as exc:
        raise WeatherError("The weather service sent a reply that could not be read.") from exc
