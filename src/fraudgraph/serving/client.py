"""The dashboard's view of the service: over HTTP (docker compose) or in-process (the public demo).

Both return the same JSON-shaped dicts and raise the same errors, so the dashboard does not
care which one it has (tests/test_api.py checks they agree).
"""

import json
from collections.abc import Callable

from fraudgraph.serving.service import InvalidRequest, NotFound, ServingService, Unavailable

ERRORS = {404: NotFound, 422: InvalidRequest, 501: Unavailable}


class LocalClient:
    def __init__(self, service: ServingService):
        self.service = service

    def _call(self, name: str, *args):
        # A JSON round trip gives exactly the types an HTTP client would see.
        return json.loads(json.dumps(getattr(self.service, name)(*args)))

    def meta(self):
        return self._call("meta")

    def steps(self):
        return self._call("steps")

    def transaction(self, tx_id: int):
        return self._call("transaction", tx_id)

    def alerts(self, time_step: int, k: int):
        return self._call("alerts", time_step, k)

    def neighbourhood(self, tx_id: int, hops: int):
        return self._call("neighbourhood", tx_id, hops)

    def paths(self, source: int, target: int):
        return self._call("paths", source, target)

    def communities(self, time_step: int, min_size: int = 5, top: int = 10):
        return self._call("communities", time_step, min_size, top)

    def fan_in(self, time_step: int, min_flagged: int = 2, top: int = 20):
        return self._call("fan_in", time_step, min_flagged, top)


class ApiClient:
    def __init__(self, base_url: str, get: Callable | None = None, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        if get is None:
            import requests  # only the dashboard image has it

            def get(url, params=None):
                return requests.get(url, params=params, timeout=timeout)

        self._get = get

    def _call(self, path: str, **params):
        r = self._get(f"{self.base_url}{path}", params=params or None)
        if r.status_code in ERRORS:
            raise ERRORS[r.status_code](r.json().get("detail", ""))
        r.raise_for_status()
        return r.json()

    def meta(self):
        return self._call("/meta")

    def steps(self):
        return self._call("/steps")

    def transaction(self, tx_id: int):
        return self._call(f"/transactions/{tx_id}")

    def alerts(self, time_step: int, k: int):
        return self._call("/alerts", time_step=time_step, k=k)

    def neighbourhood(self, tx_id: int, hops: int):
        return self._call(f"/transactions/{tx_id}/neighbourhood", hops=hops)

    def paths(self, source: int, target: int):
        return self._call("/paths", source=source, target=target)

    def communities(self, time_step: int, min_size: int = 5, top: int = 10):
        return self._call("/communities", time_step=time_step, min_size=min_size, top=top)

    def fan_in(self, time_step: int, min_flagged: int = 2, top: int = 20):
        return self._call("/fan-in", time_step=time_step, min_flagged=min_flagged, top=top)
