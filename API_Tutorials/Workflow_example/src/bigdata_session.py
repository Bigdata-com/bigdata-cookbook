"""Session wrapper for Bigdata.com API key authentication."""

from __future__ import annotations

import os
from typing import Union

import requests

AUTH_MODE_API_KEY = "api_key"


class BigDataSession:
    """Small wrapper around `requests.Session` with API key headers."""

    DEFAULT_BASE_URL = "https://api.bigdata.com"
    KG_ENTITIES_PATH = "/v1/knowledge-graph/entities/id"

    def __init__(
        self,
        api_key: str | None = None,
        api_base_url: str | None = None,
        participant_id: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("BIGDATA_API_KEY")
        self.api_base_url = api_base_url or os.getenv(
            "BIGDATA_API_BASE_URL", self.DEFAULT_BASE_URL
        )
        self.participant_id = participant_id or os.getenv("PARTICIPANT_ID")
        self.auth_mode = AUTH_MODE_API_KEY

        if not self.api_key:
            raise ValueError("Set BIGDATA_API_KEY in .env or pass api_key explicitly")

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "X-API-KEY": self.api_key,
            }
        )
        if self.participant_id:
            self.session.headers["X-PARTICIPANT-ID"] = self.participant_id

    def _build_url(self, endpoint: str) -> str:
        path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        return f"{self.api_base_url.rstrip('/')}{path}"

    def post(self, endpoint: str, json: dict | None = None, **kwargs) -> requests.Response:
        url = self._build_url(endpoint)
        return self.session.post(url, json=json, **kwargs)

    def get(self, endpoint: str, **kwargs) -> requests.Response:
        url = self._build_url(endpoint)
        return self.session.get(url, **kwargs)

    @property
    def search_endpoint(self) -> str:
        return self._build_url("/v1/search")

    @property
    def comention_endpoint(self) -> str:
        return self._build_url("/v1/search/co-mentions/entities")

    @property
    def volume_endpoint(self) -> str:
        return self._build_url("/v1/search/volume")

    @property
    def kg_companies_endpoint(self) -> str:
        return self._build_url("/v1/knowledge-graph/companies")

    @property
    def kg_entities_endpoint(self) -> str:
        return self._build_url(self.KG_ENTITIES_PATH)

    def get_entities_by_ids(self, entity_ids: Union[list[str], str]) -> requests.Response:
        ids = [entity_ids] if isinstance(entity_ids, str) else list(entity_ids)
        return self.post(self.KG_ENTITIES_PATH, json={"values": ids})

    def __repr__(self) -> str:
        return f"BigDataSession(auth_mode='api_key', api_base_url='{self.api_base_url}')"
