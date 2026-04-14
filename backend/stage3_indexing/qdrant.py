"""Qdrant REST helper for stage-3 indexing."""

from __future__ import annotations

from typing import Any

import httpx


class QdrantRestClient:
    """Qdrant REST API에 최소 기능만 제공하는 얇은 client."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        timeout: float = 30.0,
    ) -> None:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["api-key"] = api_key
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers=headers,
        )

    def close(self) -> None:
        """열린 HTTP client를 정리한다."""
        self._client.close()

    def get_collection(self, collection_name: str) -> dict[str, Any] | None:
        """컬렉션이 있으면 정보를, 없으면 None을 반환한다."""
        response = self._client.get(f"/collections/{collection_name}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def ensure_dense_collection(
        self,
        *,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> dict[str, Any]:
        """dense cosine 컬렉션이 없으면 생성한다."""
        existing = self.get_collection(collection_name)
        if existing is not None:
            return {"created": False, "collection": existing}

        response = self._client.put(
            f"/collections/{collection_name}",
            json={
                "vectors": {
                    "size": vector_size,
                    "distance": distance,
                }
            },
        )
        response.raise_for_status()
        return {"created": True, "collection": response.json()}

    def upsert_points(
        self,
        *,
        collection_name: str,
        points: list[dict[str, Any]],
        wait: bool = True,
    ) -> dict[str, Any]:
        """point batch를 Qdrant에 upsert 한다."""
        response = self._client.put(
            f"/collections/{collection_name}/points",
            params={"wait": str(wait).lower()},
            json={"points": points},
        )
        response.raise_for_status()
        return response.json()
