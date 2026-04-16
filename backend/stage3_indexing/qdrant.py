"""Shared Qdrant REST helper used by stage-3 indexing and stage-4 retrieval."""

from __future__ import annotations

from typing import Any

import httpx


class QdrantRestClient:
    """stage3/stage4에서 공통으로 쓰는 최소 Qdrant REST client."""

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

    def _extract_dense_vector_config(
        self,
        collection: dict[str, Any],
    ) -> tuple[int, str] | None:
        """Qdrant collection 응답에서 기본 dense vector 설정을 읽는다."""
        result = collection.get("result") or {}
        config = result.get("config") or {}
        params = config.get("params") or {}
        vectors = params.get("vectors")

        if isinstance(vectors, dict) and "size" in vectors and "distance" in vectors:
            size = vectors.get("size")
            distance = vectors.get("distance")
            if size is None or distance is None:
                return None
            return int(size), str(distance)

        return None

    def ensure_dense_collection(
        self,
        *,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> dict[str, Any]:
        """dense cosine 컬렉션이 없으면 생성하고, 있으면 스키마를 검증한다."""
        existing = self.get_collection(collection_name)
        if existing is not None:
            existing_config = self._extract_dense_vector_config(existing)
            if existing_config is None:
                raise ValueError(
                    f"Qdrant collection '{collection_name}'의 dense vector 설정을 해석하지 못했습니다."
                )

            existing_size, existing_distance = existing_config
            if existing_size != vector_size or existing_distance != distance:
                raise ValueError(
                    "Qdrant collection schema mismatch: "
                    f"name={collection_name}, "
                    f"expected(size={vector_size}, distance={distance}), "
                    f"actual(size={existing_size}, distance={existing_distance})"
                )
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

    def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        limit: int = 10,
        with_payload: bool | list[str] | dict[str, Any] = True,
        with_vector: bool = False,
        query_filter: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """dense query vector로 point top-k를 조회한다."""
        request_payload: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": with_vector,
        }
        if query_filter:
            request_payload["filter"] = query_filter
        if score_threshold is not None:
            request_payload["score_threshold"] = score_threshold

        response = self._client.post(
            f"/collections/{collection_name}/points/query",
            json=request_payload,
        )
        response.raise_for_status()
        payload = response.json()
        result = payload.get("result") or {}
        if isinstance(result, list):
            return result
        return list(result.get("points") or [])
