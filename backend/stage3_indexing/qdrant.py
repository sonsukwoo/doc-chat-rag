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

    def delete_collection(self, collection_name: str) -> dict[str, Any] | None:
        """컬렉션이 있으면 삭제하고, 없으면 None을 반환한다."""
        response = self._client.delete(f"/collections/{collection_name}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def _extract_dense_vector_config(
        self,
        collection: dict[str, Any],
    ) -> tuple[int, str] | None:
        """Qdrant collection 응답에서 기본 dense vector 설정을 읽는다."""
        named_vectors = self._extract_named_dense_vector_configs(collection)
        if named_vectors is None:
            return None
        return named_vectors.get("")

    def _extract_named_dense_vector_configs(
        self,
        collection: dict[str, Any],
    ) -> dict[str, tuple[int, str]] | None:
        """Qdrant collection 응답에서 dense named vector 설정을 읽는다."""
        result = collection.get("result") or {}
        config = result.get("config") or {}
        params = config.get("params") or {}
        vectors = params.get("vectors")

        if isinstance(vectors, dict) and "size" in vectors and "distance" in vectors:
            size = vectors.get("size")
            distance = vectors.get("distance")
            if size is None or distance is None:
                return None
            return {"": (int(size), str(distance))}

        if isinstance(vectors, dict):
            named_vectors: dict[str, tuple[int, str]] = {}
            for name, vector_config in vectors.items():
                if not isinstance(vector_config, dict):
                    continue
                size = vector_config.get("size")
                distance = vector_config.get("distance")
                if size is None or distance is None:
                    continue
                named_vectors[str(name)] = (int(size), str(distance))
            if named_vectors:
                return named_vectors

        return None

    def _extract_sparse_vector_names(
        self,
        collection: dict[str, Any],
    ) -> set[str]:
        """Qdrant collection 응답에서 sparse named vector 이름을 읽는다."""
        result = collection.get("result") or {}
        config = result.get("config") or {}
        params = config.get("params") or {}
        sparse_vectors = (
            params.get("sparse_vectors")
            or params.get("sparseVectors")
            or params.get("sparse_vectors_config")
        )

        if isinstance(sparse_vectors, dict) and (
            "modifier" in sparse_vectors or "index" in sparse_vectors
        ):
            return {""}

        if not isinstance(sparse_vectors, dict):
            return set()

        return {
            str(name)
            for name, vector_config in sparse_vectors.items()
            if isinstance(vector_config, dict)
        }

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

    def ensure_hybrid_collection(
        self,
        *,
        collection_name: str,
        vector_size: int,
        dense_vector_name: str = "dense",
        bm25_vector_name: str = "bm25",
        distance: str = "Cosine",
    ) -> dict[str, Any]:
        """dense+bM25 sparse 컬렉션이 없으면 생성하고, 있으면 스키마를 검증한다."""
        existing = self.get_collection(collection_name)
        if existing is not None:
            dense_vectors = self._extract_named_dense_vector_configs(existing)
            if dense_vectors is None or dense_vector_name not in dense_vectors:
                raise ValueError(
                    f"Qdrant collection '{collection_name}'에서 dense named vector '{dense_vector_name}'를 찾지 못했습니다."
                )

            existing_size, existing_distance = dense_vectors[dense_vector_name]
            if existing_size != vector_size or existing_distance != distance:
                raise ValueError(
                    "Qdrant collection schema mismatch: "
                    f"name={collection_name}, "
                    f"expected_dense(name={dense_vector_name}, size={vector_size}, distance={distance}), "
                    f"actual_dense(name={dense_vector_name}, size={existing_size}, distance={existing_distance})"
                )

            sparse_vector_names = self._extract_sparse_vector_names(existing)
            if bm25_vector_name not in sparse_vector_names:
                raise ValueError(
                    "Qdrant collection schema mismatch: "
                    f"name={collection_name}, missing_sparse={bm25_vector_name}"
                )

            return {"created": False, "collection": existing}

        response = self._client.put(
            f"/collections/{collection_name}",
            json={
                "vectors": {
                    dense_vector_name: {
                        "size": vector_size,
                        "distance": distance,
                    }
                },
                "sparse_vectors": {
                    bm25_vector_name: {
                        "modifier": "idf",
                    }
                },
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

    def delete_points_by_filter(
        self,
        *,
        collection_name: str,
        query_filter: dict[str, Any],
        wait: bool = True,
    ) -> dict[str, Any]:
        """payload filter에 맞는 point를 삭제한다."""
        response = self._client.post(
            f"/collections/{collection_name}/points/delete",
            params={"wait": str(wait).lower()},
            json={"filter": query_filter},
        )
        response.raise_for_status()
        return response.json()

    def query_points(
        self,
        *,
        collection_name: str,
        query: Any,
        limit: int = 10,
        with_payload: bool | list[str] | dict[str, Any] = True,
        with_vector: bool = False,
        using: str | None = None,
        prefetch: list[dict[str, Any]] | None = None,
        query_filter: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """dense/hybrid query payload로 point top-k를 조회한다."""
        request_payload: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": with_vector,
        }
        if using:
            request_payload["using"] = using
        if prefetch:
            request_payload["prefetch"] = prefetch
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
