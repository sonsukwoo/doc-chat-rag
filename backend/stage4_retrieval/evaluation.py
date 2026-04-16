"""Stage-4 retrieval evaluation helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from .pipeline import run_stage4_retrieval

Stage4Runner = Callable[..., dict[str, Any]]


def build_stage4_eval_output_path(
    *,
    eval_set_path: str | Path,
    top_k: int,
    output_path: str | Path | None = None,
) -> str:
    """평가 리포트 저장 경로를 계산한다."""
    if output_path is not None:
        return str(Path(output_path).expanduser().resolve())

    eval_path = Path(eval_set_path).expanduser().resolve()
    report_dir = (eval_path.parent / "reports").resolve()
    return str((report_dir / f"{eval_path.stem}_top{top_k}_report.json").resolve())


def _load_eval_payload(eval_set_path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(eval_set_path).expanduser().resolve().read_text())
    if not isinstance(payload, dict):
        raise ValueError("평가셋 파일은 dict 형태여야 합니다.")
    return payload


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _reciprocal_rank(
    retrieved_ids: list[str],
    gold_ids: list[str],
) -> float:
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0
    for index, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in gold_set:
            return 1.0 / index
    return 0.0


def _compute_match_metrics(
    *,
    retrieved_ids: list[str],
    gold_ids: list[str],
) -> dict[str, Any]:
    if not gold_ids:
        return {
            "matched_ids": [],
            "hit": None,
            "recall": None,
            "mrr": None,
        }

    gold_set = set(gold_ids)
    matched_ids = [item for item in retrieved_ids if item in gold_set]
    return {
        "matched_ids": matched_ids,
        "hit": bool(matched_ids),
        "recall": len(set(matched_ids)) / len(gold_set),
        "mrr": _reciprocal_rank(retrieved_ids, gold_ids),
    }


def _build_case_lookup(
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        str(document["doc_id"]): document
        for document in payload.get("documents") or []
        if isinstance(document, dict) and str(document.get("doc_id") or "")
    }


def _aggregate_case_group(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    completed_cases = [case for case in cases if case["status"] == "completed"]
    chunk_cases = [
        case for case in completed_cases if case["chunk_hit"] is not None
    ]
    parent_cases = [
        case for case in completed_cases if case["parent_hit"] is not None
    ]

    return {
        "case_count": len(cases),
        "completed_case_count": len(completed_cases),
        "skipped_case_count": sum(1 for case in cases if case["status"] == "skipped"),
        "error_case_count": sum(1 for case in cases if case["status"] == "error"),
        "chunk_hit_rate": _safe_mean(
            [1.0 if case["chunk_hit"] else 0.0 for case in chunk_cases]
        ),
        "chunk_mean_recall": _safe_mean(
            [float(case["chunk_recall"]) for case in chunk_cases]
        ),
        "chunk_mrr": _safe_mean(
            [float(case["chunk_mrr"]) for case in chunk_cases]
        ),
        "parent_case_count": len(parent_cases),
        "parent_hit_rate": _safe_mean(
            [1.0 if case["parent_hit"] else 0.0 for case in parent_cases]
        ),
        "parent_mean_recall": _safe_mean(
            [float(case["parent_recall"]) for case in parent_cases]
        ),
        "parent_mrr": _safe_mean(
            [float(case["parent_mrr"]) for case in parent_cases]
        ),
    }


def _select_cases(
    *,
    payload: dict[str, Any],
    doc_ids: list[str] | None,
    case_ids: list[str] | None,
) -> list[dict[str, Any]]:
    selected_cases: list[dict[str, Any]] = []
    allowed_doc_ids = set(doc_ids or [])
    allowed_case_ids = set(case_ids or [])

    for case in payload.get("cases") or []:
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("case_id") or "")
        doc_id = str(case.get("doc_id") or "")
        if allowed_doc_ids and doc_id not in allowed_doc_ids:
            continue
        if allowed_case_ids and case_id not in allowed_case_ids:
            continue
        selected_cases.append(case)
    return selected_cases


def run_stage4_retrieval_evaluation(
    *,
    eval_set_path: str | Path,
    top_k: int,
    fetch_k: int | None = None,
    output_path: str | Path | None = None,
    collection_name: str | None = None,
    restrict_to_document: bool = True,
    doc_ids: list[str] | None = None,
    case_ids: list[str] | None = None,
    retrieval_runner: Stage4Runner | None = None,
) -> dict[str, Any]:
    """평가셋 전체를 순회하며 stage4 retrieval 품질을 채점한다."""
    payload = _load_eval_payload(eval_set_path)
    document_lookup = _build_case_lookup(payload)
    selected_cases = _select_cases(
        payload=payload,
        doc_ids=doc_ids,
        case_ids=case_ids,
    )
    if not selected_cases:
        raise ValueError("선택된 평가 케이스가 없습니다.")

    resolved_output_path = build_stage4_eval_output_path(
        eval_set_path=eval_set_path,
        top_k=top_k,
        output_path=output_path,
    )
    runner = retrieval_runner or run_stage4_retrieval

    case_reports: list[dict[str, Any]] = []
    for case in selected_cases:
        case_id = str(case["case_id"])
        doc_id = str(case["doc_id"])
        document = document_lookup.get(doc_id)
        if document is None:
            raise ValueError(f"doc_id={doc_id} 문서 메타데이터가 없습니다.")

        source_dir = Path(str(document["source_dir"])).expanduser().resolve()
        stage3_dir = source_dir / "stage3"
        chunks_json_path = (
            (stage3_dir / "chunks.json").resolve()
            if (stage3_dir / "chunks.json").exists()
            else (source_dir / "chunks.json").resolve()
        )
        parents_json_path = (
            (stage3_dir / "parents.json").resolve()
            if (stage3_dir / "parents.json").exists()
            else (source_dir / "parents.json").resolve()
        )
        query = str(case["query"])
        gold_chunk_ids = [str(item) for item in case.get("gold_chunk_ids") or []]
        gold_parent_ids = [str(item) for item in case.get("gold_parent_ids") or []]

        try:
            retrieval_output = runner(
                {
                    "query": query,
                    "chunks_json_path": str(chunks_json_path),
                    "parents_json_path": str(parents_json_path),
                    "output_dir": str((source_dir / "stage4").resolve() if (source_dir / "stage4").exists() else source_dir),
                    "document_id": doc_id,
                    "collection_name": collection_name,
                    "top_k": top_k,
                    "fetch_k": fetch_k,
                    "restrict_to_document": restrict_to_document,
                },
                persist_manifest=False,
            )
            status = str(retrieval_output["status"])
            retrievals = list(retrieval_output.get("retrievals") or [])
            retrieved_chunk_ids = [
                str(item.get("chunk_id") or "")
                for item in retrievals
                if str(item.get("chunk_id") or "")
            ]
            retrieved_parent_ids = [
                str(item.get("parent_id") or "")
                for item in retrievals
                if str(item.get("parent_id") or "")
            ]
            if status == "completed":
                chunk_metrics = _compute_match_metrics(
                    retrieved_ids=retrieved_chunk_ids,
                    gold_ids=gold_chunk_ids,
                )
                parent_metrics = _compute_match_metrics(
                    retrieved_ids=retrieved_parent_ids,
                    gold_ids=gold_parent_ids,
                )
            else:
                chunk_metrics = {
                    "matched_ids": [],
                    "hit": None,
                    "recall": None,
                    "mrr": None,
                }
                parent_metrics = {
                    "matched_ids": [],
                    "hit": None,
                    "recall": None,
                    "mrr": None,
                }

            case_reports.append(
                {
                    "case_id": case_id,
                    "doc_id": doc_id,
                    "query": query,
                    "query_type": str(case.get("query_type") or ""),
                    "difficulty": str(case.get("difficulty") or ""),
                    "status": status,
                    "skip_reason": retrieval_output.get("skip_reason"),
                    "top_k": top_k,
                    "gold_chunk_ids": gold_chunk_ids,
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "matched_chunk_ids": chunk_metrics["matched_ids"],
                    "chunk_hit": chunk_metrics["hit"],
                    "chunk_recall": chunk_metrics["recall"],
                    "chunk_mrr": chunk_metrics["mrr"],
                    "gold_parent_ids": gold_parent_ids,
                    "retrieved_parent_ids": retrieved_parent_ids,
                    "matched_parent_ids": parent_metrics["matched_ids"],
                    "parent_hit": parent_metrics["hit"],
                    "parent_recall": parent_metrics["recall"],
                    "parent_mrr": parent_metrics["mrr"],
                    "top_hits": [
                        {
                            "rank": rank,
                            "chunk_id": str(item.get("chunk_id") or ""),
                            "parent_id": str(item.get("parent_id") or "") or None,
                            "score": float(item.get("score") or 0.0),
                            "chunk_type": str(item.get("chunk_type") or ""),
                            "section_title": item.get("section_title"),
                        }
                        for rank, item in enumerate(retrievals, start=1)
                    ],
                    "notes": str(case.get("notes") or ""),
                }
            )
        except Exception as exc:
            case_reports.append(
                {
                    "case_id": case_id,
                    "doc_id": doc_id,
                    "query": query,
                    "query_type": str(case.get("query_type") or ""),
                    "difficulty": str(case.get("difficulty") or ""),
                    "status": "error",
                    "skip_reason": None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "top_k": top_k,
                    "gold_chunk_ids": gold_chunk_ids,
                    "retrieved_chunk_ids": [],
                    "matched_chunk_ids": [],
                    "chunk_hit": None,
                    "chunk_recall": None,
                    "chunk_mrr": None,
                    "gold_parent_ids": gold_parent_ids,
                    "retrieved_parent_ids": [],
                    "matched_parent_ids": [],
                    "parent_hit": None,
                    "parent_recall": None,
                    "parent_mrr": None,
                    "top_hits": [],
                    "notes": str(case.get("notes") or ""),
                }
            )

    cases_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cases_by_query_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case_report in case_reports:
        cases_by_doc[case_report["doc_id"]].append(case_report)
        cases_by_query_type[case_report["query_type"]].append(case_report)

    output = {
        "eval_set_path": str(Path(eval_set_path).expanduser().resolve()),
        "output_path": resolved_output_path,
        "collection_name": collection_name,
        "top_k": top_k,
        "fetch_k": fetch_k,
        "restrict_to_document": restrict_to_document,
        "case_count": len(case_reports),
        "completed_case_count": sum(
            1 for case in case_reports if case["status"] == "completed"
        ),
        "skipped_case_count": sum(
            1 for case in case_reports if case["status"] == "skipped"
        ),
        "error_case_count": sum(
            1 for case in case_reports if case["status"] == "error"
        ),
        "metrics": _aggregate_case_group(case_reports),
        "per_doc": {
            doc_id: _aggregate_case_group(doc_cases)
            for doc_id, doc_cases in sorted(
                cases_by_doc.items(),
                key=lambda item: int(item[0]) if item[0].isdigit() else item[0],
            )
        },
        "per_query_type": {
            query_type: _aggregate_case_group(query_cases)
            for query_type, query_cases in sorted(cases_by_query_type.items())
        },
        "cases": case_reports,
    }

    output_path_obj = Path(resolved_output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    output_path_obj.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    return output
