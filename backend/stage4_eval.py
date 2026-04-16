"""CLI entrypoint for stage-4 retrieval evaluation."""

from __future__ import annotations

import argparse
import json

from backend.stage4_retrieval.evaluation import run_stage4_retrieval_evaluation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation against a labeled eval set.",
    )
    parser.add_argument(
        "--eval-set",
        default="eval/retrieval_v1.json",
        help="평가셋 JSON 파일 경로",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="최종 retrieval top-k",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=None,
        help="최종 top-k를 자르기 전에 넓게 가져올 dense 후보 수",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="평가 리포트 저장 경로",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Qdrant collection 이름 override",
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        default=[],
        help="특정 문서만 평가할 때 doc_id를 반복 지정",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="특정 케이스만 평가할 때 case_id를 반복 지정",
    )
    parser.add_argument(
        "--no-document-filter",
        action="store_true",
        help="document_id 필터 없이 retrieval을 평가",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    response = run_stage4_retrieval_evaluation(
        eval_set_path=args.eval_set,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        output_path=args.output,
        collection_name=args.collection_name,
        restrict_to_document=not args.no_document_filter,
        doc_ids=args.doc_id or None,
        case_ids=args.case_id or None,
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
