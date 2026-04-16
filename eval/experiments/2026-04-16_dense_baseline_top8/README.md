# 2026-04-16 Dense Baseline Top-8

현재 stage4 dense retrieval 기준선을 별도 보관한 실험 스냅샷이다.

## 설정

- retrieval 방식: dense only
- reranker: 없음
- BM25 / sparse: 없음
- top_k: 8
- restrict_to_document: true
- eval set: `eval/retrieval_v1.json`

## 전체 지표

- case_count: 52
- chunk_hit_rate: 0.8846153846153846
- chunk_mean_recall: 0.8141025641025641
- chunk_mrr: 0.725
- parent_hit_rate: 0.9230769230769231
- parent_mean_recall: 0.9038461538461539
- parent_mrr: 0.7185897435897436

## 문서별 지표

- doc 1: hit 1.0 / recall 1.0 / mrr 0.8125
- doc 2: hit 1.0 / recall 0.7777777777777778 / mrr 1.0
- doc 3: hit 0.7777777777777778 / recall 0.7777777777777778 / mrr 0.7037037037037037
- doc 4: hit 0.75 / recall 0.75 / mrr 0.5572916666666666
- doc 6: hit 0.875 / recall 0.7291666666666666 / mrr 0.671875
- doc 7: hit 0.9 / recall 0.85 / mrr 0.6033333333333333

## 비고

- `doc6` 평가셋 gold 라벨 수정이 반영된 기준선이다.
- 이후 reranker, BM25, hybrid 검색 실험은 이 스냅샷과 비교한다.
- 원본 상세 리포트는 같은 디렉터리의 `report.json`을 사용한다.
