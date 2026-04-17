# 2026-04-18 Sparse-Filtered Hybrid Retrieval

현재 stage4 retrieval 정책을 정리하기 위해 수행한 sparse branch 재설계 실험 스냅샷이다.

## 목적

- BM25 sparse branch를 전체 청크에 무차별 적용하지 않고,
- `sparse_keep + sparse_text` 정책으로 lexical anchor가 있는 청크만 sparse에 태웠을 때,
- dense 대비 hybrid가 얼마나 회복되는지 확인한다.

## 적용 변경

- stage3 chunking
  - 각 청크에 `sparse_keep`, `sparse_text`, `sparse_exclude_reason` 추가
  - text 청크는 `section_title + 앞부분 lexical preview`
  - visual 청크는 `caption + summary` 중심 sparse text 구성
- stage3 indexing
  - dense는 전체 청크 유지
  - BM25는 `sparse_keep == true` 청크만 업로드
- stage4 retrieval
  - hybrid 검색 시 `sparse_keep == true` + 실제 sparse vector 존재 조건으로만 BM25 branch 조회

## 결론

- 전체 질의 기준
  - dense가 여전히 기본값으로 더 안정적이다.
  - hybrid는 `hit / recall`은 개선됐지만 `MRR`은 dense보다 낮다.
- hybrid 친화 질의 기준
  - hybrid가 `hit / recall`에서 dense보다 좋다.
  - 다만 ranking 품질(`MRR`)은 아직 dense가 근소 우위다.

따라서 현재 운영 정책은 아래와 같다.

- 기본값: `dense + window`
- 옵션: `hybrid`
- 활성화 기준: `표/그림 번호`, `정확한 API/환경변수명`, `파라미터명`, `고유명사/식별자` 등 lexical anchor가 강한 질의
- `MMR`: 코드에 남겨 두되 기본 비활성화

## 대표 리포트

- 전체 질의 비교
  - `eval/reports/retrieval_v1_sparse_filtered_compare.json`
- sparse-filtered hybrid 가중치 비교
  - `eval/reports/retrieval_v1_sparse_filtered_weighted_rrf_compare.json`
- hybrid 친화 질의 슬라이스 비교
  - `eval/reports/retrieval_v1_hybrid_friendly_compare.json`

## 참고 수치

### 전체 질의

- dense
  - `chunk_hit_rate = 0.8654`
  - `chunk_mean_recall = 0.8173`
  - `chunk_mrr = 0.7130`
- hybrid `(RRF 3:1)`
  - `chunk_hit_rate = 0.9231`
  - `chunk_mean_recall = 0.8654`
  - `chunk_mrr = 0.6568`

### hybrid 친화 질의

- dense
  - `chunk_hit_rate = 0.8824`
  - `chunk_mean_recall = 0.8235`
  - `chunk_mrr = 0.6368`
- hybrid `(equal weight)`
  - `chunk_hit_rate = 0.9412`
  - `chunk_mean_recall = 0.9216`
  - `chunk_mrr = 0.6261`
