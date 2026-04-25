# Stage5 Chatbot Refactor Plan

이 문서는 현재 `thread_id` 기반 stage5 챗봇 리팩토링의 기준 계획을 고정하는 작업 메모다.
목표는 세 가지다.

- 초반 분류 모델의 책임을 줄여 안정성을 높인다.
- 다중 문서/다중 질문 질의에서 검색 품질을 높인다.
- 검색 전 clarify를 없애고, 검색 후 근거 부족 시에만 되묻게 만든다.

## 1. 현재 문제 정의

현재 stage5의 가장 큰 문제는 `classify_query` 한 노드가 너무 많은 일을 한 번에 한다는 점이다.

- 질문 유형 분류
- 문서 선택
- 대화 기억 사용 여부 판단
- 청크 검색 / 대화 기억 / 일반 답변 결정
- 다중 문서 여부 판단
- 문서별 검색 질의 생성
- retrieval mode 힌트 생성

이 구조에서는 초반 판단 하나가 틀리면 뒤의 검색 전체가 흔들린다.

특히 현재 품질을 떨어뜨리는 핵심 문제는 아래 두 가지다.

- 문서 프로파일이 검색 질의에 개입해 사용자 원문 질의를 흐리는 문제
- 여러 질문이 한 번에 들어왔을 때 질문별 문서 매핑이 약해서 결과가 섞이는 문제

## 2. 고정할 핵심 원칙

- 프로파일은 `검색어 생성용`이 아니라 `문서 선택/질문-문서 매핑용`으로만 사용한다.
- 프로파일만으로 문서 내용을 답하는 경로는 두지 않는다.
- 문서명이 명시되지 않은 다중/광역 질문은 프로파일 추정으로 문서를 강제 배치하지 않는다.
- 질문이 하나면 굳이 쪼개지 않고 원문 그대로 검색한다.
- 여러 질문이 한 턴에 섞였을 때만 task로 쪼갠다.
- 같은 문서에 여러 질문이 들어와도 질문 의도가 독립이면 task로 쪼갠다.
- 문서명/파일명이 명시된 task는 해당 문서에만 병렬 검색한다.
- 문서명이 명시되지 않은 다중/광역 질문은 원문 질의를 active documents에 문서별로 균형 검색한다.
- 초반 분류 단계에서는 절대 먼저 사용자에게 되묻지 않는다.
- 문서 질문은 먼저 검색을 시도하고, 근거가 부족할 때만 clarify한다.

## 3. 목표 그래프

목표 그래프는 아래 흐름으로 정리한다.

1. `load_request_context`
2. `classify_intent`
3. `respond_without_documents` 또는 `respond_from_memory` 또는 `plan_retrieval`
4. `agent_llm`
5. `tools`
6. `grounding_check`
7. `fallback_or_retrieve_deeper`
8. `compose_answer_with_citations`

핵심 변화는 기존 `classify_query` 하나를 아래 두 단계로 분리하는 것이다.

- `classify_intent`
  - 질문의 큰 방향만 결정
- `plan_retrieval`
  - 검색이 필요할 때만 어떤 문서를 볼지, 질문을 쪼갤지, 어떤 문서에 매핑할지 결정

## 4. 1단계 분리 기준

### 4-1. classify_intent

여기서는 큰 방향만 정한다.

- `answer_strategy`
  - `direct`
  - `retrieve_chunks`
- `memory_mode`
  - `none`
  - `memory_only`
  - `resolve_for_retrieval`

의미:

- `direct`: 문서와 무관한 일반 질문
- `retrieve_chunks`: 실제 문서 청크 검색이 필요한 질문
- `memory_only`: 이전 대화 자체만으로 답할 수 있는 질문
- `resolve_for_retrieval`: 이전 대화 기억이 필요하지만 최종 답은 다시 검색해야 하는 follow-up 질문

중요:

- 여기서는 문서별 검색 질의를 만들지 않는다.
- 여기서는 clarify를 만들지 않는다.
- 여기서는 profile anchoring을 하지 않는다.

### 4-2. plan_retrieval

`answer_strategy == retrieve_chunks`일 때만 탄다.

여기서 정한다.

- 어떤 문서를 볼지
- 질문을 쪼갤지
- 쪼갠다면 각 하위 질문이 어떤 문서에 붙는지
- dense/hybrid 힌트를 줄지

## 5. retrieval task 설계 방향

최종적으로는 `selected_document_ids + per_document_queries` 구조를 아래 구조로 바꾸는 방향으로 간다.

- `plan_type`
  - `single`
  - `split`
- `retrieval_tasks`
  - `task_id`
  - `subquery`
  - `document_ids`

예:

- `5번 문서에서는 자동완성 범위, 1번 문서에서는 최고 성능 모델, 3번 문서에서는 졸업논문 절차`

는 아래처럼 해석되는 게 목표다.

- task1: `자동완성 범위` -> `5번 문서`
- task2: `최고 성능 모델 이름` -> `1번 문서`
- task3: `졸업논문 제출 절차` -> `3번 문서`

중요:

- `subquery` 는 사용자 원문 표현을 최대한 유지한다.
- 프로파일 제목/토픽/키워드를 질의 뒤에 자동으로 붙이지 않는다.
- 프로파일만으로 문서 내용을 답하는 경로는 제거한다.
- agent 단계에는 상세 프로파일이 아니라 문서 식별 정보만 전달한다.
- 문서명이 명시되지 않은 다중 질문에서 LLM이 프로파일만 보고 문서별 task를 만들더라도 실행 전 원문 기반 문서별 균형 검색으로 되돌린다.

## 6. 검색 실행 원칙

### 단일 질문

- 원문 질의 그대로 검색
- 선택된 문서들에 대해 병렬 검색 가능
- 검색 결과를 합치고 최종 리랭크 1회

### 다중 질문(split)

- task별로 병렬 검색
- 각 task는 해당 문서에만 검색
- task별 후보를 유지
- 최종 답변은 task별 근거를 조합해 생성

### 같은 문서에 여러 질문

- 의도가 둘 이상이면 task로 쪼갠다
- 같은 문서라도 task 단위 검색을 허용한다

예:

- `2번 문서에서 create_agent 설명하고 checkpointer 역할도 알려줘`

는

- task1: `create_agent 설명` -> `2번 문서`
- task2: `checkpointer 역할` -> `2번 문서`

로 처리 가능해야 한다.

## 7. 리랭크 원칙

현재 단계에서는 아래 원칙을 유지한다.

- 검색은 병렬
- 리랭크는 과도하게 병렬화하지 않는다
- 기본적으로 리랭크 모델은 최소 횟수만 사용한다
- 다중 task 검색에서는 task별 후보를 먼저 정리한 뒤 병합한다
- 같은 턴의 여러 task가 있으면 task별 상위 소수 청크만 유지한 채 답변 컨텍스트를 구성한다

현재 보류:

- task별 후처리/리랭크 병렬화
- 문서별 리랭크 후 다시 글로벌 리랭크하는 이중 리랭크

즉 지금 우선순위는 리랭크 구조보다도 `질문-문서 매핑`과 `query contamination 제거`다.

## 8. clarify 정책

clarify는 아래 정책으로 고정한다.

- `classify_intent` 에서는 clarify 금지
- `plan_retrieval` 에서도 clarify 금지
- 문서 질문은 먼저 반드시 한 번 검색
- 검색 후에도 근거가 부족할 때만 `grounding_check` 에서 clarify 허용

즉 “검색 전 되묻기”는 없애고, “검색 후 근거 부족 clarify”만 남긴다.

## 9. 대화 기억(memory) 처리 원칙

대화 기억은 두 종류로 나눈다.

- `memory_only`
  - 대화 자체만으로 답 가능
  - 검색 불필요
- `resolve_for_retrieval`
  - 이전 대화가 필요하지만 최종 답은 검색해야 함

예:

- `내가 몇 번 문서 물어봤지?` -> `memory_only`
- `아까 본 Figure 4 설명해줘` -> `resolve_for_retrieval + retrieve_chunks`

추가 구조화 상태를 점진적으로 도입한다.

- `last_resolved_document_ids`
- `last_retrieval_tasks`
- `last_referenced_entities`
- `last_visual_asset_refs`
- `last_answer_strategy`
- `retrieval_attempt_count`

## 10. 단계별 구현 순서

### Step 1

- `plan.md`를 현재 방향으로 갱신
- `classify_query`를 실제로 `classify_intent -> plan_retrieval`로 분리
- 기존 기능을 최대한 유지하면서 초반 책임만 분산

### Step 2

- `IntentClassificationResult` 도입
- `memory_mode` 분리
- `query_analysis`와 별도로 `intent_analysis` 상태 추가

### Step 3

- 기존 `DocumentSelectionResult` 기반 planner를 정리
- profile anchoring 제거
- `per_document_queries` 자동 보강 제거

현재 상태:

- 완료: profile anchoring 제거
- 완료: planner가 `retrieval_tasks`를 우선 사용하도록 전환
- 완료: `per_document_queries` fallback 제거

### Step 4

- `retrieval_tasks` 구조 도입
- 다중 질문일 때만 split plan 생성
- 같은 문서 내 다중 질문도 task 허용

현재 상태:

- 완료: `retrieval_tasks` 상태/모델 도입
- 완료: 단일 질문이면 1개 task, 다중 문서 비교면 문서별 기본 task 생성
- 완료: 같은 문서 내 다중 질문도 planner가 여러 task를 반환할 수 있는 구조 확보
- 완료: profile-only 답변 경로 제거 후 follow-up 검색에 필요한 문서 범위 state 기록
- 다음 단계: planner 프롬프트 품질 튜닝과 follow-up entity 해석 보강

### Step 5

- `execute_retrieval_plan` 경로 추가
- task별 병렬 검색 도입
- task별 결과를 유지한 채 답변 조립

현재 상태:

- 완료: stage4 retrieval entrypoint가 `retrieval_tasks`를 직접 받아 task별 병렬 검색 가능
- 완료: 다중 task일 때 task coverage를 보존하는 interleave merge 적용
- 완료: 다중 task일 때 task별 후처리/리랭크를 수행한 뒤 상위 소수 청크만 병합하도록 조정
- 완료: 문서명이 명시되지 않은 다중/광역 질문은 원문 질의를 active documents에 문서별 균형 검색하도록 실행 전 보정
- 보류: task별 후처리/리랭크 병렬화

### Step 5-1

- `clarify_if_needed` resume 동작 정리
- 문서 지정 응답은 문서 범위 업데이트로만 해석
- 문서 지정이 아닌 자유 텍스트 응답은 기존 질문에 덧붙이지 않고 새 질문으로 처리

현재 상태:

- 진행 예정: clarify 응답이 기존 user_message를 오염시키는 문제 수정
- 진행 예정: clarify 이후 새 질문이 이전 질문에 끌려가지 않도록 회귀 테스트 추가

### Step 5-2

- search trace는 실제 실행 경로를 기준으로 표시
- task search와 per-document search를 중복 표기하지 않게 조정

현재 상태:

- 진행 예정: multi-task 경로에서 `per_document_search_used`가 잘못 살아나는 표시 문제 수정

### Step 6

- clarify를 post-search only로 더 엄격히 고정
- follow-up 맥락 해석을 구조화 상태 중심으로 보강

현재 상태:

- 완료: `last_resolved_document_ids`, `last_retrieval_tasks`, `last_referenced_entities`, `last_visual_asset_refs` 구조화 상태 도입
- 완료: grounding / compose_answer 경로에서 follow-up state 기록
- 완료: planner가 `resolve_for_retrieval` 또는 follow-up marker가 있을 때 구조화 state를 우선 사용
- 남은 작업: `previous_search_payload` fallback 축소와 clarify UX 정리

## 11. 현재 진행 상태

지금까지 반영된 내용:

- `classify_intent -> plan_retrieval` 실제 분리 완료
- profile anchoring 제거 완료
- `retrieval_tasks` 상태/모델/trace 도입 완료
- stage4 task 기반 병렬 검색 연결 완료
- follow-up 구조화 state 도입 및 planner 연계 완료
- `per_document_queries` fallback 제거 완료
- stage5 / stage4 관련 테스트 통과 확인

지금 남은 핵심 작업:

- follow-up memory 해석에서 `previous_search_payload` 의존도를 더 줄이고 entity 매칭을 보강
- clarify UX와 trace/UI 표시를 새 task 구조에 맞게 정리
- 다중 task 답변 품질에 맞는 top_k / rerank 정책 추가 조정
