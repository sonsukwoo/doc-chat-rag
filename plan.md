# Chatbot Graph Plan

이 문서는 현재 프로젝트의 챗봇 그래프 설계 기준을 고정해 두는 작업 메모입니다.
목적은 두 가지입니다.

- 구현 전에 결정된 아키텍처를 잊지 않기
- 이후 변경이 생겨도 어떤 판단을 유지하고 무엇을 보류했는지 추적하기

## 1. 현재 결론

- 챗봇 오케스트레이션은 LangGraph로 구현한다.
- 시작 구조는 멀티 에이전트 hand-off가 아니라 `single stateful agent + ToolNode + retrieval subgraph`로 간다.
- 단기 대화 메모리는 LangGraph `PostgresSaver`로 관리한다.
- 채팅방 격리 기준은 `room_id`다.
- LangGraph 체크포인트 재개 기준은 `thread_id`다.
- 초기에는 `room_id == thread_id`로 시작할 수 있지만, 스키마는 분리 가능한 형태로 설계한다.
- Qdrant는 채팅방마다 컬렉션을 나누기보다 공용 컬렉션 + payload filter 방식을 기본으로 한다.
- child chunk 벡터는 Qdrant에 저장한다.
- parent/window 확장용 메타데이터는 Postgres에서 조회하는 방향으로 간다.
- `parents.json`은 artifact/debug 용도로 유지할 수 있지만, 런타임의 주 조회원은 아니다.

## 2. 왜 이 구조로 가는가

- 현재 핵심 문제는 부서 간 hand-off보다 `문서 검색`, `근거 확장`, `맥락 유지`, `되묻기`, `채팅방 격리`다.
- 따라서 고객센터형 멀티 에이전트보다 custom workflow가 더 적합하다.
- ToolNode, tools_condition, interrupt, checkpointer는 LangGraph 공식 패턴과도 잘 맞는다.

## 3. v1 챗봇 그래프

예상 흐름:

1. `load_request_context`
2. `classify_query`
3. `clarify_if_needed`
4. `agent_llm`
5. `tools`
6. `grounding_check`
7. `fallback_or_retrieve_deeper`
8. `compose_answer_with_citations`

그래프 개요:

- `load_request_context`
  - room 정보, 연결 문서, 유저 메시지, 실행 컨텍스트를 state에 적재
- `classify_query`
  - 질문 유형과 retrieval 정책 초안 결정
- `clarify_if_needed`
  - 문서 범위가 모호하면 `interrupt()`로 되묻기
- `agent_llm`
  - 답변 초안 생성 또는 tool call 생성
- `tools`
  - ToolNode로 검색/확장/asset 조회 실행
- `grounding_check`
  - retrieval 0건은 규칙 기반으로 부족 판정
  - retrieval hit가 있으면 구조화된 LLM으로 충분성/추가검색/되묻기 필요 여부 판정
- `fallback_or_retrieve_deeper`
  - 필요 시 deeper retrieval 수행
- `compose_answer_with_citations`
  - citation 포함 최종 답변 생성

## 4. 상태 설계 원칙

초기 state 후보:

- `messages`
- `room_id`
- `thread_id`
- `user_id`
- `active_document_ids`
- `query_analysis`
- `retrieval_policy`
- `retrieval_hits`
- `expanded_context_blocks`
- `citations`
- `needs_clarification`
- `clarification_payload`
- `answer_draft`
- `final_answer`
- `logs`

원칙:

- `messages`만으로 모든 것을 해결하지 않는다.
- 검색 결과, 검색 정책, 최종 citation은 별도 필드로 둔다.
- 대용량 artifact 전체를 state에 넣지 않는다.

## 5. 툴 설계 원칙

모델에게 노출할 툴은 "업무 의미 단위"로만 둔다.
저수준 스위치(`hybrid_on`, `rerank_on`)는 툴로 직접 노출하지 않는다.

v1 후보 툴:

- `search_room_knowledge`
- `expand_context_window`
- `load_visual_asset`
- `list_room_documents`
- `web_search` (선택)

규칙:

- `room_id`는 모델 인자로 직접 주지 않는다.
- tool runtime context 또는 graph state에서 주입한다.

## 6. 검색 정책 기본값

현재 기준 기본 retrieval 전략:

- 기본 경로: `dense + window`
- 깊은 검색 경로: `dense + rerank + window`
- 조건부 경로: lexical 성격이 강하면 `hybrid + rerank + window` 후보

중요:

- retrieval 정책은 모델이 임의로 조작하게 하지 않는다.
- `classify_query`와 `fallback_or_retrieve_deeper`가 정책을 통제한다.

## 7. 저장소 책임 분리

### Postgres

- `chat_rooms`
- `chat_threads`
- `room_documents`
- `document_parents`
- `document_assets`
- LangGraph checkpointer 전용 테이블

### Qdrant

- child chunk 벡터 저장
- payload:
  - `room_id`
  - `document_id`
  - `chunk_id`
  - `parent_id`
  - `page`
  - `chunk_type`
  - `asset_ref`

### Filesystem

- `backend/outputs/<document_id>/...`
- artifact/debug/rebuild 용도

## 8. 문서 업로드/삭제/교체 규칙

- 업로드:
  - stage1~3 처리
  - child chunk upsert
  - parent/assets 메타는 Postgres 저장
- 같은 문서 재업로드:
  - 기존 `room_id + document_id` 또는 별도 logical key 기준 삭제 후 재업로드
- 문서 삭제:
  - Qdrant에서 해당 room/document payload filter 삭제
  - Postgres parent/assets/doc row 정리

## 9. 보류 항목

아래 항목은 v1 범위에서 제외한다.

- full hand-off multi-agent
- supervisor 기반 팀 구조
- 복잡한 계층형 multi-agent
- parent 중심 retrieval을 기본 전략으로 승격
- web search 상시 활성화

## 10. 다음 구현 순서

1. `room_id / thread_id / document_id` 데이터 모델 확정
2. Postgres 테이블 설계
3. stage3 indexing payload에 `room_id` 추가
4. parent 메타 Postgres 저장 경로 추가
5. stage4 retrieval을 room-aware로 수정
6. LangGraph chatbot state 정의
7. 기본 노드 구현
8. ToolNode 연결
9. `interrupt()` 기반 clarify 연결
10. citation 포함 답변 출력 고정

현재 완료 범위:

- 1, 2 완료
- 3 완료
- 4 완료
  - `document_parents`, `document_assets`, `document_chunks` 저장 repository/service가 추가됨
  - `run_stage3_for_document(..., room_id=...)` 경로에서 stage3 결과를 Postgres와 동기화할 수 있음
- 5 완료
- 6 완료
- 7 완료
- 8 완료
  - stage5 service가 room context를 읽어 `search_room_knowledge` 툴을 실제 stage4 검색에 연결함
  - `agent_llm`은 실제 tool-calling model node로 교체됨
  - tool은 `runtime.state`에서 room/document 범위를 읽음
- 9 완료
- 10 부분 완료
  - retrieval hit 기반 citation과 grounded final answer를 생성함
  - `expand_context_window`는 Postgres `document_chunks` 기반 실제 window 문맥을 조회함
  - `load_visual_asset`는 Postgres `document_assets` 기반 실제 asset 메타를 조회함
  - visual asset을 최종 stage5 output에 첨부할 수 있음
  - 다만 richer citation formatting과 웹 검색 연결은 다음 단계

## 11. 문서 운영 규칙

이 파일은 아래 경우에만 갱신한다.

- 아키텍처 방향이 바뀌었을 때
- v1 범위가 확정/축소되었을 때
- storage 책임이 변경되었을 때
- retrieval 기본 전략이 변경되었을 때

세부 구현 로그나 실험 결과는 이 파일이 아니라 별도 문서에 남긴다.

## 12. 제품 플로우 고정안

현재 프로젝트의 최종 제품 플로우는 "문서 파이프라인"이 아니라 "채팅방 중심 문서 워크스페이스"로 고정한다.

사용자 플로우:

1. 사용자가 `새 채팅방 만들기`를 누른다.
2. 첫 진입에서는 문서 업로드가 필수다.
3. 문서를 올리면 자동으로 `stage1 -> stage2`까지 진행한다.
4. stage2가 끝나면 검수 화면으로 이동한다.
5. 사용자는 preview 기반으로 element drop / restore를 검토한다.
6. 사용자가 확정하면 `review overlay -> stage3 chunking/indexing`을 수행한다.
7. 인덱싱이 끝나면 채팅방이 `ready` 상태가 되고 바로 채팅할 수 있다.
8. 채팅 중에는 문서 패널에서 현재 방 문서를 확인하고, 문서를 추가/제거/재검수/재색인할 수 있다.

중요 구현 원칙:

- 사용자 경험상 "문서를 넣어야 방이 생성되는 것처럼" 보이게 한다.
- 내부 구현은 `draft room -> ready room` 전환 구조로 둔다.
- 검수는 원본 파일을 직접 수정하는 방식이 아니라 `review_decisions.json` overlay를 누적하는 방식으로 둔다.
- 문서 수정이 생기면 해당 문서만 다시 chunking/indexing 한다.
- 채팅 검색 범위는 항상 현재 `room_id`와 연결된 `active_document_ids`로 제한한다.

문서 패널에서 지원해야 할 기능:

- 현재 방에 연결된 문서 목록 보기
- 문서 추가
- 문서 제거
- 기존 문서 재검수
- 기존 drop restore
- review 수정 후 재색인

이 플로우는 향후 프론트/백엔드/저장소 설계의 기준으로 사용한다.

## 13. 현재 stage5 그래프

현재 `backend/stage5_chatbot` 패키지는 아래 경계를 기준으로 실제 동작 가능한 graph까지 연결돼 있다.

- `config.py`
  - stage5 env 설정
  - 모델명, checkpointer backend, retrieval 기본값
- `schemas.py`
  - stage5 외부 입력/출력 스키마
- `state.py`
  - LangGraph 공유 상태 정의
- `prompts.py`
  - 시스템 프롬프트 helper
- `tools.py`
  - ToolNode에 연결할 툴 정의
  - room/document 범위는 `runtime.state`에서 읽음
- `nodes.py`
  - load/classify/clarify/agent/grounding/fallback/compose 노드
  - `agent_llm`은 실제 tool-calling model node
  - `fallback_or_retrieve_deeper`는 graph가 정책적으로 deeper retrieval 수행
- `compose_answer_with_citations`는 retrieval hit 기반 최종 grounded answer 생성
  - `grounding_check`는 0건 fast-path는 deterministic, 1건 이상은 structured LLM 판정
- `checkpointer.py`
  - memory/postgres checkpointer factory
- `graph.py`
  - StateGraph 조립
- `service.py`
  - 외부에서 호출하는 stage5 진입점
- `backend/stage5.py`
  - backward-compatible import entrypoint

현재 상태는 아래 범위까지 연결된 상태다.

- PostgresSaver 기반 checkpoint 저장
- room context 로드
- room-aware `search_room_knowledge` 툴 연결
- stage4 room-scoped retrieval 서비스 연결
- 실제 tool-calling agent loop
- retrieval hit 기반 grounded final answer 생성
- Postgres `document_chunks` 기반 context window 확장
- Postgres `document_assets` 기반 visual asset 조회

아직 남은 것은 아래다.

- grounding 결과를 반영한 deeper retrieval 정책 연결
- grounding 결과를 반영한 deeper retrieval / clarify 정책 고도화
- citation 포함 최종 답변 렌더링 고도화
- `web_search` 실제 연결

## 14. Room 중심 서비스 구현 순서

다음 구현은 CLI 파이프라인을 늘리는 방식이 아니라, room-aware 서비스 플로우를 먼저 완성하는 순서로 진행한다.

1. room CRUD API
2. room에 첫 문서를 업로드하면서 `draft room`을 생성하는 API
3. room 기준 `stage1 -> stage2 -> review -> stage3` 실행 연결
4. room 기준 stage5 chat API
5. clarification interrupt / resume API
6. 문서 패널용 문서 추가 / 제거 / 재검수 / 재색인 API
7. 프론트에서 GPT 스타일 채팅 UI와 문서 패널 연결

통합 테스트는 위 흐름이 붙은 뒤에만 진행한다.
그 전에는 Qdrant/Postgres/outputs를 초기화하고 재적재하는 비용이 큰데, 아직 진입점이 완전히 통일되지 않아 의미가 떨어진다.

## 15. 통합 테스트 원칙

프론트 연결 후 end-to-end 테스트를 시작할 때는 아래 기준으로 초기화 후 다시 적재한다.

- `backend/outputs` 테스트 산출물 정리
- Qdrant 테스트 컬렉션 정리 또는 새 컬렉션 사용
- Postgres의 room/thread/document 메타 정리

이유:

- 기존 CLI 실험 산출물과 room-aware 서비스 경로가 혼재되면 추적이 어려워진다.
- 통합 테스트 시점에는 "새 방 생성 -> 첫 문서 업로드 -> 검수 -> 인덱싱 -> 채팅" 전체를 처음부터 재현 가능해야 한다.

운영 단계에서는 이 초기화 전략을 쓰지 않고, room/document lifecycle API로 증분 관리한다.

## 16. Postgres 스키마 기준

현재 Postgres는 DB 1개 + schema 분리 방식으로 고정한다.

- DB 이름: `rag_chat_app`
- schema:
  - `app_chat`
  - `app_doc`
  - `app_pipeline`
  - `app_checkpoint`

각 schema의 역할:

- `app_chat`
  - `rooms`
  - `threads`
- `app_doc`
  - `documents`
  - `room_documents`
  - `document_parents`
  - `document_assets`
  - `document_review_decisions`
- `app_pipeline`
  - `document_stage_status`
  - `document_stage_runs`
- `app_checkpoint`
  - LangGraph `PostgresSaver` 테이블

원칙:

- 대화 checkpoint 데이터와 앱 메타데이터를 같은 DB에 두되 schema는 분리한다.
- child chunk 벡터는 계속 Qdrant에 저장한다.
- parent / asset / room / thread / stage status 메타는 Postgres에 저장한다.
