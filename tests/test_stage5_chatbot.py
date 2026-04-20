from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from backend.stage5_chatbot.graph import build_graph
from backend.stage5_chatbot.nodes import build_grounding_check_node
from backend.stage5_chatbot.service import run_stage5_chatbot
from backend.stage5_chatbot.tools import build_stage5_tools


def _fake_stage4_runner(
    *,
    query,
    thread_id,
    active_document_ids,
    collection_name=None,
    retrieval_mode=None,
    **_,
):
    return {
        "status": "completed",
        "query": query,
        "thread_id": thread_id,
        "active_document_ids": list(active_document_ids),
        "collection_name": collection_name,
        "retrieval_mode": retrieval_mode or "dense",
        "top_k": 8,
        "fetch_k": 16,
        "per_document_search_used": len(active_document_ids) > 1,
        "score_threshold_applied": None,
        "score_fallback_applied": False,
        "rerank_applied": False,
        "rerank_error": None,
        "mmr_applied": False,
        "retrieved_count": 1,
        "retrievals": [
            {
                "document_id": active_document_ids[0] if active_document_ids else "doc-1",
                "chunk_id": "chunk-1",
                "parent_id": "parent-1",
                "primary_page": 3,
                "section_title": "1. 소개",
                "asset_relative_path": None,
                "text": "이 문서는 랭그래프 기반 RAG 구조와 실험 결과를 설명합니다.",
                "chunk_type": "text",
                "score": 0.91,
            }
        ],
    }


class _CapturingStage4Runner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        return _fake_stage4_runner(**kwargs)


class _FakeToolCallingModel:
    def __init__(self, responses, *, structured_responses=None):
        self._responses = list(responses)
        self._structured_responses = dict(structured_responses or {})

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, **kwargs):
        configured_response = self._structured_responses.get(schema.__name__)
        if configured_response is None and schema.__name__ == "GroundingDecisionResult":
            configured_response = self._structured_responses.get("GroundingCheckResult")
        if configured_response is None:
            raise AssertionError(
                f"structured response was not configured for {schema.__name__}"
            )
        if (
            schema.__name__ == "GroundingDecisionResult"
            and isinstance(configured_response, dict)
            and "action" not in configured_response
        ):
            configured_response = _normalize_grounding_decision_response(
                configured_response
            )
        if isinstance(configured_response, BaseModel):
            return _FakeStructuredOutputModel(configured_response)
        return _FakeStructuredOutputModel(schema(**configured_response))

    def invoke(self, messages):
        if not self._responses:
            raise AssertionError("fake model responses exhausted")
        return self._responses.pop(0)


class _FakeStructuredOutputModel:
    def __init__(self, response):
        self._response = response

    def invoke(self, messages):
        return self._response


def _normalize_grounding_decision_response(
    configured_response: dict[str, object],
) -> dict[str, object]:
    if bool(configured_response.get("needs_clarification")):
        return {
            "action": "clarify",
            "clarification_question": configured_response.get(
                "clarification_question"
            ),
        }
    if bool(configured_response.get("needs_deeper_retrieval")):
        return {
            "action": "retrieve_deeper",
            "clarification_question": None,
        }
    return {
        "action": "answer",
        "clarification_question": None,
    }


class Stage5ChatbotTests(unittest.TestCase):
    def test_build_graph_compile_smoke(self):
        graph = build_graph(
            llm=_FakeToolCallingModel(
                [AIMessage(content="compile smoke response")],
                structured_responses={
                    "GroundingCheckResult": {
                        "enough_evidence": True,
                        "needs_deeper_retrieval": False,
                        "needs_clarification": False,
                        "clarification_question": None,
                        "missing_aspects": [],
                    },
                    "FinalAnswerResult": {"answer": "unused", "grounded": True},
                },
            ),
            tools=build_stage5_tools(stage4_runner=_fake_stage4_runner),
            retrieval_runner=_fake_stage4_runner,
        )
        self.assertIsNotNone(graph)

    def test_dense_execution_is_used_by_default_even_when_thread_prefers_hybrid(self):
        stage4_runner = _CapturingStage4Runner()
        fake_llm = _FakeToolCallingModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_thread_knowledge",
                            "args": {"query": "이 문서를 요약해줘"},
                            "id": "tool-call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="검색 완료"),
            ],
            structured_responses={
                "DocumentSelectionResult": {
                    "query_type": "single_document",
                    "selected_document_ids": ["doc-2"],
                    "per_document_queries": {},
                    "retrieval_mode": "hybrid",
                },
                "GroundingCheckResult": {
                    "enough_evidence": True,
                    "needs_deeper_retrieval": False,
                    "needs_clarification": False,
                    "clarification_question": None,
                    "missing_aspects": [],
                },
                "FinalAnswerResult": {
                    "answer": "핵심은 랭그래프 기반 RAG 구조입니다.",
                    "grounded": True,
                },
            },
        )

        result = run_stage5_chatbot(
            {
                "thread_id": "thread-hybrid",
                "thread_default_retrieval_mode": "hybrid",
                "user_message": "이 문서를 요약해줘",
                "active_document_ids": ["doc-1"],
                "collection_name": "rag_chat_hybrid_thread-hybrid",
                "document_profiles": [
                    {
                        "document_id": "doc-1",
                        "title": "랭그래프 문서",
                        "document_type": "기술 문서",
                        "main_topics": ["랭그래프", "RAG"],
                        "short_summary": "랭그래프 기반 RAG 설명 문서",
                    }
                ],
            },
            checkpointer=InMemorySaver(),
            llm=fake_llm,
            stage4_runner=stage4_runner,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["retrieval_mode"], "dense")
        self.assertEqual(stage4_runner.calls[0]["retrieval_mode"], "dense")
        self.assertTrue(stage4_runner.calls[0]["enable_rerank"])
        self.assertEqual(result["debug_trace"]["model"], "openai:gpt-4.1-mini")
        self.assertEqual(result["debug_trace"]["selected_document_ids"], ["doc-1"])
        self.assertEqual(result["debug_trace"]["tool_calls"][0]["name"], "search_thread_knowledge")
        self.assertTrue(result["debug_trace"]["tool_calls"][0]["rerank_requested"])
        self.assertEqual(result["debug_trace"]["thread_default_retrieval_mode"], "hybrid")
        self.assertEqual(result["debug_trace"]["executed_retrieval_mode"], "dense")

    def test_thread_id_is_required_for_stage5_chatbot(self):
        with self.assertRaisesRegex(ValueError, "thread_id is required"):
            run_stage5_chatbot(
                {
                    "user_message": "안녕",
                    "active_document_ids": ["doc-1"],
                },
                checkpointer=InMemorySaver(),
                llm=_FakeToolCallingModel(
                    [AIMessage(content="unused")],
                    structured_responses={
                        "GroundingCheckResult": {
                            "enough_evidence": True,
                            "needs_deeper_retrieval": False,
                            "needs_clarification": False,
                            "clarification_question": None,
                            "missing_aspects": [],
                        },
                        "FinalAnswerResult": {"answer": "unused", "grounded": True},
                    },
                ),
                stage4_runner=_fake_stage4_runner,
            )

    def test_explicit_document_reference_narrows_retrieval_scope(self):
        stage4_runner = _CapturingStage4Runner()
        fake_llm = _FakeToolCallingModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_thread_knowledge",
                            "args": {
                                "query": "3번 문서 기준으로 랭체인 create_agent 인자 알려줘"
                            },
                            "id": "tool-call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="검색 완료"),
            ],
            structured_responses={
                "DocumentSelectionResult": {
                    "query_type": "multi_document",
                    "selected_document_ids": ["doc-1", "doc-2", "doc-3"],
                    "per_document_queries": {},
                    "retrieval_mode": "dense",
                    "use_rerank": False,
                },
                "GroundingCheckResult": {
                    "enough_evidence": True,
                    "needs_deeper_retrieval": False,
                    "needs_clarification": False,
                    "clarification_question": None,
                    "missing_aspects": [],
                },
                "FinalAnswerResult": {
                    "answer": "3번 문서 기준 답변입니다.",
                    "grounded": True,
                },
            },
        )

        result = run_stage5_chatbot(
            {
                "thread_id": "thread-multi-doc",
                "user_message": "3번 문서 기준으로 랭체인 create_agent 인자 알려줘",
                "active_document_ids": ["doc-2", "doc-3"],
                "document_profiles": [
                    {
                        "document_id": "doc-2",
                        "original_filename": "2.pdf",
                        "title": "랭체인 미들웨어",
                        "document_type": "기술 문서",
                        "main_topics": ["랭체인", "미들웨어"],
                        "short_summary": "랭체인 미들웨어 문서",
                    },
                    {
                        "document_id": "doc-3",
                        "original_filename": "3.pdf",
                        "title": "BM25 검색",
                        "document_type": "기술 문서",
                        "main_topics": ["BM25", "검색"],
                        "short_summary": "검색 문서",
                    },
                ],
            },
            checkpointer=InMemorySaver(),
            llm=fake_llm,
            stage4_runner=stage4_runner,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(stage4_runner.calls[0]["active_document_ids"], ["doc-3"])
        self.assertEqual(result["citations"][0]["document_id"], "doc-3")

    def test_here_expression_does_not_force_clarification_by_itself(self):
        stage4_runner = _CapturingStage4Runner()
        fake_llm = _FakeToolCallingModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_thread_knowledge",
                            "args": {
                                "query": "랭체인 creat_agnet에서 여기서 포함하는 인자값들 뭐뭐 있는지 알려줘"
                            },
                            "id": "tool-call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="검색 완료"),
            ],
            structured_responses={
                "DocumentSelectionResult": {
                    "query_type": "single_document",
                    "selected_document_ids": ["doc-2"],
                    "per_document_queries": {},
                    "retrieval_mode": "hybrid",
                },
                "GroundingCheckResult": {
                    "enough_evidence": True,
                    "needs_deeper_retrieval": False,
                    "needs_clarification": False,
                    "clarification_question": None,
                    "missing_aspects": [],
                },
                "FinalAnswerResult": {
                    "answer": "문서 기반 답변입니다.",
                    "grounded": True,
                },
            },
        )

        result = run_stage5_chatbot(
            {
                "thread_id": "thread-multi-doc",
                "user_message": "랭체인 creat_agnet에서 여기서 포함하는 인자값들 뭐뭐 있는지 알려줘",
                "active_document_ids": ["doc-2", "doc-3"],
                "document_profiles": [
                    {
                        "document_id": "doc-2",
                        "original_filename": "2.pdf",
                        "title": "랭체인 create_agent",
                        "document_type": "기술 문서",
                        "main_topics": ["랭체인", "create_agent"],
                        "short_summary": "create_agent 설명 문서",
                    },
                    {
                        "document_id": "doc-3",
                        "original_filename": "3.pdf",
                        "title": "BM25 검색",
                        "document_type": "기술 문서",
                        "main_topics": ["BM25", "검색"],
                        "short_summary": "검색 문서",
                    },
                ],
            },
            checkpointer=InMemorySaver(),
            llm=fake_llm,
            stage4_runner=stage4_runner,
        )

        self.assertEqual(result["status"], "completed")
        self.assertIsNone(result["interrupt"])
        self.assertEqual(stage4_runner.calls[0]["active_document_ids"], ["doc-2"])

    def test_thread_name_and_technical_query_trigger_document_search(self):
        stage4_runner = _CapturingStage4Runner()
        fake_llm = _FakeToolCallingModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_thread_knowledge",
                            "args": {
                                "query": "랭체인 create_agent에서 포함하는 인자값들 뭐가 있는지 알려줘"
                            },
                            "id": "tool-call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="검색 완료"),
            ],
            structured_responses={
                "DocumentSelectionResult": {
                    "query_type": "single_document",
                    "selected_document_ids": ["doc-2"],
                    "per_document_queries": {},
                    "retrieval_mode": "hybrid",
                },
                "GroundingCheckResult": {
                    "enough_evidence": True,
                    "needs_deeper_retrieval": False,
                    "needs_clarification": False,
                    "clarification_question": None,
                    "missing_aspects": [],
                },
                "FinalAnswerResult": {
                    "answer": "문서 기반 답변입니다.",
                    "grounded": True,
                },
            },
        )

        result = run_stage5_chatbot(
            {
                "thread_id": "thread-multi-doc",
                "thread_name": "랭체인",
                "user_message": "랭체인 create_agent에서 포함하는 인자값들 뭐가 있는지 알려줘",
                "active_document_ids": ["doc-2", "doc-3"],
                "document_profiles": [
                    {
                        "document_id": "doc-2",
                        "original_filename": "2.pdf",
                        "title": "2.pdf",
                        "document_type": "문서",
                        "main_topics": ["랭체인"],
                        "keywords": ["create_agent", "middleware"],
                        "short_summary": "랭체인 create_agent 관련 문서",
                    },
                    {
                        "document_id": "doc-3",
                        "original_filename": "3.pdf",
                        "title": "3.pdf",
                        "document_type": "문서",
                        "main_topics": ["졸업논문"],
                        "keywords": ["심사", "절차"],
                        "short_summary": "졸업논문 제출 절차 문서",
                    },
                ],
            },
            checkpointer=InMemorySaver(),
            llm=fake_llm,
            stage4_runner=stage4_runner,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(stage4_runner.calls), 1)
        self.assertEqual(stage4_runner.calls[0]["active_document_ids"], ["doc-2"])
        self.assertEqual(stage4_runner.calls[0]["retrieval_mode"], "hybrid")
        self.assertEqual(result["retrieval_mode"], "hybrid")
        self.assertEqual(result["citations"][0]["document_id"], "doc-2")

    def test_explicit_multi_document_summary_uses_profile_only_answer(self):
        stage4_runner = _CapturingStage4Runner()
        fake_llm = _FakeToolCallingModel(
            [
                AIMessage(
                    content="1번 문서는 피부 질환 논문이고 2번 문서는 랭체인 가이드입니다."
                ),
            ],
            structured_responses={
                "DocumentSelectionResult": {
                    "query_type": "multi_document",
                    "selected_document_ids": ["doc-1", "doc-2"],
                    "per_document_queries": {},
                    "answer_strategy": "profile_only",
                },
                "GroundingDecisionResult": {
                    "action": "answer",
                    "clarification_question": None,
                },
                "FinalAnswerResult": {
                    "answer": "unused",
                    "grounded": True,
                },
            },
        )

        result = run_stage5_chatbot(
            {
                "thread_id": "thread-explicit-multi-doc",
                "user_message": "1번 문서와 2번 문서 설명해줘",
                "active_document_ids": ["doc-1", "doc-2"],
                "document_profiles": [
                    {
                        "document_id": "doc-1",
                        "original_filename": "1.pdf",
                        "title": "피부 질환 논문",
                        "short_summary": "피부 질환 분류를 위한 멀티모달 VLM 연구",
                    },
                    {
                        "document_id": "doc-2",
                        "original_filename": "2.pdf",
                        "title": "랭체인 가이드",
                        "short_summary": "AI 에이전트 구축 및 랭체인 가이드",
                    },
                ],
            },
            checkpointer=InMemorySaver(),
            llm=fake_llm,
            stage4_runner=stage4_runner,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(stage4_runner.calls, [])
        self.assertEqual(result["debug_trace"]["answer_strategy"], "profile_only")
        self.assertEqual(result["debug_trace"]["tool_calls"], [])
        self.assertIn("1번 문서는 피부 질환 논문", str(result["final_answer"]))

    def test_single_document_summary_uses_profile_only_answer_from_llm(self):
        stage4_runner = _CapturingStage4Runner()
        fake_llm = _FakeToolCallingModel(
            [
                AIMessage(
                    content="1번 문서는 피부 질환 분류용 멀티모달 VLM 연구를 다루는 논문입니다."
                ),
            ],
            structured_responses={
                "DocumentSelectionResult": {
                    "query_type": "single_document",
                    "selected_document_ids": ["doc-1"],
                    "per_document_queries": {},
                    "answer_strategy": "profile_only",
                },
                "GroundingDecisionResult": {
                    "action": "answer",
                    "clarification_question": None,
                },
                "FinalAnswerResult": {
                    "answer": "unused",
                    "grounded": True,
                },
            },
        )

        result = run_stage5_chatbot(
            {
                "thread_id": "thread-single-profile-only",
                "user_message": "1번 문서 설명",
                "active_document_ids": ["doc-1", "doc-2", "doc-3"],
                "document_profiles": [
                    {
                        "document_id": "doc-1",
                        "original_filename": "1.pdf",
                        "title": "피부 질환 논문",
                        "document_type": "학술 논문",
                        "short_summary": "피부 질환 분류용 멀티모달 VLM 연구",
                    },
                    {
                        "document_id": "doc-2",
                        "original_filename": "2.pdf",
                        "title": "랭체인 가이드",
                    },
                    {
                        "document_id": "doc-3",
                        "original_filename": "3.pdf",
                        "title": "졸업논문 절차",
                    },
                ],
            },
            checkpointer=InMemorySaver(),
            llm=fake_llm,
            stage4_runner=stage4_runner,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(stage4_runner.calls, [])
        self.assertEqual(result["debug_trace"]["selection_source"], "llm")
        self.assertEqual(result["debug_trace"]["answer_strategy"], "profile_only")
        self.assertEqual(result["debug_trace"]["selected_document_ids"], ["doc-1"])

    def test_explicit_document_tool_queries_are_scoped_to_each_document(self):
        stage4_runner = _CapturingStage4Runner()
        fake_llm = _FakeToolCallingModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_thread_knowledge",
                            "args": {"query": "1번 문서 근거와 페이지를 설명"},
                            "id": "tool-call-1",
                            "type": "tool_call",
                        },
                        {
                            "name": "search_thread_knowledge",
                            "args": {"query": "2번 문서 근거와 페이지를 설명"},
                            "id": "tool-call-2",
                            "type": "tool_call",
                        },
                        {
                            "name": "search_thread_knowledge",
                            "args": {"query": "3번 문서 근거와 페이지를 설명"},
                            "id": "tool-call-3",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="세 문서 설명입니다."),
            ],
            structured_responses={
                "DocumentSelectionResult": {
                    "query_type": "multi_document",
                    "selected_document_ids": ["doc-1", "doc-2", "doc-3"],
                    "per_document_queries": {},
                    "retrieval_mode": "dense",
                    "answer_strategy": "retrieve_chunks",
                },
                "GroundingDecisionResult": {
                    "action": "answer",
                    "clarification_question": None,
                },
                "FinalAnswerResult": {
                    "answer": "세 문서 설명입니다.",
                    "grounded": True,
                },
            },
        )

        result = run_stage5_chatbot(
            {
                "thread_id": "thread-explicit-tool-scoping",
                "user_message": "1번 2번 3번 문서의 근거와 페이지를 각각 설명해줘",
                "active_document_ids": ["doc-1", "doc-2", "doc-3"],
                "document_profiles": [
                    {
                        "document_id": "doc-1",
                        "original_filename": "1.pdf",
                        "title": "피부 질환 논문",
                        "main_topics": ["피부 질환 분류"],
                    },
                    {
                        "document_id": "doc-2",
                        "original_filename": "2.pdf",
                        "title": "랭체인 가이드",
                        "main_topics": ["create_agent"],
                    },
                    {
                        "document_id": "doc-3",
                        "original_filename": "3.pdf",
                        "title": "졸업논문 제출 절차",
                        "main_topics": ["논문 제출"],
                    },
                ],
            },
            checkpointer=InMemorySaver(),
            llm=fake_llm,
            stage4_runner=stage4_runner,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            [call["active_document_ids"] for call in stage4_runner.calls],
            [["doc-1"], ["doc-2"], ["doc-3"]],
        )
        self.assertEqual(
            [call["use_per_document_search"] for call in stage4_runner.calls],
            [False, False, False],
        )
        self.assertEqual(
            [call["query"] for call in stage4_runner.calls],
            [
                "1번 문서 근거와 페이지를 설명",
                "2번 문서 근거와 페이지를 설명",
                "3번 문서 근거와 페이지를 설명",
            ],
        )

    def test_llm_multi_document_selection_passes_per_document_queries(self):
        stage4_runner = _CapturingStage4Runner()
        fake_llm = _FakeToolCallingModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_thread_knowledge",
                            "args": {"query": "두 문서 핵심 차이를 정리해줘"},
                            "id": "tool-call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="검색 완료"),
            ],
            structured_responses={
                "DocumentSelectionResult": {
                    "query_type": "comparison",
                    "selected_document_ids": ["doc-2", "doc-3"],
                    "per_document_queries": {
                        "doc-2": "2번 문서 핵심 요약",
                        "doc-3": "3번 문서 핵심 요약",
                    },
                    "retrieval_mode": "hybrid",
                    "answer_strategy": "retrieve_chunks",
                },
                "GroundingDecisionResult": {
                    "action": "answer",
                    "clarification_question": None,
                },
                "FinalAnswerResult": {
                    "answer": "두 문서 차이 요약입니다.",
                    "grounded": True,
                },
            },
        )

        result = run_stage5_chatbot(
            {
                "thread_id": "thread-multi-doc",
                "user_message": "두 문서의 근거와 세부 차이를 정리해줘",
                "active_document_ids": ["doc-2", "doc-3"],
                "document_profiles": [
                    {
                        "document_id": "doc-2",
                        "original_filename": "2.pdf",
                        "title": "랭체인 메모리",
                        "document_type": "기술 문서",
                        "main_topics": ["랭체인", "메모리"],
                        "keywords": ["checkpointer", "summary"],
                        "short_summary": "랭체인 메모리 문서",
                    },
                    {
                        "document_id": "doc-3",
                        "original_filename": "3.pdf",
                        "title": "졸업논문 제출",
                        "document_type": "행정 문서",
                        "main_topics": ["졸업논문", "제출"],
                        "keywords": ["절차", "심사"],
                        "short_summary": "졸업논문 제출 문서",
                    },
                ],
            },
            checkpointer=InMemorySaver(),
            llm=fake_llm,
            stage4_runner=stage4_runner,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(stage4_runner.calls[0]["active_document_ids"], ["doc-2", "doc-3"])
        self.assertTrue(stage4_runner.calls[0]["use_per_document_search"])
        self.assertTrue(stage4_runner.calls[0]["enable_rerank"])
        self.assertEqual(stage4_runner.calls[0]["retrieval_mode"], "hybrid")
        self.assertEqual(
            stage4_runner.calls[0]["document_queries"],
            {
                "doc-2": "2번 문서 핵심 요약",
                "doc-3": "3번 문서 핵심 요약",
            },
        )
        search_trace = result["debug_trace"]["tool_calls"][0]
        self.assertTrue(search_trace["per_document_search_used"])
        self.assertTrue(search_trace["rerank_requested"])
        self.assertEqual(search_trace["retrieval_mode"], "hybrid")

    def test_multi_document_selection_without_queries_builds_profile_anchored_queries(self):
        stage4_runner = _CapturingStage4Runner()
        fake_llm = _FakeToolCallingModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_thread_knowledge",
                            "args": {"query": "1번 2번 3번 문서의 근거를 각각 설명해줘"},
                            "id": "tool-call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="검색 완료"),
            ],
            structured_responses={
                "DocumentSelectionResult": {
                    "query_type": "multi_document",
                    "selected_document_ids": ["doc-1", "doc-2", "doc-3"],
                    "per_document_queries": {},
                    "retrieval_mode": "dense",
                    "answer_strategy": "retrieve_chunks",
                },
                "GroundingDecisionResult": {
                    "action": "answer",
                    "clarification_question": None,
                },
                "FinalAnswerResult": {
                    "answer": "세 문서 설명입니다.",
                    "grounded": True,
                },
            },
        )

        result = run_stage5_chatbot(
            {
                "thread_id": "thread-profile-anchored-queries",
                "user_message": "1번 2번 3번 문서의 근거를 각각 설명해줘",
                "active_document_ids": ["doc-1", "doc-2", "doc-3"],
                "document_profiles": [
                    {
                        "document_id": "doc-1",
                        "original_filename": "1.pdf",
                        "title": "피부 질환 논문",
                        "main_topics": ["질환 분류"],
                    },
                    {
                        "document_id": "doc-2",
                        "original_filename": "2.pdf",
                        "title": "랭체인 가이드",
                        "main_topics": ["create_agent"],
                    },
                    {
                        "document_id": "doc-3",
                        "original_filename": "3.pdf",
                        "title": "졸업논문 절차",
                        "main_topics": ["계획서 제출"],
                    },
                ],
            },
            checkpointer=InMemorySaver(),
            llm=fake_llm,
            stage4_runner=stage4_runner,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(stage4_runner.calls), 1)
        self.assertTrue(stage4_runner.calls[0]["use_per_document_search"])
        self.assertEqual(
            stage4_runner.calls[0]["document_queries"],
            {
                "doc-1": "1번 2번 3번 문서의 근거를 각각 설명해줘 피부 질환 논문 질환 분류",
                "doc-2": "1번 2번 3번 문서의 근거를 각각 설명해줘 랭체인 가이드 create_agent",
                "doc-3": "1번 2번 3번 문서의 근거를 각각 설명해줘 졸업논문 절차 계획서 제출",
            },
        )

    def test_single_document_selection_without_queries_keeps_raw_query(self):
        stage4_runner = _CapturingStage4Runner()
        fake_llm = _FakeToolCallingModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_thread_knowledge",
                            "args": {"query": "랭체인에서 create_agent 사용법"},
                            "id": "tool-call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="검색 완료"),
            ],
            structured_responses={
                "DocumentSelectionResult": {
                    "query_type": "single_document",
                    "selected_document_ids": ["doc-2"],
                    "per_document_queries": {},
                    "retrieval_mode": "hybrid",
                    "answer_strategy": "retrieve_chunks",
                },
                "GroundingDecisionResult": {
                    "action": "answer",
                    "clarification_question": None,
                },
                "FinalAnswerResult": {
                    "answer": "create_agent 설명입니다.",
                    "grounded": True,
                },
            },
        )

        result = run_stage5_chatbot(
            {
                "thread_id": "thread-single-raw-query",
                "thread_name": "랭체인",
                "user_message": "랭체인에서 create_agent 사용법",
                "active_document_ids": ["doc-2", "doc-3"],
                "document_profiles": [
                    {
                        "document_id": "doc-2",
                        "original_filename": "2.pdf",
                        "title": "AI 에이전트 구축 및 실무 적용 가이드",
                        "main_topics": [
                            "AI 에이전트의 개념과 설계 방법",
                            "랭체인과 랭그래프 기술 소개",
                        ],
                    },
                    {
                        "document_id": "doc-3",
                        "original_filename": "3.pdf",
                        "title": "졸업논문 절차",
                        "main_topics": ["계획서 제출"],
                    },
                ],
            },
            checkpointer=InMemorySaver(),
            llm=fake_llm,
            stage4_runner=stage4_runner,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(stage4_runner.calls), 1)
        self.assertEqual(stage4_runner.calls[0]["active_document_ids"], ["doc-2"])
        self.assertEqual(
            stage4_runner.calls[0]["query"],
            "랭체인에서 create_agent 사용법",
        )
        self.assertEqual(stage4_runner.calls[0]["document_queries"], {})
        self.assertEqual(result["debug_trace"]["selected_document_queries"], {})

    def test_insufficient_answer_draft_forces_deterministic_deeper_retrieval(self):
        stage4_runner = _CapturingStage4Runner()
        fake_llm = _FakeToolCallingModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_thread_knowledge",
                            "args": {
                                "query": "1번 논문에서 예시 질환 사진으로 나온 질환"
                            },
                            "id": "tool-call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="현재 연결된 문서에서 질문에 답할 수 있는 근거를 찾지 못했습니다."
                ),
            ],
            structured_responses={
                "GroundingDecisionResult": {
                    "action": "answer",
                    "clarification_question": None,
                },
                "FinalAnswerResult": {
                    "answer": "심화 검색 후 답변입니다.",
                    "grounded": True,
                },
            },
        )

        result = run_stage5_chatbot(
            {
                "thread_id": "thread-force-deeper",
                "user_message": "1번 논문에서 예시 질환 사진으로 나온 질환",
                "active_document_ids": ["doc-1"],
                "document_profiles": [
                    {
                        "document_id": "doc-1",
                        "original_filename": "1.pdf",
                        "title": "피부 질환 논문",
                        "main_topics": ["예시 질환", "피부 질환 분류"],
                    }
                ],
            },
            checkpointer=InMemorySaver(),
            llm=fake_llm,
            stage4_runner=stage4_runner,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(stage4_runner.calls), 2)
        self.assertEqual(
            stage4_runner.calls[-1]["query"],
            "1번 논문에서 예시 질환 사진으로 나온 질환",
        )
        self.assertIn(
            "grounding_check:retrieve_deeper:deterministic",
            result["debug_trace"]["logs"],
        )

    def test_smalltalk_is_answered_without_document_search(self):
        def _unexpected_stage4_runner(**kwargs):
            raise AssertionError("smalltalk must not trigger retrieval")

        result = run_stage5_chatbot(
            {
                "thread_id": "thread-smalltalk",
                "user_message": "안녕",
                "active_document_ids": ["doc-1"],
            },
            checkpointer=InMemorySaver(),
            llm=_FakeToolCallingModel(
                [AIMessage(content="unused")],
                structured_responses={
                    "GroundingCheckResult": {
                        "enough_evidence": True,
                        "needs_deeper_retrieval": False,
                        "needs_clarification": False,
                        "clarification_question": None,
                        "missing_aspects": [],
                    },
                    "FinalAnswerResult": {"answer": "unused", "grounded": True},
                },
            ),
            stage4_runner=_unexpected_stage4_runner,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["citations"], [])
        self.assertIn("안녕하세요", str(result["final_answer"]))

    def test_conversation_memory_uses_saved_user_facts(self):
        checkpointer = InMemorySaver()

        first_turn = run_stage5_chatbot(
            {
                "thread_id": "thread-memory",
                "user_message": "내 이름은 석우야",
                "active_document_ids": ["doc-1"],
            },
            checkpointer=checkpointer,
            llm=_FakeToolCallingModel(
                [AIMessage(content="기억해둘게요.")],
                structured_responses={
                    "GroundingCheckResult": {
                        "enough_evidence": True,
                        "needs_deeper_retrieval": False,
                        "needs_clarification": False,
                        "clarification_question": None,
                        "missing_aspects": [],
                    },
                    "FinalAnswerResult": {"answer": "unused", "grounded": True},
                },
            ),
            stage4_runner=_fake_stage4_runner,
        )
        second_turn = run_stage5_chatbot(
            {
                "thread_id": "thread-memory",
                "user_message": "내 이름 뭐야?",
                "active_document_ids": ["doc-1"],
            },
            checkpointer=checkpointer,
            llm=_FakeToolCallingModel(
                [AIMessage(content="unused")],
                structured_responses={
                    "GroundingCheckResult": {
                        "enough_evidence": True,
                        "needs_deeper_retrieval": False,
                        "needs_clarification": False,
                        "clarification_question": None,
                        "missing_aspects": [],
                    },
                    "FinalAnswerResult": {"answer": "unused", "grounded": True},
                },
            ),
            stage4_runner=_fake_stage4_runner,
        )

        self.assertEqual(first_turn["status"], "completed")
        self.assertEqual(second_turn["status"], "completed")
        self.assertIn("석우", str(second_turn["final_answer"]))

    def test_conversation_memory_uses_saved_nickname(self):
        checkpointer = InMemorySaver()

        run_stage5_chatbot(
            {
                "thread_id": "thread-memory-nickname",
                "user_message": "내 별명은 코덱스테스터야",
                "active_document_ids": ["doc-1"],
            },
            checkpointer=checkpointer,
            llm=_FakeToolCallingModel(
                [AIMessage(content="기억해둘게요.")],
                structured_responses={
                    "GroundingCheckResult": {
                        "enough_evidence": True,
                        "needs_deeper_retrieval": False,
                        "needs_clarification": False,
                        "clarification_question": None,
                        "missing_aspects": [],
                    },
                    "FinalAnswerResult": {"answer": "unused", "grounded": True},
                },
            ),
            stage4_runner=_fake_stage4_runner,
        )
        second_turn = run_stage5_chatbot(
            {
                "thread_id": "thread-memory-nickname",
                "user_message": "내 별명이 뭐라고 했지?",
                "active_document_ids": ["doc-1"],
            },
            checkpointer=checkpointer,
            llm=_FakeToolCallingModel(
                [AIMessage(content="unused")],
                structured_responses={
                    "GroundingCheckResult": {
                        "enough_evidence": True,
                        "needs_deeper_retrieval": False,
                        "needs_clarification": False,
                        "clarification_question": None,
                        "missing_aspects": [],
                    },
                    "FinalAnswerResult": {"answer": "unused", "grounded": True},
                },
            ),
            stage4_runner=_fake_stage4_runner,
        )

        self.assertEqual(second_turn["status"], "completed")
        self.assertIn("코덱스테스터", str(second_turn["final_answer"]))

    def test_followup_question_reuses_previous_document_scope(self):
        checkpointer = InMemorySaver()
        stage4_runner = _CapturingStage4Runner()

        first_llm = _FakeToolCallingModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_thread_knowledge",
                            "args": {
                                "query": "2번 문서 기준으로 create_agent 필수 인자만 짧게 말해줘"
                            },
                            "id": "tool-call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="model과 tools가 핵심입니다."),
            ],
            structured_responses={
                "DocumentSelectionResult": {
                    "query_type": "single_document",
                    "selected_document_ids": ["doc-2"],
                    "per_document_queries": {},
                    "retrieval_mode": "hybrid",
                },
                "GroundingCheckResult": {
                    "enough_evidence": True,
                    "needs_deeper_retrieval": False,
                    "needs_clarification": False,
                    "clarification_question": None,
                    "missing_aspects": [],
                },
                "FinalAnswerResult": {
                    "answer": "model과 tools가 핵심입니다.",
                    "grounded": True,
                },
            },
        )
        second_llm = _FakeToolCallingModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_thread_knowledge",
                            "args": {"query": "그중 middleware는 왜 쓰는거야?"},
                            "id": "tool-call-2",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="middleware는 실행 정책과 검증을 끼워 넣기 위해 씁니다."),
            ],
            structured_responses={
                "DocumentSelectionResult": {
                    "query_type": "single_document",
                    "selected_document_ids": ["doc-2"],
                    "per_document_queries": {},
                    "retrieval_mode": "hybrid",
                },
                "GroundingCheckResult": {
                    "enough_evidence": True,
                    "needs_deeper_retrieval": False,
                    "needs_clarification": False,
                    "clarification_question": None,
                    "missing_aspects": [],
                },
                "FinalAnswerResult": {
                    "answer": "middleware는 실행 정책과 검증을 끼워 넣기 위해 씁니다.",
                    "grounded": True,
                },
            },
        )

        run_stage5_chatbot(
            {
                "thread_id": "thread-followup",
                "user_message": "2번 문서 기준으로 create_agent 필수 인자만 짧게 말해줘",
                "active_document_ids": ["doc-2", "doc-3"],
                "document_profiles": [
                    {
                        "document_id": "doc-2",
                        "title": "AI 에이전트 구축 및 실무 적용 가이드",
                        "document_type": "기술서/실무 가이드",
                        "main_topics": ["랭체인", "랭그래프", "에이전트"],
                        "short_summary": "랭체인과 에이전트 구현을 다루는 문서",
                        "original_filename": "2.pdf",
                    },
                    {
                        "document_id": "doc-3",
                        "title": "졸업논문 실시계획",
                        "document_type": "행정 문서",
                        "main_topics": ["졸업논문", "계획서 제출", "심사 절차"],
                        "short_summary": "졸업논문 절차 문서",
                        "original_filename": "3.pdf",
                    },
                ],
            },
            checkpointer=checkpointer,
            llm=first_llm,
            stage4_runner=stage4_runner,
        )
        second_turn = run_stage5_chatbot(
            {
                "thread_id": "thread-followup",
                "user_message": "그중 middleware는 왜 쓰는거야?",
                "active_document_ids": ["doc-2", "doc-3"],
                "document_profiles": [
                    {
                        "document_id": "doc-2",
                        "title": "AI 에이전트 구축 및 실무 적용 가이드",
                        "document_type": "기술서/실무 가이드",
                        "main_topics": ["랭체인", "랭그래프", "에이전트"],
                        "short_summary": "랭체인과 에이전트 구현을 다루는 문서",
                        "original_filename": "2.pdf",
                    },
                    {
                        "document_id": "doc-3",
                        "title": "졸업논문 실시계획",
                        "document_type": "행정 문서",
                        "main_topics": ["졸업논문", "계획서 제출", "심사 절차"],
                        "short_summary": "졸업논문 절차 문서",
                        "original_filename": "3.pdf",
                    },
                ],
            },
            checkpointer=checkpointer,
            llm=second_llm,
            stage4_runner=stage4_runner,
        )

        self.assertEqual(second_turn["status"], "completed")
        self.assertEqual(len(stage4_runner.calls), 2)
        self.assertEqual(stage4_runner.calls[-1]["active_document_ids"], ["doc-2"])

    def test_deeper_retrieval_keeps_mode_and_expands_multi_document_candidates(self):
        stage4_runner = _CapturingStage4Runner()
        fake_llm = _FakeToolCallingModel(
            [AIMessage(content="초안 답변입니다.")],
            structured_responses={
                "DocumentSelectionResult": {
                    "query_type": "comparison",
                    "selected_document_ids": ["doc-1", "doc-2", "doc-3"],
                    "per_document_queries": {
                        "doc-1": "1번 문서 설명",
                        "doc-2": "2번 문서 설명",
                        "doc-3": "3번 문서 설명",
                    },
                    "retrieval_mode": "dense",
                    "answer_strategy": "retrieve_chunks",
                },
                "GroundingDecisionResult": {
                    "action": "retrieve_deeper",
                    "clarification_question": None,
                },
                "FinalAnswerResult": {
                    "answer": "심화 검색 후 답변입니다.",
                    "grounded": True,
                },
            },
        )

        result = run_stage5_chatbot(
            {
                "thread_id": "thread-deeper-retrieval",
                "thread_default_retrieval_mode": "hybrid",
                "user_message": "1번 2번 3번 문서의 근거를 비교해서 설명해줘",
                "active_document_ids": ["doc-1", "doc-2", "doc-3"],
                "document_profiles": [
                    {
                        "document_id": "doc-1",
                        "original_filename": "1.pdf",
                        "title": "피부 질환 논문",
                    },
                    {
                        "document_id": "doc-2",
                        "original_filename": "2.pdf",
                        "title": "랭체인 가이드",
                    },
                    {
                        "document_id": "doc-3",
                        "original_filename": "3.pdf",
                        "title": "졸업논문 절차",
                    },
                ],
            },
            checkpointer=InMemorySaver(),
            llm=fake_llm,
            stage4_runner=stage4_runner,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(stage4_runner.calls), 1)
        deep_call = stage4_runner.calls[0]
        self.assertEqual(deep_call["retrieval_mode"], "dense")
        self.assertTrue(deep_call["use_per_document_search"])
        self.assertEqual(deep_call["per_document_top_k"], 8)
        self.assertTrue(deep_call["enable_rerank"])
        self.assertGreaterEqual(deep_call["top_k"], 10)
        self.assertGreaterEqual(deep_call["fetch_k"], deep_call["top_k"])

    def test_grounding_check_applies_context_window_loader_blocks(self):
        def _fake_context_window_loader(**kwargs):
            return [
                {
                    "document_id": "doc-1",
                    "parent_id": "parent-1",
                    "matched_chunk_ids": ["chunk-1"],
                    "window_chunk_ids": ["chunk-0", "chunk-1", "chunk-2"],
                    "page_start": 3,
                    "page_end": 3,
                    "section_title": "1. 소개",
                    "context_text": "확장된 부모/윈도우 문맥입니다.",
                }
            ]

        grounding_node = build_grounding_check_node(
            llm=_FakeToolCallingModel(
                [],
                structured_responses={
                    "GroundingDecisionResult": {
                        "action": "answer",
                        "clarification_question": None,
                    }
                },
            ),
            context_window_loader=_fake_context_window_loader,
        )

        updates = grounding_node(
            {
                "thread_id": "thread-window",
                "query_analysis": {
                    "query_text": "이 문서 핵심을 설명해줘",
                    "query_kind": "document_grounded",
                },
                "retrieval_policy": {
                    "use_context_window": True,
                    "context_window_size": 1,
                },
                "retrieval_hits": [
                    {
                        "document_id": "doc-1",
                        "chunk_id": "chunk-1",
                        "parent_id": "parent-1",
                        "primary_page": 3,
                        "section_title": "1. 소개",
                        "text": "원본 child chunk",
                    }
                ],
                "document_profiles": [
                    {
                        "document_id": "doc-1",
                        "original_filename": "1.pdf",
                        "title": "테스트 문서",
                    }
                ],
            }
        )

        self.assertEqual(updates["grounding_decision"]["action"], "answer")
        self.assertTrue(updates["expanded_context_blocks"])
        self.assertIn(
            "확장된 부모/윈도우 문맥입니다.",
            updates["expanded_context_blocks"][0],
        )

    def test_open_domain_unrelated_question_skips_retrieval(self):
        def _unexpected_stage4_runner(**kwargs):
            raise AssertionError("open-domain unrelated query must not trigger retrieval")

        result = run_stage5_chatbot(
            {
                "thread_id": "thread-general",
                "user_message": "서울 날씨 어때?",
                "active_document_ids": ["doc-1"],
                "document_profiles": [
                    {
                        "document_id": "doc-1",
                        "title": "랭그래프 문서",
                        "document_type": "기술 문서",
                        "main_topics": ["랭그래프", "RAG", "체크포인터"],
                        "short_summary": "랭그래프 기반 문서",
                    }
                ],
            },
            checkpointer=InMemorySaver(),
            llm=_FakeToolCallingModel(
                [AIMessage(content="서울은 오늘 맑을 가능성이 높습니다.")],
                structured_responses={
                    "GroundingCheckResult": {
                        "enough_evidence": True,
                        "needs_deeper_retrieval": False,
                        "needs_clarification": False,
                        "clarification_question": None,
                        "missing_aspects": [],
                    },
                    "FinalAnswerResult": {"answer": "unused", "grounded": True},
                },
            ),
            stage4_runner=_unexpected_stage4_runner,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["citations"], [])
