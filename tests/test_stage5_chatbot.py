import unittest

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from backend.stage5_chatbot.graph import build_graph
from backend.stage5_chatbot.service import run_stage5_chatbot
from backend.stage5_chatbot.tools import build_stage5_tools


def _fake_stage4_runner(
    *,
    query,
    room_id,
    active_document_ids,
    collection_name=None,
    retrieval_mode=None,
    **_,
):
    return {
        "status": "completed",
        "query": query,
        "room_id": room_id,
        "active_document_ids": list(active_document_ids),
        "collection_name": collection_name,
        "retrieval_mode": retrieval_mode or "dense",
        "retrieved_count": 1,
        "retrievals": [
            {
                "document_id": active_document_ids[0] if active_document_ids else "doc-1",
                "chunk_id": "chunk-1",
                "parent_id": "parent-1",
                "primary_page": 3,
                "section_title": "1. 소개",
                "asset_relative_path": None,
                "text": "이 문서는 피부 질환 분류 모델의 구조와 실험 결과를 설명합니다.",
                "chunk_type": "text",
                "score": 0.91,
            }
        ],
    }


class _FakeToolCallingModel:
    def __init__(self, responses, *, structured_responses=None):
        self._responses = list(responses)
        self._structured_responses = dict(structured_responses or {})

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        configured_response = self._structured_responses.get(schema.__name__)
        if configured_response is None:
            raise AssertionError(f"structured response was not configured for {schema.__name__}")
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

    def test_run_stage5_chatbot_runs_tool_loop_and_returns_citations(self):
        fake_llm = _FakeToolCallingModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_room_knowledge",
                            "args": {"query": "이 문서를 요약해줘"},
                            "id": "tool-call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content=""),
            ],
            structured_responses={
                "GroundingCheckResult": {
                    "enough_evidence": True,
                    "needs_deeper_retrieval": False,
                    "needs_clarification": False,
                    "clarification_question": None,
                    "missing_aspects": [],
                },
                "FinalAnswerResult": {
                    "answer": "핵심은 피부 질환 분류 모델 구조와 실험 결과입니다. 3페이지 소개 섹션을 참고하세요.",
                    "grounded": True,
                },
            },
        )

        result = run_stage5_chatbot(
            {
                "room_id": "room-alpha",
                "thread_id": "thread-alpha",
                "user_message": "이 문서를 요약해줘",
                "active_document_ids": ["doc-1"],
                "collection_name": "rag_chat_hybrid",
            },
            checkpointer=InMemorySaver(),
            llm=fake_llm,
            stage4_runner=_fake_stage4_runner,
        )

        self.assertEqual(result["status"], "completed")
        self.assertIn("피부 질환 분류 모델", str(result["final_answer"]))
        self.assertEqual(result["retrieval_mode"], "dense")
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["document_id"], "doc-1")
        self.assertEqual(result["citations"][0]["chunk_id"], "chunk-1")

    def test_run_stage5_chatbot_returns_safe_fallback_when_structured_answer_is_not_grounded(self):
        fake_llm = _FakeToolCallingModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_room_knowledge",
                            "args": {"query": "근거 없는 답을 만들지 마"},
                            "id": "tool-call-2",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content=""),
            ],
            structured_responses={
                "GroundingCheckResult": {
                    "enough_evidence": True,
                    "needs_deeper_retrieval": False,
                    "needs_clarification": False,
                    "clarification_question": None,
                    "missing_aspects": [],
                },
                "FinalAnswerResult": {
                    "answer": "근거가 부족하지만 추측 답변입니다.",
                    "grounded": False,
                },
            },
        )

        result = run_stage5_chatbot(
            {
                "room_id": "room-alpha",
                "thread_id": "thread-fallback",
                "user_message": "근거가 없으면 모른다고 말해줘",
                "active_document_ids": ["doc-1"],
            },
            checkpointer=InMemorySaver(),
            llm=fake_llm,
            stage4_runner=_fake_stage4_runner,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            result["final_answer"],
            "현재 연결된 문서에서 질문에 답할 수 있는 근거를 찾지 못했습니다.",
        )
        self.assertEqual(len(result["citations"]), 1)

    def test_run_stage5_chatbot_resolves_context_window_and_visual_assets(self):
        def _fake_asset_stage4_runner(
            *,
            query,
            room_id,
            active_document_ids,
            collection_name=None,
            retrieval_mode=None,
            **_,
        ):
            return {
                "status": "completed",
                "query": query,
                "room_id": room_id,
                "active_document_ids": list(active_document_ids),
                "collection_name": collection_name,
                "retrieval_mode": retrieval_mode or "dense",
                "retrieved_count": 1,
                "retrievals": [
                    {
                        "document_id": active_document_ids[0]
                        if active_document_ids
                        else "doc-1",
                        "chunk_id": "figure-0001",
                        "parent_id": "parent-1",
                        "primary_page": 5,
                        "section_title": "2. 결과",
                        "asset_relative_path": "figures/page_5_figure_1.png",
                        "text": "Figure summary text",
                        "chunk_type": "figure",
                        "score": 0.93,
                    }
                ],
            }

        def _fake_context_loader(*, room_id, active_document_ids, chunk_ids, window_size):
            return [
                {
                    "document_id": active_document_ids[0],
                    "parent_id": "parent-1",
                    "section_title": "2. 결과",
                    "page_start": 5,
                    "page_end": 5,
                    "heading_path": ["2. 결과"],
                    "matched_chunk_ids": ["figure-0001"],
                    "window_chunk_ids": ["text-0008", "figure-0001", "text-0009"],
                    "context_text": "주변 문맥입니다.",
                    "expansion_mode": "postgres_parent_window",
                }
            ]

        def _fake_visual_asset_loader(*, room_id, active_document_ids, asset_refs):
            return [
                {
                    "asset_ref": asset_refs[0],
                    "document_id": active_document_ids[0],
                    "chunk_id": "figure-0001",
                    "asset_kind": "figure",
                    "relative_path": "figures/page_5_figure_1.png",
                    "asset_stage": "stage2",
                    "page": 5,
                    "caption": "테스트 그림",
                    "summary_text": "Figure summary text",
                    "heading_path": ["2. 결과"],
                    "pages": [5],
                }
            ]

        fake_llm = _FakeToolCallingModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_room_knowledge",
                            "args": {"query": "그림 내용을 설명해줘"},
                            "id": "tool-call-figure-search",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "expand_context_window",
                            "args": {"chunk_ids": ["doc-1:figure-0001"]},
                            "id": "tool-call-window",
                            "type": "tool_call",
                        },
                        {
                            "name": "load_visual_asset",
                            "args": {"asset_ref": "doc-1:figure-0001"},
                            "id": "tool-call-asset",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content=""),
            ],
            structured_responses={
                "GroundingCheckResult": {
                    "enough_evidence": True,
                    "needs_deeper_retrieval": False,
                    "needs_clarification": False,
                    "clarification_question": None,
                    "missing_aspects": [],
                },
                "FinalAnswerResult": {
                    "answer": "5페이지 그림은 결과 비교를 보여줍니다.",
                    "grounded": True,
                },
            },
        )

        result = run_stage5_chatbot(
            {
                "room_id": "room-alpha",
                "thread_id": "thread-asset",
                "user_message": "그림 내용을 설명해줘",
                "active_document_ids": ["doc-1"],
                "collection_name": "rag_chat_hybrid",
                "_context_window_loader": _fake_context_loader,
                "_visual_asset_loader": _fake_visual_asset_loader,
            },
            checkpointer=InMemorySaver(),
            llm=fake_llm,
            stage4_runner=_fake_asset_stage4_runner,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["asset_ref"], "doc-1:figure-0001")
        self.assertEqual(len(result["visual_assets"]), 1)
        self.assertEqual(result["visual_assets"][0]["relative_path"], "figures/page_5_figure_1.png")

    def test_run_stage5_chatbot_interrupts_when_grounding_llm_requests_clarification(self):
        fake_llm = _FakeToolCallingModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_room_knowledge",
                            "args": {"query": "이 문서의 결과가 좋은가?"},
                            "id": "tool-call-3",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content=""),
            ],
            structured_responses={
                "GroundingCheckResult": {
                    "enough_evidence": False,
                    "needs_deeper_retrieval": False,
                    "needs_clarification": True,
                    "clarification_question": "정확도 기준인지, 효율 기준인지 알려주세요.",
                    "missing_aspects": ["평가 기준"],
                },
                "FinalAnswerResult": {"answer": "unused", "grounded": True},
            },
        )

        result = run_stage5_chatbot(
            {
                "room_id": "room-alpha",
                "thread_id": "thread-clarify-grounding",
                "user_message": "이 문서의 결과가 좋은가?",
                "active_document_ids": ["doc-1"],
            },
            checkpointer=InMemorySaver(),
            llm=fake_llm,
            stage4_runner=_fake_stage4_runner,
        )

        self.assertEqual(result["status"], "interrupted")
        self.assertIsNotNone(result["interrupt"])
        self.assertEqual(
            result["interrupt"]["question"],
            "정확도 기준인지, 효율 기준인지 알려주세요.",
        )

    def test_run_stage5_chatbot_interrupts_when_no_documents_are_connected(self):
        result = run_stage5_chatbot(
            {
                "room_id": "room-empty",
                "thread_id": "thread-empty",
                "user_message": "이 문서의 핵심을 알려줘",
                "active_document_ids": [],
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

        self.assertEqual(result["status"], "interrupted")
        self.assertIsNotNone(result["interrupt"])
        self.assertEqual(result["interrupt"]["kind"], "clarification")
