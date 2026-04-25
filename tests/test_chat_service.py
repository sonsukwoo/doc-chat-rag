from __future__ import annotations

import unittest
from contextlib import contextmanager
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from backend.services.chat_service import load_thread_chat_view


class ChatServiceTests(unittest.TestCase):
    def test_load_thread_chat_view_restores_pending_interrupt_from_checkpoint(self):
        raw_messages = [
            HumanMessage(content="Figure 4 설명 좀 해줘"),
        ]

        class _FakeInterrupt:
            def __init__(self, value):
                self.value = value

        class _FakeCheckpointer:
            def get_tuple(self, config):
                return {
                    "checkpoint": {
                        "channel_values": {
                            "messages": raw_messages,
                            "needs_clarification": True,
                            "clarification_response": None,
                            "clarification_payload": None,
                            "citations": [{"document_id": "doc-1", "chunk_id": "chunk-1"}],
                            "evidence_chunks": [
                                {
                                    "document_id": "doc-1",
                                    "chunk_id": "chunk-1",
                                    "text_excerpt": "Figure 4 관련 청크",
                                }
                            ],
                            "retrieval_mode": "hybrid",
                            "debug_trace": {
                                "logs": ["grounding_check:clarify:llm"],
                                "executed_retrieval_mode": "hybrid",
                            },
                        }
                    },
                    "pending_writes": [
                        (
                            "task-1",
                            "__interrupt__",
                            [
                                _FakeInterrupt(
                                    {
                                        "kind": "clarification",
                                        "question": "Figure 4가 어느 문서에 있는 그림인지 알려주세요.",
                                        "reason": "질문 대상 문서를 먼저 확정해야 합니다.",
                                    }
                                )
                            ],
                        )
                    ],
                }

        @contextmanager
        def _fake_checkpointer_context():
            yield _FakeCheckpointer()

        with patch(
            "backend.services.chat_service.get_thread_detail",
            return_value={
                "thread_id": "thread-1",
                "thread_name": "테스트 스레드",
                "collection_name": "rag_chat_hybrid_thread-1",
                "description": None,
                "default_retrieval_mode": "hybrid",
                "metadata": {"lifecycle_status": "ready"},
                "active_document_ids": ["doc-1", "doc-2"],
                "document_count": 2,
                "created_at": None,
                "updated_at": None,
                "archived_at": None,
            },
        ), patch(
            "backend.services.chat_service.stage5_checkpointer_context",
            _fake_checkpointer_context,
        ):
            payload = load_thread_chat_view("thread-1")

        self.assertIsNotNone(payload["interrupt"])
        self.assertEqual(
            payload["interrupt"]["question"],
            "Figure 4가 어느 문서에 있는 그림인지 알려주세요.",
        )
        self.assertEqual(payload["messages"][-1]["kind"], "interrupt")
        self.assertEqual(payload["messages"][-1]["retrieval_mode"], "hybrid")
        self.assertEqual(len(payload["messages"][-1]["citations"]), 1)
        self.assertEqual(len(payload["messages"][-1]["evidence_chunks"]), 1)
        self.assertIsNotNone(payload["messages"][-1]["debug_trace"])

    def test_load_thread_chat_view_does_not_expose_stale_interrupt_payload(self):
        raw_messages = [
            HumanMessage(content="1번 문서 설명"),
            AIMessage(
                content="문서 요약 답변입니다.",
                name="stage5_legacy_answer",
                additional_kwargs={
                    "thread_chat": {
                        "citations": [],
                        "evidence_chunks": [],
                        "retrieval_mode": "dense",
                    }
                },
            ),
        ]

        class _FakeCheckpointer:
            def get_tuple(self, config):
                return {
                    "checkpoint": {
                        "channel_values": {
                            "messages": raw_messages,
                            "needs_clarification": False,
                            "clarification_response": None,
                            "clarification_payload": {
                                "kind": "clarification",
                                "question": "어느 문서인가요?",
                            },
                        }
                    }
                }

        @contextmanager
        def _fake_checkpointer_context():
            yield _FakeCheckpointer()

        with patch(
            "backend.services.chat_service.get_thread_detail",
            return_value={
                "thread_id": "thread-1",
                "thread_name": "테스트 스레드",
                "collection_name": "rag_chat_hybrid_thread-1",
                "description": None,
                "default_retrieval_mode": "hybrid",
                "metadata": {"lifecycle_status": "ready"},
                "active_document_ids": ["doc-1"],
                "document_count": 1,
                "created_at": None,
                "updated_at": None,
                "archived_at": None,
            },
        ), patch(
            "backend.services.chat_service.stage5_checkpointer_context",
            _fake_checkpointer_context,
        ):
            payload = load_thread_chat_view("thread-1")

        self.assertIsNone(payload["interrupt"])
        self.assertEqual(payload["messages"][-1]["kind"], "answer")

    def test_load_thread_chat_view_keeps_persisted_interrupt_message_in_history(self):
        raw_messages = [
            HumanMessage(content="Figure 4 설명 좀 해줘"),
            AIMessage(
                content="어느 문서의 Figure 4인지 알려주세요.\n\n문서 범위를 먼저 확정해야 합니다.",
                name="stage5_clarification",
                additional_kwargs={
                    "thread_chat": {
                        "kind": "interrupt",
                        "created_at": "2026-04-21T00:00:00+00:00",
                    }
                },
            ),
            HumanMessage(content="새 질문입니다."),
        ]

        class _FakeCheckpointer:
            def get_tuple(self, config):
                return {
                    "checkpoint": {
                        "channel_values": {
                            "messages": raw_messages,
                            "needs_clarification": False,
                            "clarification_response": None,
                            "clarification_payload": None,
                        }
                    }
                }

        @contextmanager
        def _fake_checkpointer_context():
            yield _FakeCheckpointer()

        with patch(
            "backend.services.chat_service.get_thread_detail",
            return_value={
                "thread_id": "thread-1",
                "thread_name": "테스트 스레드",
                "collection_name": "rag_chat_hybrid_thread-1",
                "description": None,
                "default_retrieval_mode": "hybrid",
                "metadata": {"lifecycle_status": "ready"},
                "active_document_ids": ["doc-1"],
                "document_count": 1,
                "created_at": None,
                "updated_at": None,
                "archived_at": None,
            },
        ), patch(
            "backend.services.chat_service.stage5_checkpointer_context",
            _fake_checkpointer_context,
        ):
            payload = load_thread_chat_view("thread-1")

        self.assertEqual(payload["messages"][1]["kind"], "interrupt")
        self.assertIn("어느 문서의 Figure 4인지 알려주세요.", payload["messages"][1]["content"])

    def test_load_thread_chat_view_hides_intermediate_ai_drafts(self):
        final_trace = {
            "model": "openai:gpt-4.1-mini",
            "selected_document_ids": ["doc-2"],
            "tool_calls": [{"name": "search_thread_knowledge", "label": "문서 검색"}],
            "logs": ["load_request_context", "compose_answer_with_citations"],
        }

        raw_messages = [
            HumanMessage(content="1번 2번 3번 문서에 대해 설명"),
            AIMessage(content="중간 초안 답변입니다."),
            AIMessage(
                content="최종 답변입니다.",
                name="stage5_final_answer",
                additional_kwargs={
                    "thread_chat": {
                        "citations": [],
                        "evidence_chunks": [],
                        "retrieval_mode": "hybrid",
                        "debug_trace": final_trace,
                    }
                },
            ),
        ]

        class _FakeCheckpointer:
            def get_tuple(self, config):
                return {
                    "checkpoint": {
                        "channel_values": {
                            "messages": raw_messages,
                            "needs_clarification": False,
                            "clarification_response": None,
                            "clarification_payload": None,
                        }
                    }
                }

        @contextmanager
        def _fake_checkpointer_context():
            yield _FakeCheckpointer()

        with patch(
            "backend.services.chat_service.get_thread_detail",
            return_value={
                "thread_id": "thread-1",
                "thread_name": "테스트 스레드",
                "collection_name": "rag_chat_hybrid_thread-1",
                "description": None,
                "default_retrieval_mode": "hybrid",
                "metadata": {"lifecycle_status": "ready"},
                "active_document_ids": ["doc-2"],
                "document_count": 1,
                "created_at": None,
                "updated_at": None,
                "archived_at": None,
            },
        ), patch(
            "backend.services.chat_service.stage5_checkpointer_context",
            _fake_checkpointer_context,
        ):
            payload = load_thread_chat_view("thread-1")

        self.assertEqual(len(payload["messages"]), 2)
        assistant_message = payload["messages"][-1]
        self.assertEqual(assistant_message["content"], "최종 답변입니다.")
        self.assertEqual(assistant_message["retrieval_mode"], "hybrid")

    def test_load_thread_chat_view_preserves_trace_metadata_on_final_message(self):
        final_trace = {
            "model": "openai:gpt-4.1-mini",
            "selected_document_ids": ["doc-2"],
            "tool_calls": [{"name": "search_thread_knowledge", "label": "문서 검색"}],
            "logs": ["load_request_context", "compose_answer_with_citations"],
        }

        raw_messages = [
            HumanMessage(content="질문"),
            AIMessage(content="초안 답변"),
            AIMessage(
                content="초안 답변",
                name="stage5_final_answer",
                additional_kwargs={
                    "thread_chat": {
                        "citations": [
                            {
                                "document_id": "doc-2",
                                "chunk_id": "chunk-1",
                                "page": 3,
                            }
                        ],
                        "evidence_chunks": [
                            {
                                "document_id": "doc-2",
                                "chunk_id": "chunk-1",
                                "text_excerpt": "근거 문장",
                            }
                        ],
                        "retrieval_mode": "hybrid",
                        "debug_trace": final_trace,
                    }
                },
            ),
        ]

        class _FakeCheckpointer:
            def get_tuple(self, config):
                return {
                    "checkpoint": {
                        "channel_values": {
                            "messages": raw_messages,
                            "needs_clarification": False,
                            "clarification_response": None,
                            "clarification_payload": None,
                        }
                    }
                }

        @contextmanager
        def _fake_checkpointer_context():
            yield _FakeCheckpointer()

        with patch(
            "backend.services.chat_service.get_thread_detail",
            return_value={
                "thread_id": "thread-1",
                "thread_name": "테스트 스레드",
                "collection_name": "rag_chat_hybrid_thread-1",
                "description": None,
                "default_retrieval_mode": "hybrid",
                "metadata": {"lifecycle_status": "ready"},
                "active_document_ids": ["doc-2"],
                "document_count": 1,
                "created_at": None,
                "updated_at": None,
                "archived_at": None,
            },
        ), patch(
            "backend.services.chat_service.stage5_checkpointer_context",
            _fake_checkpointer_context,
        ):
            payload = load_thread_chat_view("thread-1")

        self.assertEqual(len(payload["messages"]), 2)
        assistant_message = payload["messages"][-1]
        self.assertEqual(assistant_message["content"], "초안 답변")
        self.assertEqual(assistant_message["retrieval_mode"], "hybrid")
        self.assertEqual(assistant_message["debug_trace"]["model"], "openai:gpt-4.1-mini")
        self.assertEqual(assistant_message["citations"][0]["document_id"], "doc-2")
        self.assertEqual(
            assistant_message["evidence_chunks"][0]["text_excerpt"],
            "근거 문장",
        )
