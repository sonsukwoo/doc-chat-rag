from __future__ import annotations

import unittest
from contextlib import contextmanager
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from backend.services.chat_service import load_thread_chat_view


class ChatServiceTests(unittest.TestCase):
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
