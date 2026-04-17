import unittest

from backend.stage3_chunking.sparse_policy import determine_sparse_policy


class SparsePolicyTests(unittest.TestCase):
    def test_visual_chunk_requires_caption_or_summary_anchor(self):
        kept = determine_sparse_policy(
            chunk_type="table",
            body_text="Table 1 body",
            section_title="2. 결과",
            metadata={
                "caption": "Table 1. 예시 표",
                "summary_text": "실험 결과를 요약한 표다.",
            },
        )
        self.assertTrue(kept["sparse_keep"])
        self.assertIn("Table 1. 예시 표", kept["sparse_text"])

        dropped = determine_sparse_policy(
            chunk_type="figure",
            body_text="Figure body only",
            section_title="2. 결과",
            metadata={},
        )
        self.assertFalse(dropped["sparse_keep"])
        self.assertEqual(dropped["sparse_exclude_reason"], "missing_visual_anchor")

    def test_prose_with_role_hint_is_filtered(self):
        result = determine_sparse_policy(
            chunk_type="text",
            body_text="Kim et al. 2024 [1] [2]",
            section_title=None,
            metadata={
                "group_type": "prose",
                "estimated_tokens": 12,
                "sentence_like_ratio": 0.0,
                "line_count": 3,
                "average_line_tokens": 4,
                "sparse_role_hints": ["reference_like"],
            },
        )
        self.assertFalse(result["sparse_keep"])
        self.assertEqual(result["sparse_exclude_reason"], "role_hint_filtered")

    def test_prose_with_section_anchor_is_kept(self):
        result = determine_sparse_policy(
            chunk_type="text",
            body_text="이 절에서는 하이브리드 검색의 핵심 파라미터를 설명합니다.",
            section_title="3. 검색 전략",
            metadata={
                "group_type": "prose",
                "estimated_tokens": 14,
                "sentence_like_ratio": 1.0,
                "line_count": 1,
                "average_line_tokens": 14,
            },
        )
        self.assertTrue(result["sparse_keep"])
        self.assertIn("3. 검색 전략", result["sparse_text"])


if __name__ == "__main__":
    unittest.main()
