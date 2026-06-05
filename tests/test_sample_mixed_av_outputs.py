import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nla.scripts import sample_mixed_av_outputs as mixed


class _Cfg:
    d_model = 8


class _Client:
    cfg = _Cfg()


class _Decoder:
    client = _Client()

    def generate(self, vector, _args):
        return f"random norm={np.linalg.norm(vector):.6f}"


def _args(**overrides):
    values = {
        "count": 4,
        "allow_odd": False,
        "seed": 123,
        "blind": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class SampleMixedAVOutputsTest(unittest.TestCase):
    def test_read_real_trace_filters_null_and_empty_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            rows = [
                {"row_index": 0, "nla_output": "real A", "token_text": "A"},
                {"row_index": 1, "nla_output": None, "token_text": "B"},
                {"row_index": 2, "nla_output": "", "token_text": "C"},
                {"row_index": 3, "nla_output": "real D", "token_text": "D"},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            out = mixed.read_real_trace(path)

        self.assertEqual([row["nla_output"] for row in out], ["real A", "real D"])
        self.assertEqual([row["_trace_row_index"] for row in out], [0, 3])

    def test_split_counts_requires_even_without_allow_odd(self):
        self.assertEqual(mixed.split_counts(4, allow_odd=False), (2, 2))
        self.assertEqual(mixed.split_counts(5, allow_odd=True), (2, 3))
        with self.assertRaisesRegex(ValueError, "even"):
            mixed.split_counts(5, allow_odd=False)

    def test_sample_unit_vector_is_deterministic_and_normalized(self):
        v1 = mixed.sample_unit_vector(8, 42)
        v2 = mixed.sample_unit_vector(8, 42)

        np.testing.assert_array_equal(v1, v2)
        self.assertAlmostEqual(float(np.linalg.norm(v1)), 1.0, places=6)

    def test_build_mixed_records_is_deterministic_and_balanced(self):
        real_rows = [
            {"_trace_row_index": i, "row_index": i, "token_text": str(i), "nla_output": f"real {i}"}
            for i in range(10)
        ]

        first = mixed.build_mixed_records(real_rows, _Decoder(), _args())
        second = mixed.build_mixed_records(real_rows, _Decoder(), _args())

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(sum(rec["kind"] == "real" for rec in first), 2)
        self.assertEqual(sum(rec["kind"] == "random" for rec in first), 2)
        self.assertEqual([rec["sample_index"] for rec in first], [0, 1, 2, 3])

    def test_blind_records_preserve_answer_kind_without_visible_kind(self):
        real_rows = [{"_trace_row_index": 0, "row_index": 7, "nla_output": "real"}]

        records = mixed.build_mixed_records(real_rows, _Decoder(), _args(count=2, blind=True))

        self.assertEqual({rec["answer_kind"] for rec in records}, {"real", "random"})
        self.assertTrue(all("kind" not in rec for rec in records))


if __name__ == "__main__":
    unittest.main()
