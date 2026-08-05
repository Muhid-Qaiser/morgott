from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import prepare


class LogInjectLongSpanPanelTests(unittest.TestCase):
    def test_formatted_spans_cover_only_attacked_entries(self) -> None:
        text, spans = prepare._formatted(
            ["clean one", "attack two", "clean three"], {1}
        )
        normalized = prepare.strict_normalize(text)

        self.assertEqual(len(spans), 1)
        self.assertEqual(
            normalized[spans[0]["start"] : spans[0]["end"]], "[entry 2] attack two"
        )


if __name__ == "__main__":
    unittest.main()
