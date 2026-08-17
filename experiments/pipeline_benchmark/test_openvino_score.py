import unittest

from experiments.pipeline_benchmark.openvino_score import _split_scores


class OpenVinoScoreTests(unittest.TestCase):
    def test_splits_flat_window_scores_by_artifact(self):
        self.assertEqual(_split_scores([1, 2], [0.1, 0.2, 0.3]), [[0.1], [0.2, 0.3]])
        with self.assertRaises(ValueError):
            _split_scores([2], [0.1])


if __name__ == "__main__":
    unittest.main()
