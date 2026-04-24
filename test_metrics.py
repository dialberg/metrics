import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from metrics import ClassificationMeasures


class ClassificationMeasuresRocAucTests(unittest.TestCase):
    def setUp(self):
        self.metrics = ClassificationMeasures(
            [
                "cat", "dog", "cat", "bird", "dog", "bird",
                "cat", "dog", "bird", "cat", "dog", "bird",
            ],
            [
                "cat", "dog", "cat", "bird", "dog", "bird",
                "cat", "dog", "bird", "cat", "bird", "cat",
            ],
            [
                [0.10, 0.75, 0.15],
                [0.50, 0.05, 0.45],
                [0.15, 0.60, 0.25],
                [0.70, 0.20, 0.10],
                [0.15, 0.30, 0.55],
                [0.55, 0.25, 0.20],
                [0.25, 0.45, 0.50],
                [0.30, 0.25, 0.45],
                [0.40, 0.35, 0.25],
                [0.20, 0.50, 0.30],
                [0.45, 0.20, 0.35],
                [0.35, 0.55, 0.25],
            ],
        )

    def test_roc_auc_scores(self):
        self.assertAlmostEqual(self.metrics.roc_auc("bird"), 0.875)
        self.assertAlmostEqual(self.metrics.roc_auc("cat"), 0.9375)
        self.assertAlmostEqual(self.metrics.roc_auc("dog"), 0.90625)
        self.assertAlmostEqual(self.metrics.macro_roc_auc(), 0.90625)
        self.assertAlmostEqual(self.metrics.weighted_roc_auc(), 0.90625)
        self.assertAlmostEqual(self.metrics.micro_roc_auc(), 0.9079861111111112)

    def test_roc_curve_chart_is_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            chart_path = os.path.join(temp_dir, "roc_auc_test_chart.png")
            ax = self.metrics.plot_roc_curve(save_path=chart_path, show=False)

            self.assertTrue(os.path.exists(chart_path))
            self.assertGreater(os.path.getsize(chart_path), 0)
            self.assertEqual(ax.get_title(), "ROC Curve")
            legend_labels = [
                text.get_text()
                for text in ax.get_legend().get_texts()
            ]
            self.assertIn("bird (AUC = 0.875)", legend_labels)
            self.assertIn("cat (AUC = 0.938)", legend_labels)
            self.assertIn("dog (AUC = 0.906)", legend_labels)

            plt.close(ax.figure)


if __name__ == "__main__":
    unittest.main()
