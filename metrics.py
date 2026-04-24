import numpy as np


class ClassificationMeasures:
    """Classification accuracy measures for binary or multiclass labels."""

    def __init__(self, y_true, y_pred):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self._validate_inputs()
        self.labels = [
            label.item() if hasattr(label, "item") else label
            for label in np.unique(np.concatenate((self.y_true, self.y_pred)))
        ]

    def _validate_inputs(self):
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError("y_true and y_pred must have the same length")
        if self.y_true.size == 0:
            raise ValueError("y_true and y_pred cannot be empty")

    def accuracy(self):
        """Use when every class has similar importance and the dataset is balanced."""
        return float(np.mean(self.y_true == self.y_pred))

    def error_rate(self):
        """Use as the complement of accuracy when mistakes are easier to reason about."""
        return 1 - self.accuracy()

    def confusion_matrix(self):
        """Use to inspect which true classes are confused with which predicted classes."""
        matrix = {
            true: {pred: 0 for pred in self.labels}
            for true in self.labels
        }

        for true in self.labels:
            true_mask = self.y_true == true
            for pred in self.labels:
                matrix[true][pred] = int(np.sum(true_mask & (self.y_pred == pred)))

        return matrix

    def true_positives(self, label):
        """Use for one-vs-rest analysis: correct predictions of the selected label."""
        return int(np.sum((self.y_true == label) & (self.y_pred == label)))

    def false_positives(self, label):
        """Use for one-vs-rest analysis: other labels incorrectly predicted as this label."""
        return int(np.sum((self.y_true != label) & (self.y_pred == label)))

    def false_negatives(self, label):
        """Use for one-vs-rest analysis: selected-label examples missed by the model."""
        return int(np.sum((self.y_true == label) & (self.y_pred != label)))

    def true_negatives(self, label):
        """Use for one-vs-rest analysis: non-label examples correctly rejected."""
        return int(np.sum((self.y_true != label) & (self.y_pred != label)))

    def precision(self, label):
        """Use when false positives are costly, such as spam flags or fraud alerts."""
        tp = self.true_positives(label)
        fp = self.false_positives(label)
        denominator = tp + fp
        return tp / denominator if denominator else 0

    def recall(self, label):
        """Use when false negatives are costly, such as medical screening or risk detection."""
        tp = self.true_positives(label)
        fn = self.false_negatives(label)
        denominator = tp + fn
        return tp / denominator if denominator else 0

    def specificity(self, label):
        """Use when correctly rejecting non-label cases matters for a selected class."""
        tn = self.true_negatives(label)
        fp = self.false_positives(label)
        denominator = tn + fp
        return tn / denominator if denominator else 0

    def f1_score(self, label):
        """Use when precision and recall both matter and classes may be imbalanced."""
        precision = self.precision(label)
        recall = self.recall(label)
        denominator = precision + recall
        return 2 * precision * recall / denominator if denominator else 0

    def balanced_accuracy(self):
        """Use instead of accuracy when classes are imbalanced."""
        recalls = [self.recall(label) for label in self.labels]
        return float(np.mean(recalls))

    def report(self):
        """Use to get a compact summary of common classification metrics."""
        return {
            "accuracy": self.accuracy(),
            "error_rate": self.error_rate(),
            "balanced_accuracy": self.balanced_accuracy(),
            "labels": {
                label: {
                    "precision": self.precision(label),
                    "recall": self.recall(label),
                    "specificity": self.specificity(label),
                    "f1_score": self.f1_score(label),
                }
                for label in self.labels
            },
            "confusion_matrix": self.confusion_matrix(),
        }


class RegressionMeasures:
    """Regression error and goodness-of-fit measures."""

    def __init__(self, y_true, y_pred):
        self.y_true = np.asarray(y_true, dtype=float)
        self.y_pred = np.asarray(y_pred, dtype=float)
        self._validate_inputs()

    def _validate_inputs(self):
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError("y_true and y_pred must have the same length")
        if self.y_true.size == 0:
            raise ValueError("y_true and y_pred cannot be empty")

    def errors(self):
        """Use to inspect signed residuals and detect systematic over- or under-prediction."""
        return self.y_true - self.y_pred

    def absolute_errors(self):
        """Use when only error magnitude matters, not direction."""
        return np.abs(self.errors())

    def squared_errors(self):
        """Use when larger errors should be penalized more heavily than small errors."""
        return np.square(self.errors())

    # mae
    def mean_absolute_error(self):
        """Use as a robust, easy-to-read average error in the target's original units."""
        return float(np.mean(self.absolute_errors()))

    # mse
    def mean_squared_error(self):
        """Use when large errors should dominate the score; units are squared."""
        return float(np.mean(self.squared_errors()))

    # rmse
    def root_mean_squared_error(self):
        """Use like MSE when you want the result back in the target's original units."""
        return float(np.sqrt(self.mean_squared_error()))

    # mdae
    def median_absolute_error(self):
        """Use when outliers exist and a typical absolute error is more useful than a mean."""
        return float(np.median(self.absolute_errors()))

    def max_error(self):
        """Use when worst-case prediction error is important."""
        return float(np.max(self.absolute_errors()))

    # mape
    def mean_absolute_percentage_error(self):
        """Use for relative error when true values are positive and far from zero."""
        non_zero_mask = self.y_true != 0

        if not np.any(non_zero_mask):
            return 0

        percentages = np.abs(
            (self.y_true[non_zero_mask] - self.y_pred[non_zero_mask])
            / self.y_true[non_zero_mask]
        )
        return float(np.mean(percentages))

    # smape
    def symmetric_mean_absolute_percentage_error(self):
        """Use for relative error when predictions and true values may have different scales."""
        denominator = np.abs(self.y_true) + np.abs(self.y_pred)
        non_zero_mask = denominator != 0

        if not np.any(non_zero_mask):
            return 0

        percentages = (
            2
            * np.abs(self.y_pred[non_zero_mask] - self.y_true[non_zero_mask])
            / denominator[non_zero_mask]
        )
        return float(np.mean(percentages))

    def r2_score(self):
        """Use to measure variance explained by the model against a mean baseline."""
        total_sum_squares = np.sum(np.square(self.y_true - np.mean(self.y_true)))

        if total_sum_squares == 0:
            return 1 if self.mean_squared_error() == 0 else 0

        return float(1 - (np.sum(self.squared_errors()) / total_sum_squares))

    def adjusted_r2_score(self, num_features):
        """Use for linear-style models when comparing models with different feature counts."""
        sample_count = self.y_true.size

        if sample_count <= num_features + 1:
            raise ValueError("sample count must be greater than num_features + 1")

        r2 = self.r2_score()
        return 1 - ((1 - r2) * (sample_count - 1) / (sample_count - num_features - 1))

    def explained_variance_score(self):
        """Use to measure how much variance remains after prediction errors."""
        error_values = self.errors()
        error_variance = np.var(error_values)
        true_variance = np.var(self.y_true)

        if true_variance == 0:
            return 1 if error_variance == 0 else 0

        return float(1 - (error_variance / true_variance))

    # wape
    def weighted_absolute_percentage_error(self):
        """Use for aggregate relative error, especially demand or volume forecasting."""
        denominator = np.sum(np.abs(self.y_true))

        if denominator == 0:
            return 0

        return float(np.sum(self.absolute_errors()) / denominator)

    def huber_loss(self, delta=1.0):
        """Use when you want squared-error behavior with less sensitivity to outliers."""
        if delta <= 0:
            raise ValueError("delta must be greater than 0")

        absolute_errors = self.absolute_errors()
        quadratic = np.minimum(absolute_errors, delta)
        linear = absolute_errors - quadratic
        losses = 0.5 * np.square(quadratic) + delta * linear
        return float(np.mean(losses))

    def log_cosh_loss(self):
        """Use as a smooth loss that behaves like MSE for small errors and MAE for large ones."""
        return float(np.mean(np.log(np.cosh(self.errors()))))

    def quantile_loss(self, quantile=0.5):
        """Use for quantile regression or when under- and over-prediction costs differ."""
        if not 0 < quantile < 1:
            raise ValueError("quantile must be between 0 and 1")

        errors = self.errors()
        losses = np.maximum(quantile * errors, (quantile - 1) * errors)
        return float(np.mean(losses))

    def theils_u(self):
        """Use for time series forecasts to compare the model against a naive previous-value forecast."""
        if self.y_true.size < 2:
            raise ValueError("Theil's U requires at least two observations")

        model_rmse = np.sqrt(np.mean(np.square(self.y_true[1:] - self.y_pred[1:])))
        naive_rmse = np.sqrt(np.mean(np.square(self.y_true[1:] - self.y_true[:-1])))

        if naive_rmse == 0:
            return 0 if model_rmse == 0 else float("inf")

        return float(model_rmse / naive_rmse)

    def mean_absolute_scaled_error(self):
        """Use for time series forecasts when you need scale-free error versus a naive forecast."""
        if self.y_true.size < 2:
            raise ValueError("MASE requires at least two observations")

        naive_mae = np.mean(np.abs(self.y_true[1:] - self.y_true[:-1]))

        if naive_mae == 0:
            return 0 if self.mean_absolute_error() == 0 else float("inf")

        return float(self.mean_absolute_error() / naive_mae)

    def report(self):
        """Use to get a compact summary of common regression metrics."""
        return {
            "mean_absolute_error": self.mean_absolute_error(),
            "mean_squared_error": self.mean_squared_error(),
            "root_mean_squared_error": self.root_mean_squared_error(),
            "median_absolute_error": self.median_absolute_error(),
            "max_error": self.max_error(),
            "mean_absolute_percentage_error": self.mean_absolute_percentage_error(),
            "symmetric_mean_absolute_percentage_error": self.symmetric_mean_absolute_percentage_error(),
            "weighted_absolute_percentage_error": self.weighted_absolute_percentage_error(),
            "huber_loss": self.huber_loss(),
            "log_cosh_loss": self.log_cosh_loss(),
            "quantile_loss": self.quantile_loss(),
            "theils_u": self.theils_u(),
            "mean_absolute_scaled_error": self.mean_absolute_scaled_error(),
            "r2_score": self.r2_score(),
            "explained_variance_score": self.explained_variance_score(),
        }


if __name__ == "__main__":
    actual_classes = ["cat", "dog", "cat", "bird", "dog"]
    predicted_classes = ["cat", "cat", "cat", "bird", "dog"]

    classification_metrics = ClassificationMeasures(actual_classes, predicted_classes)
    print(classification_metrics.report())

    actual_values = [3, -0.5, 2, 7]
    predicted_values = [2.5, 0, 2, 8]

    regression_metrics = RegressionMeasures(actual_values, predicted_values)
    print(regression_metrics.report())
