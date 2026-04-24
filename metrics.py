import numpy as np


class ClassificationMeasures:
    """Classification accuracy measures for binary or multiclass labels."""

    def __init__(self, y_true, y_pred, y_score=None):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_score = None if y_score is None else np.asarray(y_score, dtype=float)
        self._validate_inputs()
        self.labels = [
            label.item() if hasattr(label, "item") else label
            for label in np.unique(np.concatenate((self.y_true, self.y_pred)))
        ]
        self._validate_scores()

    def _validate_inputs(self):
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError("y_true and y_pred must have the same length")
        if self.y_true.size == 0:
            raise ValueError("y_true and y_pred cannot be empty")

    def _validate_scores(self):
        if self.y_score is None:
            return
        if self.y_score.ndim == 1:
            if self.y_score.shape[0] != self.y_true.size:
                raise ValueError("1D y_score must have the same length as y_true")
            if len(self.labels) != 2:
                raise ValueError("1D y_score is only supported for binary classification")
        elif self.y_score.ndim == 2:
            expected_shape = (self.y_true.size, len(self.labels))
            if self.y_score.shape != expected_shape:
                raise ValueError(
                    "2D y_score must have shape (n_samples, n_labels), "
                    "with columns ordered like labels"
                )
        else:
            raise ValueError("y_score must be a 1D or 2D array")

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

    def _require_scores(self):
        if self.y_score is None:
            raise ValueError("ROC AUC requires y_score values")

    @staticmethod
    def _average_ranks(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.size, dtype=float)
        sorted_values = values[order]
        index = 0

        while index < values.size:
            next_index = index + 1
            while next_index < values.size and sorted_values[next_index] == sorted_values[index]:
                next_index += 1

            average_rank = (index + 1 + next_index) / 2
            ranks[order[index:next_index]] = average_rank
            index = next_index

        return ranks

    def _scores_for_label(self, label):
        if label not in self.labels:
            raise ValueError("label must exist in y_true or y_pred")

        if self.y_score.ndim == 1:
            if label == self.labels[-1]:
                return self.y_score
            if label == self.labels[0]:
                return -self.y_score
            raise ValueError("1D y_score can only be used with binary labels")

        label_index = self.labels.index(label)
        return self.y_score[:, label_index]

    def roc_auc(self, label=None):
        """Use to measure ranking quality for a binary or one-vs-rest classifier."""
        self._require_scores()

        if label is None:
            if len(self.labels) != 2:
                raise ValueError("label is required for multiclass ROC AUC")
            label = self.labels[-1]

        positive_mask = self.y_true == label
        positive_count = int(np.sum(positive_mask))
        negative_count = self.y_true.size - positive_count

        if positive_count == 0 or negative_count == 0:
            raise ValueError("ROC AUC requires both positive and negative examples")

        scores = self._scores_for_label(label)
        ranks = self._average_ranks(scores)
        positive_rank_sum = np.sum(ranks[positive_mask])
        auc = (
            positive_rank_sum
            - (positive_count * (positive_count + 1) / 2)
        ) / (positive_count * negative_count)
        return float(auc)

    def roc_curve(self, label=None):
        """Use to get false-positive and true-positive rates for ROC plotting."""
        self._require_scores()

        if label is None:
            if len(self.labels) != 2:
                raise ValueError("label is required for multiclass ROC curves")
            label = self.labels[-1]

        positive_mask = self.y_true == label
        positive_count = int(np.sum(positive_mask))
        negative_count = self.y_true.size - positive_count

        if positive_count == 0 or negative_count == 0:
            raise ValueError("ROC curve requires both positive and negative examples")

        scores = self._scores_for_label(label)
        thresholds = np.unique(scores)[::-1]
        false_positive_rates = [0.0]
        true_positive_rates = [0.0]

        for threshold in thresholds:
            predicted_positive = scores >= threshold
            true_positives = np.sum(predicted_positive & positive_mask)
            false_positives = np.sum(predicted_positive & ~positive_mask)
            true_positive_rates.append(true_positives / positive_count)
            false_positive_rates.append(false_positives / negative_count)

        return {
            "false_positive_rate": np.asarray(false_positive_rates, dtype=float),
            "true_positive_rate": np.asarray(true_positive_rates, dtype=float),
            "thresholds": np.concatenate(([float("inf")], thresholds)),
        }

    def plot_roc_curve(self, labels=None, show=True, save_path=None, ax=None):
        """Use to draw ROC curves with AUC values for binary or multiclass scores."""
        self._require_scores()

        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise ImportError(
                "plot_roc_curve requires matplotlib. Install it with "
                "`pip install matplotlib`."
            ) from error

        if labels is None:
            labels = [self.labels[-1]] if self.y_score.ndim == 1 else self.labels
        elif np.isscalar(labels) or isinstance(labels, str):
            labels = [labels]

        if ax is None:
            _, ax = plt.subplots()

        for label in labels:
            curve = self.roc_curve(label)
            auc = self.roc_auc(label)
            ax.plot(
                curve["false_positive_rate"],
                curve["true_positive_rate"],
                drawstyle="steps-post",
                marker="o",
                markersize=3,
                label=f"{label} (AUC = {auc:.3f})",
            )

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        if save_path is not None:
            ax.figure.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()

        return ax

    def macro_roc_auc(self):
        """Use for multiclass ROC AUC when every class should have equal weight."""
        self._require_scores()
        auc_values = [self.roc_auc(label) for label in self.labels]
        return float(np.mean(auc_values))

    def weighted_roc_auc(self):
        """Use for multiclass ROC AUC weighted by each class's support."""
        self._require_scores()
        supports = np.array([np.sum(self.y_true == label) for label in self.labels])
        auc_values = np.array([self.roc_auc(label) for label in self.labels])
        return float(np.average(auc_values, weights=supports))

    def micro_roc_auc(self):
        """Use for multiclass ROC AUC when every one-vs-rest decision has equal weight."""
        self._require_scores()

        if self.y_score.ndim == 1:
            return self.roc_auc()

        binary_true = np.column_stack([
            self.y_true == label
            for label in self.labels
        ]).ravel()
        scores = self.y_score.ravel()
        positive_count = int(np.sum(binary_true))
        negative_count = binary_true.size - positive_count

        if positive_count == 0 or negative_count == 0:
            raise ValueError("ROC AUC requires both positive and negative examples")

        ranks = self._average_ranks(scores)
        positive_rank_sum = np.sum(ranks[binary_true])
        auc = (
            positive_rank_sum
            - (positive_count * (positive_count + 1) / 2)
        ) / (positive_count * negative_count)
        return float(auc)

    def report(self):
        """Use to get a compact summary of common classification metrics."""
        report = {
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

        if self.y_score is not None:
            report["roc_auc"] = {
                "macro": self.macro_roc_auc(),
                "weighted": self.weighted_roc_auc(),
                "micro": self.micro_roc_auc(),
                "labels": {
                    label: self.roc_auc(label)
                    for label in self.labels
                },
            }

        return report


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


class TimeSeriesMeasures:
    """Time series forecast accuracy and residual diagnostics."""

    def __init__(self, y_true, y_pred, seasonality=1):
        self.y_true = np.asarray(y_true, dtype=float)
        self.y_pred = np.asarray(y_pred, dtype=float)
        self.seasonality = seasonality
        self._validate_inputs()

    def _validate_inputs(self):
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError("y_true and y_pred must have the same length")
        if self.y_true.size == 0:
            raise ValueError("y_true and y_pred cannot be empty")
        if not isinstance(self.seasonality, int) or self.seasonality < 1:
            raise ValueError("seasonality must be a positive integer")

    def errors(self):
        """Use to inspect signed forecast errors; positive values mean under-forecasting."""
        return self.y_true - self.y_pred

    def absolute_errors(self):
        """Use when only forecast error magnitude matters, not direction."""
        return np.abs(self.errors())

    def squared_errors(self):
        """Use when larger forecast misses should be penalized more heavily."""
        return np.square(self.errors())

    # me
    def mean_error(self):
        """Use to measure average signed error; positive values mean under-forecasting."""
        return float(np.mean(self.errors()))

    # mfe
    def mean_forecast_error(self):
        """Use as an alias for mean_error in forecasting contexts."""
        return self.mean_error()

    # mae
    def mean_absolute_error(self):
        """Use as a scale-dependent average forecast error in the series' original units."""
        return float(np.mean(self.absolute_errors()))

    # mse
    def mean_squared_error(self):
        """Use when large forecast errors should dominate the score."""
        return float(np.mean(self.squared_errors()))

    # rmse
    def root_mean_squared_error(self):
        """Use like MSE when you want the score back in the series' original units."""
        return float(np.sqrt(self.mean_squared_error()))

    # mape
    def mean_absolute_percentage_error(self):
        """Use for relative forecast error when true values are not zero."""
        non_zero_mask = self.y_true != 0

        if not np.any(non_zero_mask):
            return 0

        percentages = self.absolute_errors()[non_zero_mask] / np.abs(self.y_true[non_zero_mask])
        return float(np.mean(percentages))

    # smape
    def symmetric_mean_absolute_percentage_error(self):
        """Use for relative forecast error when true and predicted scales can both vary."""
        denominator = np.abs(self.y_true) + np.abs(self.y_pred)
        non_zero_mask = denominator != 0

        if not np.any(non_zero_mask):
            return 0

        percentages = 2 * self.absolute_errors()[non_zero_mask] / denominator[non_zero_mask]
        return float(np.mean(percentages))

    # wape
    def weighted_absolute_percentage_error(self):
        """Use for aggregate relative forecast error, especially demand or volume series."""
        denominator = np.sum(np.abs(self.y_true))

        if denominator == 0:
            return 0

        return float(np.sum(self.absolute_errors()) / denominator)

    def _naive_errors(self):
        if self.y_true.size <= self.seasonality:
            raise ValueError("series length must be greater than seasonality")

        return self.y_true[self.seasonality:] - self.y_true[:-self.seasonality]

    # mase
    def mean_absolute_scaled_error(self):
        """Use for scale-free accuracy versus a seasonal naive forecast baseline."""
        naive_mae = np.mean(np.abs(self._naive_errors()))

        if naive_mae == 0:
            return 0 if self.mean_absolute_error() == 0 else float("inf")

        return float(self.mean_absolute_error() / naive_mae)

    # rmsse
    def root_mean_squared_scaled_error(self):
        """Use for scale-free RMSE versus a seasonal naive forecast baseline."""
        naive_mse = np.mean(np.square(self._naive_errors()))

        if naive_mse == 0:
            return 0 if self.root_mean_squared_error() == 0 else float("inf")

        return float(self.root_mean_squared_error() / np.sqrt(naive_mse))

    def theils_u(self):
        """Use to compare forecast RMSE against a seasonal naive forecast baseline."""
        naive_errors = self._naive_errors()
        model_errors = self.y_true[self.seasonality:] - self.y_pred[self.seasonality:]
        model_rmse = np.sqrt(np.mean(np.square(model_errors)))
        naive_rmse = np.sqrt(np.mean(np.square(naive_errors)))

        if naive_rmse == 0:
            return 0 if model_rmse == 0 else float("inf")

        return float(model_rmse / naive_rmse)

    def forecast_bias(self):
        """Use to measure average forecast tendency; positive values mean over-forecasting."""
        return float(np.mean(self.y_pred - self.y_true))

    def tracking_signal(self):
        """Use to detect persistent bias as cumulative error divided by mean absolute error."""
        mean_absolute_deviation = self.mean_absolute_error()

        if mean_absolute_deviation == 0:
            return 0

        return float(np.sum(self.errors()) / mean_absolute_deviation)

    def directional_accuracy(self):
        """Use when predicting the direction of change matters more than exact magnitude."""
        if self.y_true.size < 2:
            raise ValueError("directional accuracy requires at least two observations")

        actual_direction = np.sign(self.y_true[1:] - self.y_true[:-1])
        forecast_direction = np.sign(self.y_pred[1:] - self.y_true[:-1])
        return float(np.mean(actual_direction == forecast_direction))

    def residual_autocorrelation(self, lag=1):
        """Use to check whether forecast residuals still contain time dependence."""
        if not isinstance(lag, int) or lag < 1:
            raise ValueError("lag must be a positive integer")
        if self.y_true.size <= lag:
            raise ValueError("series length must be greater than lag")

        errors = self.errors()
        current_errors = errors[lag:]
        lagged_errors = errors[:-lag]

        if np.std(current_errors) == 0 or np.std(lagged_errors) == 0:
            return 0

        return float(np.corrcoef(current_errors, lagged_errors)[0, 1])

    def report(self):
        """Use to get a compact summary of common time series forecast metrics."""
        report = {
            "mean_error": self.mean_error(),
            "mean_absolute_error": self.mean_absolute_error(),
            "mean_squared_error": self.mean_squared_error(),
            "root_mean_squared_error": self.root_mean_squared_error(),
            "mean_absolute_percentage_error": self.mean_absolute_percentage_error(),
            "symmetric_mean_absolute_percentage_error": self.symmetric_mean_absolute_percentage_error(),
            "weighted_absolute_percentage_error": self.weighted_absolute_percentage_error(),
            "mean_absolute_scaled_error": self.mean_absolute_scaled_error(),
            "root_mean_squared_scaled_error": self.root_mean_squared_scaled_error(),
            "theils_u": self.theils_u(),
            "forecast_bias": self.forecast_bias(),
            "tracking_signal": self.tracking_signal(),
        }

        if self.y_true.size > 1:
            report["directional_accuracy"] = self.directional_accuracy()
            report["residual_autocorrelation_lag_1"] = self.residual_autocorrelation(lag=1)

        return report


if __name__ == "__main__":
    actual_classes = [
        "cat", "dog", "cat", "bird", "dog", "bird",
        "cat", "dog", "bird", "cat", "dog", "bird",
    ]
    predicted_classes = [
        "cat", "dog", "cat", "bird", "dog", "bird",
        "cat", "dog", "bird", "cat", "bird", "cat",
    ]
    predicted_class_scores = [
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
    ]

    classification_metrics = ClassificationMeasures(
        actual_classes,
        predicted_classes,
        predicted_class_scores,
    )
    print(classification_metrics.report())
    classification_metrics.plot_roc_curve(
        save_path="roc_auc_test_chart.png",
        show=False,
    )
    print("ROC AUC chart saved to roc_auc_test_chart.png")

    actual_values = [3, -0.5, 2, 7]
    predicted_values = [2.5, 0, 2, 8]

    regression_metrics = RegressionMeasures(actual_values, predicted_values)
    print(regression_metrics.report())

    actual_series = [100, 112, 108, 115, 120, 118]
    predicted_series = [98, 110, 111, 113, 122, 117]

    time_series_metrics = TimeSeriesMeasures(actual_series, predicted_series, seasonality=1)
    print(time_series_metrics.report())
