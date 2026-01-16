//! Evaluation Module for QSVM Bot Detection
//!
//! This module provides:
//! 1. Classification metrics (Accuracy, Precision, Recall, F1-Score)
//! 2. Confusion matrix computation
//! 3. Model comparison between QSVM and Classical SVM
//! 4. Timing benchmarks
//!
//! ## Metrics Explained
//!
//! For binary classification (Bot = +1, Human = -1):
//!
//! - **True Positive (TP)**: Predicted Bot, Actually Bot
//! - **True Negative (TN)**: Predicted Human, Actually Human
//! - **False Positive (FP)**: Predicted Bot, Actually Human
//! - **False Negative (FN)**: Predicted Human, Actually Bot

use ndarray::Array1;
use std::time::Instant;

/// Confusion Matrix for binary classification
///
/// Layout:
/// ```text
///                 Predicted
///              -1 (Human)  +1 (Bot)
/// Actual  -1     TN         FP
///         +1     FN         TP
/// ```
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// True Positives: correctly predicted positive class
    pub tp: usize,
    /// True Negatives: correctly predicted negative class
    pub tn: usize,
    /// False Positives: incorrectly predicted positive (Type I error)
    pub fp: usize,
    /// False Negatives: incorrectly predicted negative (Type II error)
    pub fn_: usize,
}

impl ConfusionMatrix {
    /// Computes the confusion matrix from predictions and actual labels.
    ///
    /// # Arguments
    /// * `predictions` - Predicted labels (-1.0 or +1.0)
    /// * `actual` - True labels (-1.0 or +1.0)
    ///
    /// # Returns
    /// A ConfusionMatrix struct with TP, TN, FP, FN counts
    pub fn from_predictions(predictions: &Array1<f64>, actual: &Array1<f64>) -> Self {
        let mut tp = 0;
        let mut tn = 0;
        let mut fp = 0;
        let mut fn_ = 0;

        // Iterate through all predictions and count each type
        for (pred, act) in predictions.iter().zip(actual.iter()) {
            let p = *pred > 0.0; // Predicted positive?
            let a = *act > 0.0; // Actually positive?

            match (p, a) {
                (true, true) => tp += 1,   // Predicted +, Actually + = TP
                (true, false) => fp += 1,  // Predicted +, Actually - = FP
                (false, true) => fn_ += 1, // Predicted -, Actually + = FN
                (false, false) => tn += 1, // Predicted -, Actually - = TN
            }
        }

        ConfusionMatrix { tp, tn, fp, fn_ }
    }

    /// Prints a nicely formatted confusion matrix
    pub fn print(&self) {
        println!("\n┌─────────────────────────────────────┐");
        println!("│         CONFUSION MATRIX            │");
        println!("├─────────────────────────────────────┤");
        println!("│              Predicted              │");
        println!("│           -1 (Human)  +1 (Bot)      │");
        println!("├─────────────────────────────────────┤");
        println!("│ Actual -1    {:>5}       {:>5}      │", self.tn, self.fp);
        println!("│        +1    {:>5}       {:>5}      │", self.fn_, self.tp);
        println!("└─────────────────────────────────────┘");
    }
}

/// Computes Accuracy: (TP + TN) / Total
///
/// Accuracy measures the proportion of correct predictions.
///
/// # Formula
/// Accuracy = (TP + TN) / (TP + TN + FP + FN)
///
/// # Note
/// Accuracy can be misleading with imbalanced datasets!
/// If 95% of samples are class A, predicting all A gives 95% accuracy.
pub fn accuracy(cm: &ConfusionMatrix) -> f64 {
    let total = (cm.tp + cm.tn + cm.fp + cm.fn_) as f64;
    if total == 0.0 {
        return 0.0;
    }
    (cm.tp + cm.tn) as f64 / total
}

/// Computes Precision: TP / (TP + FP)
///
/// Precision measures: "Of all predicted positive, how many are actually positive?"
/// High precision = few false positives
///
/// # Formula
/// Precision = TP / (TP + FP)
///
/// # Use case
/// Important when false positives are costly (e.g., spam detection)
pub fn precision(cm: &ConfusionMatrix) -> f64 {
    let predicted_positive = (cm.tp + cm.fp) as f64;
    if predicted_positive == 0.0 {
        return 0.0;
    }
    cm.tp as f64 / predicted_positive
}

/// Computes Recall (Sensitivity): TP / (TP + FN)
///
/// Recall measures: "Of all actual positive, how many did we catch?"
/// High recall = few false negatives
///
/// # Formula
/// Recall = TP / (TP + FN)
///
/// # Use case
/// Important when false negatives are costly (e.g., disease detection)
pub fn recall(cm: &ConfusionMatrix) -> f64 {
    let actual_positive = (cm.tp + cm.fn_) as f64;
    if actual_positive == 0.0 {
        return 0.0;
    }
    cm.tp as f64 / actual_positive
}

/// Computes F1-Score: Harmonic mean of Precision and Recall
///
/// F1 balances precision and recall, useful when you need both.
///
/// # Formula
/// F1 = 2 * (Precision * Recall) / (Precision + Recall)
///
/// # Note
/// F1 = 1.0 is perfect, F1 = 0.0 is worst
pub fn f1_score(cm: &ConfusionMatrix) -> f64 {
    let p = precision(cm);
    let r = recall(cm);

    if p + r == 0.0 {
        return 0.0;
    }

    2.0 * p * r / (p + r)
}

/// Holds all evaluation metrics for a model
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub model_name: String,
    pub confusion_matrix: ConfusionMatrix,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub training_time_ms: u128,
    pub prediction_time_ms: u128,
}

impl EvaluationResult {
    /// Creates evaluation result from confusion matrix and timing
    pub fn new(
        model_name: &str,
        cm: ConfusionMatrix,
        training_time_ms: u128,
        prediction_time_ms: u128,
    ) -> Self {
        EvaluationResult {
            model_name: model_name.to_string(),
            accuracy: accuracy(&cm),
            precision: precision(&cm),
            recall: recall(&cm),
            f1: f1_score(&cm),
            confusion_matrix: cm,
            training_time_ms,
            prediction_time_ms,
        }
    }

    /// Prints a formatted evaluation report
    pub fn print(&self) {
        println!("\n╔═════════════════════════════════════════╗");
        println!("║  {} Results", self.model_name);
        println!("╠═════════════════════════════════════════╣");
        println!(
            "║  Accuracy:   {:>6.2}%                     ║",
            self.accuracy * 100.0
        );
        println!(
            "║  Precision:  {:>6.2}%                     ║",
            self.precision * 100.0
        );
        println!(
            "║  Recall:     {:>6.2}%                     ║",
            self.recall * 100.0
        );
        println!(
            "║  F1 Score:   {:>6.2}%                     ║",
            self.f1 * 100.0
        );
        println!("╠═════════════════════════════════════════╣");
        println!(
            "║  Training Time:   {:>8} ms            ║",
            self.training_time_ms
        );
        println!(
            "║  Prediction Time: {:>8} ms            ║",
            self.prediction_time_ms
        );
        println!("╚═════════════════════════════════════════╝");

        self.confusion_matrix.print();
    }
}

/// Compares two models and prints a comparison table
///
/// # Arguments
/// * `qsvm_result` - Evaluation result for QSVM
/// * `classical_result` - Evaluation result for Classical SVM
pub fn compare_models(qsvm_result: &EvaluationResult, classical_result: &EvaluationResult) {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║              MODEL COMPARISON: QSVM vs Classical SVM       ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  Metric         │   QSVM        │  Classical    │  Winner ║");
    println!("╠════════════════════════════════════════════════════════════╣");

    // Accuracy comparison
    let acc_winner = if qsvm_result.accuracy > classical_result.accuracy {
        "QSVM"
    } else if qsvm_result.accuracy < classical_result.accuracy {
        "Classic"
    } else {
        "Tie"
    };
    println!(
        "║  Accuracy       │  {:>6.2}%      │  {:>6.2}%      │  {:>7} ║",
        qsvm_result.accuracy * 100.0,
        classical_result.accuracy * 100.0,
        acc_winner
    );

    // Precision comparison
    let prec_winner = if qsvm_result.precision > classical_result.precision {
        "QSVM"
    } else if qsvm_result.precision < classical_result.precision {
        "Classic"
    } else {
        "Tie"
    };
    println!(
        "║  Precision      │  {:>6.2}%      │  {:>6.2}%      │  {:>7} ║",
        qsvm_result.precision * 100.0,
        classical_result.precision * 100.0,
        prec_winner
    );

    // Recall comparison
    let rec_winner = if qsvm_result.recall > classical_result.recall {
        "QSVM"
    } else if qsvm_result.recall < classical_result.recall {
        "Classic"
    } else {
        "Tie"
    };
    println!(
        "║  Recall         │  {:>6.2}%      │  {:>6.2}%      │  {:>7} ║",
        qsvm_result.recall * 100.0,
        classical_result.recall * 100.0,
        rec_winner
    );

    // F1 comparison
    let f1_winner = if qsvm_result.f1 > classical_result.f1 {
        "QSVM"
    } else if qsvm_result.f1 < classical_result.f1 {
        "Classic"
    } else {
        "Tie"
    };
    println!(
        "║  F1 Score       │  {:>6.2}%      │  {:>6.2}%      │  {:>7} ║",
        qsvm_result.f1 * 100.0,
        classical_result.f1 * 100.0,
        f1_winner
    );

    println!("╠════════════════════════════════════════════════════════════╣");

    // Time comparison
    let train_winner = if qsvm_result.training_time_ms < classical_result.training_time_ms {
        "QSVM"
    } else if qsvm_result.training_time_ms > classical_result.training_time_ms {
        "Classic"
    } else {
        "Tie"
    };
    println!(
        "║  Train Time     │  {:>6} ms    │  {:>6} ms    │  {:>7} ║",
        qsvm_result.training_time_ms, classical_result.training_time_ms, train_winner
    );

    let pred_winner = if qsvm_result.prediction_time_ms < classical_result.prediction_time_ms {
        "QSVM"
    } else if qsvm_result.prediction_time_ms > classical_result.prediction_time_ms {
        "Classic"
    } else {
        "Tie"
    };
    println!(
        "║  Predict Time   │  {:>6} ms    │  {:>6} ms    │  {:>7} ║",
        qsvm_result.prediction_time_ms, classical_result.prediction_time_ms, pred_winner
    );

    println!("╚════════════════════════════════════════════════════════════╝");

    // Summary
    println!("\n📊 SUMMARY:");
    let acc_diff = (qsvm_result.accuracy - classical_result.accuracy) * 100.0;
    if acc_diff > 0.0 {
        println!(
            "   ✅ QSVM outperforms Classical SVM by {:.2}% accuracy",
            acc_diff
        );
    } else if acc_diff < 0.0 {
        println!(
            "   ℹ️  Classical SVM outperforms QSVM by {:.2}% accuracy",
            -acc_diff
        );
    } else {
        println!("   🤝 Both models have equal accuracy");
    }

    let speedup =
        classical_result.training_time_ms as f64 / qsvm_result.training_time_ms.max(1) as f64;
    if speedup > 1.0 {
        println!("   ⚡ QSVM is {:.1}x faster in training", speedup);
    } else {
        println!(
            "   ⏱️  Classical SVM is {:.1}x faster in training",
            1.0 / speedup
        );
    }
}

/// Helper function to measure execution time
///
/// # Usage
/// ```ignore
/// let (result, duration_ms) = measure_time(|| {
///     // Some computation
/// });
/// ```
pub fn measure_time<F, R>(f: F) -> (R, u128)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed().as_millis();
    (result, duration)
}

// =============================================================================
// TESTS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix() {
        // Predictions: [+1, +1, -1, -1, +1]
        // Actual:      [+1, -1, -1, +1, +1]
        let predictions = Array1::from_vec(vec![1.0, 1.0, -1.0, -1.0, 1.0]);
        let actual = Array1::from_vec(vec![1.0, -1.0, -1.0, 1.0, 1.0]);

        let cm = ConfusionMatrix::from_predictions(&predictions, &actual);

        // TP: pred=+1, act=+1 → positions 0, 4 → 2
        // TN: pred=-1, act=-1 → position 2 → 1
        // FP: pred=+1, act=-1 → position 1 → 1
        // FN: pred=-1, act=+1 → position 3 → 1
        assert_eq!(cm.tp, 2);
        assert_eq!(cm.tn, 1);
        assert_eq!(cm.fp, 1);
        assert_eq!(cm.fn_, 1);
    }

    #[test]
    fn test_metrics() {
        let cm = ConfusionMatrix {
            tp: 50,
            tn: 40,
            fp: 10,
            fn_: 5,
        };

        // Accuracy = (50 + 40) / (50 + 40 + 10 + 5) = 90/105 ≈ 0.857
        let acc = accuracy(&cm);
        assert!((acc - 0.857).abs() < 0.01);

        // Precision = 50 / (50 + 10) = 50/60 ≈ 0.833
        let prec = precision(&cm);
        assert!((prec - 0.833).abs() < 0.01);

        // Recall = 50 / (50 + 5) = 50/55 ≈ 0.909
        let rec = recall(&cm);
        assert!((rec - 0.909).abs() < 0.01);

        // F1 = 2 * 0.833 * 0.909 / (0.833 + 0.909) ≈ 0.870
        let f1 = f1_score(&cm);
        assert!((f1 - 0.870).abs() < 0.01);
    }

    #[test]
    fn test_measure_time() {
        let (result, duration_ms) = measure_time(|| {
            // Simple computation
            (1..1000).sum::<i32>()
        });

        assert_eq!(result, 499500);
        assert!(duration_ms < 100); // Should be very fast
    }
}
