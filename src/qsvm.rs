//! QSVM (Quantum Support Vector Machine) Classifier Module
//!
//! This module implements a Support Vector Machine that uses a quantum kernel
//! to measure similarity between data points. The quantum kernel maps data
//! into a high-dimensional Hilbert space where linear separation becomes easier.
//!
//! ## How QSVM Works
//!
//! 1. Compute quantum kernel matrix K where K[i,j] = |⟨φ(xᵢ)|φ(xⱼ)⟩|²
//! 2. Solve the SVM optimization problem using this kernel
//! 3. Find the optimal weights (alpha) for each support vector
//! 4. Use these weights to classify new samples

use crate::quantum_kernel::{build_kernel_matrix, quantum_kernel};
use ndarray::{Array1, Array2};

/// QSVM Classifier
///
/// This struct holds the trained model parameters:
/// - `support_vectors`: The training samples that lie on or near the decision boundary
/// - `support_labels`: The class labels (-1 or +1) of the support vectors  
/// - `alpha`: The Lagrange multipliers (weights) for each support vector
/// - `bias`: The bias term (b) in the decision function
#[derive(Debug, Clone)]
pub struct QSVM {
    /// The support vectors (samples that define the decision boundary)
    pub support_vectors: Array2<f64>,
    /// Labels of the support vectors (-1.0 or +1.0)
    pub support_labels: Array1<f64>,
    /// Alpha weights for each support vector (importance of each sample)
    pub alpha: Array1<f64>,
    /// Bias term in the decision function: f(x) = Σ αᵢyᵢK(xᵢ,x) + b
    pub bias: f64,
}

impl QSVM {
    /// Trains the QSVM classifier using the SMO (Sequential Minimal Optimization) algorithm.
    ///
    /// # Arguments
    /// * `features` - Training data matrix of shape (n_samples, n_features)
    /// * `targets` - Labels, must be -1.0 or +1.0 for each sample
    /// * `c` - Regularization parameter (higher = less regularization, risk of overfitting)
    /// * `tol` - Tolerance for convergence (smaller = more precise but slower)
    /// * `max_iter` - Maximum number of iterations
    ///
    /// # Returns
    /// A trained QSVM model
    ///
    /// # The SVM Optimization Problem
    /// We want to find α that maximizes:
    ///   Σᵢαᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
    ///
    /// Subject to:
    ///   0 ≤ αᵢ ≤ C  for all i
    ///   Σᵢαᵢyᵢ = 0
    pub fn fit(
        features: &Array2<f64>,
        targets: &Array1<f64>,
        c: f64,
        tol: f64,
        max_iter: usize,
    ) -> Self {
        let n_samples = features.nrows();

        // =========================================================================
        // Step 1: Build the quantum kernel matrix
        // =========================================================================
        // K[i,j] = quantum similarity between sample i and sample j
        // This is the most computationally expensive step!
        println!(
            "Building quantum kernel matrix for {} samples...",
            n_samples
        );
        let kernel = build_kernel_matrix(features);
        println!("Kernel matrix built!");

        // =========================================================================
        // Step 2: Initialize alpha values to zero
        // =========================================================================
        // Alpha represents how important each training sample is for the decision
        // Most alphas will stay zero; only support vectors will have α > 0
        let mut alpha = Array1::zeros(n_samples);

        // Bias term (will be calculated after finding alphas)
        let mut bias = 0.0;

        // =========================================================================
        // Step 3: SMO Algorithm - Iteratively optimize pairs of alphas
        // =========================================================================
        // SMO picks two alphas at a time and optimizes them while keeping others fixed
        // This is much faster than optimizing all alphas at once

        for iteration in 0..max_iter {
            let mut num_changed = 0;

            for i in 0..n_samples {
                // Calculate the error for sample i
                // Error = (predicted value) - (actual label)
                let prediction_i = Self::predict_raw(&kernel, &alpha, targets, bias, i);
                let error_i = prediction_i - targets[i];

                // Check if this sample violates the KKT conditions
                // KKT conditions tell us if the current solution is optimal
                let y_i = targets[i];
                let alpha_i = alpha[i];

                // Violation check: is this sample worth optimizing?
                // If sample is misclassified (y*E < 0) and α < C, or
                // If sample is correctly classified (y*E > 0) and α > 0
                if (y_i * error_i < -tol && alpha_i < c) || (y_i * error_i > tol && alpha_i > 0.0) {
                    // Select a second alpha to optimize with (pick one with max |E_i - E_j|)
                    let j = Self::select_second_alpha(
                        i, error_i, &kernel, &alpha, targets, bias, n_samples,
                    );

                    let error_j = Self::predict_raw(&kernel, &alpha, targets, bias, j) - targets[j];
                    let y_j = targets[j];

                    // Save old alphas
                    let alpha_i_old = alpha[i];
                    let alpha_j_old = alpha[j];

                    // Calculate bounds for alpha_j (must satisfy 0 ≤ α ≤ C)
                    let (l, h) = if y_i != y_j {
                        // Different labels: α_i - α_j = constant
                        let l = f64::max(0.0, alpha_j_old - alpha_i_old);
                        let h = f64::min(c, c + alpha_j_old - alpha_i_old);
                        (l, h)
                    } else {
                        // Same labels: α_i + α_j = constant
                        let l = f64::max(0.0, alpha_i_old + alpha_j_old - c);
                        let h = f64::min(c, alpha_i_old + alpha_j_old);
                        (l, h)
                    };

                    if (l - h).abs() < 1e-10 {
                        continue;
                    }

                    // Calculate eta = 2*K(i,j) - K(i,i) - K(j,j)
                    // This determines how much to update alpha_j
                    let eta = 2.0 * kernel[[i, j]] - kernel[[i, i]] - kernel[[j, j]];

                    if eta >= 0.0 {
                        continue;
                    }

                    // Update alpha_j
                    alpha[j] = alpha_j_old - y_j * (error_i - error_j) / eta;

                    // Clip alpha_j to [L, H]
                    alpha[j] = alpha[j].clamp(l, h);

                    if (alpha[j] - alpha_j_old).abs() < 1e-5 {
                        continue;
                    }

                    // Update alpha_i (maintain constraint: Σ αy = 0)
                    alpha[i] = alpha_i_old + y_i * y_j * (alpha_j_old - alpha[j]);

                    // Update bias
                    let b1 = bias
                        - error_i
                        - y_i * (alpha[i] - alpha_i_old) * kernel[[i, i]]
                        - y_j * (alpha[j] - alpha_j_old) * kernel[[i, j]];
                    let b2 = bias
                        - error_j
                        - y_i * (alpha[i] - alpha_i_old) * kernel[[i, j]]
                        - y_j * (alpha[j] - alpha_j_old) * kernel[[j, j]];

                    if alpha[i] > 0.0 && alpha[i] < c {
                        bias = b1;
                    } else if alpha[j] > 0.0 && alpha[j] < c {
                        bias = b2;
                    } else {
                        bias = (b1 + b2) / 2.0;
                    }

                    num_changed += 1;
                }
            }

            // If no alphas changed, we've converged
            if num_changed == 0 {
                println!("Converged at iteration {}", iteration);
                break;
            }

            if iteration % 10 == 0 {
                println!("Iteration {}: {} alphas changed", iteration, num_changed);
            }
        }

        // =========================================================================
        // Step 4: Extract support vectors (samples with α > 0)
        // =========================================================================
        // Only samples with α > 0 affect the decision boundary
        let support_indices: Vec<usize> = alpha
            .iter()
            .enumerate()
            .filter(|(_, &a)| a > 1e-8)
            .map(|(i, _)| i)
            .collect();

        println!(
            "Found {} support vectors out of {} samples",
            support_indices.len(),
            n_samples
        );

        // Extract support vector data
        let support_vectors = features.select(ndarray::Axis(0), &support_indices);
        let support_labels = targets.select(ndarray::Axis(0), &support_indices);
        let alpha = alpha.select(ndarray::Axis(0), &support_indices);

        QSVM {
            support_vectors,
            support_labels,
            alpha,
            bias,
        }
    }

    /// Helper: Calculate raw prediction for sample i using precomputed kernel
    fn predict_raw(
        kernel: &Array2<f64>,
        alpha: &Array1<f64>,
        targets: &Array1<f64>,
        bias: f64,
        i: usize,
    ) -> f64 {
        let mut sum = 0.0;
        for j in 0..alpha.len() {
            if alpha[j] > 1e-8 {
                sum += alpha[j] * targets[j] * kernel[[j, i]];
            }
        }
        sum + bias
    }

    /// Helper: Select second alpha for SMO (heuristic: max |E_i - E_j|)
    fn select_second_alpha(
        i: usize,
        error_i: f64,
        kernel: &Array2<f64>,
        alpha: &Array1<f64>,
        targets: &Array1<f64>,
        bias: f64,
        n_samples: usize,
    ) -> usize {
        let mut max_diff = 0.0;
        let mut best_j = if i == 0 { 1 } else { 0 };

        for j in 0..n_samples {
            if j != i && alpha[j] > 1e-8 {
                let error_j = Self::predict_raw(kernel, alpha, targets, bias, j) - targets[j];
                let diff = (error_i - error_j).abs();
                if diff > max_diff {
                    max_diff = diff;
                    best_j = j;
                }
            }
        }
        best_j
    }

    /// Predicts class labels for new samples (PARALLELIZED with Rayon).
    ///
    /// # Arguments
    /// * `features` - New samples to classify, shape (n_samples, n_features)
    ///
    /// # Returns
    /// Predicted labels (-1.0 or +1.0) for each sample
    ///
    /// # The Decision Function
    /// For each new sample x:
    ///   f(x) = Σᵢ αᵢyᵢK(sᵢ,x) + b
    ///
    /// where sᵢ are the support vectors
    ///
    /// If f(x) ≥ 0, predict +1
    /// If f(x) < 0, predict -1
    ///
    /// # Performance
    /// Uses Rayon for parallel prediction across all CPU cores.
    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
        use rayon::prelude::*;
        use std::sync::Mutex;

        let n_samples = features.nrows();

        // Progress tracking
        let progress = Mutex::new(0usize);
        let print_interval = std::cmp::max(n_samples / 10, 1); // Print every 10%

        // Parallel prediction using Rayon
        let predictions: Vec<f64> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                // Get the new sample to classify
                let x: Vec<f64> = features.row(i).to_vec();

                // Calculate decision function: sum of weighted kernel evaluations
                let mut decision_value = self.bias;

                for (j, sv) in self.support_vectors.rows().into_iter().enumerate() {
                    // Compute kernel between support vector and new sample
                    let sv_vec: Vec<f64> = sv.to_vec();
                    let k = quantum_kernel(&sv_vec, &x);

                    // Add weighted contribution: αⱼ * yⱼ * K(sⱼ, x)
                    decision_value += self.alpha[j] * self.support_labels[j] * k;
                }

                // Update progress (thread-safe)
                {
                    let mut p = progress.lock().unwrap();
                    *p += 1;
                    if *p % print_interval == 0 || *p == n_samples {
                        println!(
                            "Predicting: {}/{} ({:.0}%)",
                            *p,
                            n_samples,
                            *p as f64 / n_samples as f64 * 100.0
                        );
                    }
                }

                // Convert to label: sign(f(x))
                if decision_value >= 0.0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();

        Array1::from_vec(predictions)
    }

    /// Calculates the accuracy of predictions.
    ///
    /// # Arguments
    /// * `predictions` - Predicted labels
    /// * `actual` - True labels
    ///
    /// # Returns
    /// Accuracy as a value between 0.0 and 1.0
    pub fn accuracy(predictions: &Array1<f64>, actual: &Array1<f64>) -> f64 {
        let correct: usize = predictions
            .iter()
            .zip(actual.iter())
            .filter(|(p, a)| (*p - *a).abs() < 0.5)
            .count();

        correct as f64 / predictions.len() as f64
    }
}

// =============================================================================
// TESTS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qsvm_simple() {
        // Simple linearly separable data
        // Class +1: points with x > 0.5
        // Class -1: points with x < 0.5
        let features = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.1, 0.1, // -1
                0.2, 0.2, // -1
                0.3, 0.3, // -1
                0.7, 0.7, // +1
                0.8, 0.8, // +1
                0.9, 0.9, // +1
            ],
        )
        .unwrap();

        let targets = Array1::from_vec(vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);

        // Train QSVM
        let model = QSVM::fit(&features, &targets, 1.0, 0.001, 100);

        // Predict on training data
        let predictions = model.predict(&features);

        // Calculate accuracy
        let acc = QSVM::accuracy(&predictions, &targets);
        println!("Training accuracy: {:.2}%", acc * 100.0);

        // Should get at least 80% accuracy on this simple problem
        assert!(acc >= 0.8, "Accuracy should be at least 80%");
    }
}
