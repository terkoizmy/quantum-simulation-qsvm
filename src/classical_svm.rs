//! Classical SVM Module
//!
//! This module implements a Classical Support Vector Machine using the linfa-svm crate.
//! This serves as a baseline to compare against our Quantum SVM.
//!
//! ## Purpose
//!
//! By comparing QSVM vs Classical SVM, we can measure:
//! - Accuracy difference (does quantum kernel improve classification?)
//! - Training time difference (quantum kernel is slower due to simulation)
//! - When quantum advantages might appear (high-dimensional, complex patterns)

use linfa::prelude::*;
use linfa_svm::{Svm, SvmParams};
use ndarray::{Array1, Array2};

/// Classical SVM Classifier
///
/// This struct wraps the linfa-svm model for comparison with QSVM.
/// It uses the RBF (Radial Basis Function) kernel, which is the most
/// common kernel for classical SVMs.
///
/// RBF Kernel: K(x,y) = exp(-γ||x-y||²)
/// where γ controls the "width" of the kernel
pub struct ClassicalSVM {
    /// The trained linfa SVM model
    model: Option<Svm<f64, bool>>,
}

impl ClassicalSVM {
    /// Creates a new empty ClassicalSVM
    pub fn new() -> Self {
        ClassicalSVM { model: None }
    }

    /// Trains the Classical SVM on the given data.
    ///
    /// # Arguments
    /// * `features` - Training data matrix of shape (n_samples, n_features)
    /// * `targets` - Labels, will be converted to bool (true = positive class)
    /// * `eps` - Tolerance for stopping criterion
    ///
    /// # How it works
    /// 1. Convert targets to boolean format
    /// 2. Create a linfa Dataset
    /// 3. Configure RBF kernel SVM with regularization
    /// 4. Fit the model to the data
    pub fn fit(&mut self, features: &Array2<f64>, targets: &Array1<f64>, eps: f64) {
        // =========================================================================
        // Step 1: Convert targets from f64 (-1.0, +1.0) to bool (false, true)
        // =========================================================================
        // linfa-svm uses boolean targets for binary classification
        let bool_targets: Vec<bool> = targets.iter().map(|&t| t > 0.0).collect();
        let targets_arr = Array1::from_vec(bool_targets);

        // =========================================================================
        // Step 2: Create linfa Dataset
        // =========================================================================
        // A Dataset bundles features and targets together
        let dataset = Dataset::new(features.clone(), targets_arr);

        // =========================================================================
        // Step 3: Configure the SVM with Gaussian/RBF kernel
        // =========================================================================
        // gaussian_kernel() creates an RBF kernel
        // eps() sets the convergence tolerance
        let model = Svm::<f64, bool>::params()
            .gaussian_kernel(80.0) // γ = 80 (kernel width parameter)
            .eps(eps) // Convergence tolerance
            .fit(&dataset) // Train the model
            .expect("SVM training failed");

        // =========================================================================
        // Step 4: Store the trained model
        // =========================================================================
        self.model = Some(model);
        println!("Classical SVM trained successfully!");
    }

    /// Predicts class labels for new samples.
    ///
    /// # Arguments
    /// * `features` - New samples to classify, shape (n_samples, n_features)
    ///
    /// # Returns
    /// Predicted labels (-1.0 or +1.0) for each sample
    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
        match &self.model {
            Some(model) => {
                // Get boolean predictions from linfa-svm
                let predictions = model.predict(features);

                // Convert back to f64 (-1.0 or +1.0) format
                predictions
                    .iter()
                    .map(|&b| if b { 1.0 } else { -1.0 })
                    .collect()
            }
            None => {
                panic!("Model not trained! Call fit() first.");
            }
        }
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
        // Count how many predictions match the actual labels
        let correct: usize = predictions
            .iter()
            .zip(actual.iter())
            .filter(|(p, a)| (*p - *a).abs() < 0.5) // Same sign = correct
            .count();

        correct as f64 / predictions.len() as f64
    }
}

// Implement Default trait for convenience
impl Default for ClassicalSVM {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classical_svm_simple() {
        // Simple linearly separable data
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

        // Train Classical SVM
        let mut model = ClassicalSVM::new();
        model.fit(&features, &targets, 0.001);

        // Predict on training data
        let predictions = model.predict(&features);

        // Calculate accuracy
        let acc = ClassicalSVM::accuracy(&predictions, &targets);
        println!("Classical SVM Training accuracy: {:.2}%", acc * 100.0);

        // Should get good accuracy on linearly separable data
        assert!(acc >= 0.8, "Accuracy should be at least 80%");
    }
}
