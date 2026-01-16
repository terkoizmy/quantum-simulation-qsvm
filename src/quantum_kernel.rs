//! Quantum Kernel Module for QSVM
//!
//! This module implements the quantum feature map and kernel computation
//! using the qip quantum simulation library.
//!
//! # Quantum Kernel Overview
//!
//! A quantum kernel computes the similarity between two data points by:
//! 1. Encoding each data point into a quantum state using a feature map
//! 2. Computing the overlap (fidelity) between the two quantum states
//!
//! K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²

use ndarray::Array2;
use num_complex::Complex64;
use qip::prelude::*;
use std::num::NonZeroUsize;

// =============================================================================
// CHALLENGE 1: Implement the ZZ Feature Map
// =============================================================================
/// Creates a quantum circuit that encodes classical data into a quantum state.
///
/// # ZZ Feature Map
/// For data x = [x₀, x₁, ..., xₙ₋₁]:
///
/// 1. Apply Hadamard to all qubits: H|0⟩ = |+⟩
/// 2. Apply rotation RZ(xᵢ) to each qubit i
/// 3. Apply entangling layers with CNOT + RZ(xᵢ·xⱼ)
///
/// Circuit diagram (2 qubits):
/// ```text
/// |0⟩ ─ H ─ RZ(x₀) ─ ● ─ RZ(x₀·x₁) ─ ● ─
///                    │               │
/// |0⟩ ─ H ─ RZ(x₁) ─ X ───────────── X ─
/// ```
///
/// # Arguments
/// * `data` - Slice of f64 features (length = n_qubits)
///
/// # Returns
/// * `Vec<Complex64>` - The statevector after encoding
///
/// # Hints
/// ```ignore
/// use qip::prelude::*;
/// use std::num::NonZeroUsize;
///
/// let n = NonZeroUsize::new(n_qubits).unwrap();
/// let mut builder = LocalBuilder::<f64>::default();
/// let q = builder.register(n);
///
/// // Apply Hadamard to each qubit
/// let q = builder.h(q);
///
/// // Apply RZ rotations (phase gates)
/// // qip uses different gate names - check docs
///
/// // Get the statevector
/// let (state, _) = builder.calculate_state();
/// ```
pub fn encode_features(data: &[f64]) -> Vec<Complex64> {
    let n_qubits = data.len();

    // Handle edge case
    if n_qubits == 0 {
        return vec![Complex64::new(1.0, 0.0)];
    }

    let mut builder = LocalBuilder::<f64>::default();

    // Step 1: Create individual qubits
    let mut qubits: Vec<_> = (0..n_qubits).map(|_| builder.qubit()).collect();

    // Step 2: Apply Hadamard to each qubit
    qubits = qubits.into_iter().map(|q| builder.h(q)).collect();

    // Step 3: Apply RZ(data[i]) rotation to each qubit
    qubits = qubits
        .into_iter()
        .enumerate()
        .map(|(i, q)| builder.rz(q, data[i]))
        .collect();

    // Step 4: Apply entangling layer (CNOT between adjacent qubits)
    if n_qubits > 1 {
        for i in 0..(n_qubits - 1) {
            let q_ctrl = qubits.remove(i);
            let q_targ = qubits.remove(i);
            let (q_ctrl, q_targ) = builder.cnot(q_ctrl, q_targ).unwrap();
            qubits.insert(i, q_ctrl);
            qubits.insert(i + 1, q_targ);
        }
    }

    // Keep qubits in scope for state calculation
    let _ = qubits;

    // Step 5: Calculate the statevector
    let (state, _) = builder.calculate_state();

    // Step 6: Convert to Vec<Complex64>
    state
        .into_iter()
        .map(|c| Complex64::new(c.re, c.im))
        .collect()
}

// =============================================================================
// CHALLENGE 2: Implement quantum kernel element
// =============================================================================
/// Computes the quantum kernel between two data points.
///
/// # Formula
/// K(xᵢ, xⱼ) = |⟨φ(xᵢ)|φ(xⱼ)⟩|²
///
/// This is the fidelity (squared overlap) between the two quantum states.
///
/// # Arguments
/// * `x_i` - First data point features
/// * `x_j` - Second data point features
///
/// # Returns
/// * `f64` - Kernel value in range [0, 1]
///
/// # Steps
/// 1. Encode x_i into quantum state φ(x_i)
/// 2. Encode x_j into quantum state φ(x_j)
/// 3. Compute inner product ⟨φ(x_i)|φ(x_j)⟩
/// 4. Return |inner_product|²
pub fn quantum_kernel(x_i: &[f64], x_j: &[f64]) -> f64 {
    // TODO: Implement quantum kernel computation
    //
    // Step 1: Encode both data points
    let state_i = encode_features(x_i);
    let state_j = encode_features(x_j);
    //
    // Step 2: Compute inner product ⟨φᵢ|φⱼ⟩
    // Inner product for complex vectors:
    // ⟨a|b⟩ = Σ conj(aᵢ) * bᵢ
    //
    let inner_product: Complex64 = state_i
        .iter()
        .zip(state_j.iter())
        .map(|(a, b)| a.conj() * b)
        .sum();

    // Step 3: Return fidelity = |⟨φᵢ|φⱼ⟩|²
    inner_product.norm_sqr()

    // todo!("Implement quantum_kernel")
}

// =============================================================================
// CHALLENGE 3: Build kernel matrix
// =============================================================================
/// Builds the full kernel matrix for a dataset.
///
/// # Arguments
/// * `data` - Feature matrix of shape (n_samples, n_features)
///
/// # Returns
/// * `Array2<f64>` - Kernel matrix of shape (n_samples, n_samples)
///
/// # Properties
/// - K[i,j] = K[j,i] (symmetric)
/// - K[i,i] = 1.0 (self-similarity is perfect)
///
/// # Example
/// For 3 samples, the kernel matrix is:
/// ```text
/// K = [K(x₀,x₀)  K(x₀,x₁)  K(x₀,x₂)]
///     [K(x₁,x₀)  K(x₁,x₁)  K(x₁,x₂)]
///     [K(x₂,x₀)  K(x₂,x₁)  K(x₂,x₂)]
/// ```
///
/// # Performance
/// Uses Rayon for parallel computation across all CPU cores.
/// Speedup is approximately linear with number of cores.
pub fn build_kernel_matrix(data: &Array2<f64>) -> Array2<f64> {
    use rayon::prelude::*;
    use std::sync::Mutex;

    let n_samples = data.nrows();

    // Use a Mutex to safely update the shared kernel matrix from multiple threads
    let kernel = Mutex::new(Array2::zeros((n_samples, n_samples)));

    // Progress counter for tracking
    let progress = Mutex::new(0usize);

    // Parallel iteration over rows using Rayon
    // Each thread computes one row of the upper triangle
    (0..n_samples).into_par_iter().for_each(|i| {
        let x_i = data.row(i).to_vec();

        // Compute this row's kernel values (upper triangle only)
        let mut row_values = Vec::with_capacity(n_samples - i);
        for j in i..n_samples {
            let x_j = data.row(j).to_vec();
            let k_ij = quantum_kernel(&x_i, &x_j);
            row_values.push((j, k_ij));
        }

        // Update the kernel matrix (thread-safe)
        {
            let mut k = kernel.lock().unwrap();
            for (j, k_ij) in row_values {
                k[[i, j]] = k_ij;
                k[[j, i]] = k_ij; // Symmetry
            }
        }

        // Update progress (thread-safe)
        {
            let mut p = progress.lock().unwrap();
            *p += 1;
            if *p % 50 == 0 || *p == n_samples {
                println!(
                    "Computing kernel: {}/{} ({:.0}%)",
                    *p,
                    n_samples,
                    *p as f64 / n_samples as f64 * 100.0
                );
            }
        }
    });

    // Extract the final matrix from the Mutex
    kernel.into_inner().unwrap()
}

// =============================================================================
// HELPER: Simplified feature map for testing
// =============================================================================
/// A simpler feature encoding for testing (classical approximation).
///
/// This creates a unit vector with phase encoding:
/// φ(x) = [e^(ix₀), e^(ix₁), ...] / ||...||
pub fn encode_features_simple(data: &[f64]) -> Vec<Complex64> {
    let n = data.len();
    if n == 0 {
        return vec![Complex64::new(1.0, 0.0)];
    }

    // Create complex amplitudes from data
    let state: Vec<Complex64> = data
        .iter()
        .map(|&x| Complex64::new(x.cos(), x.sin()))
        .collect();

    // Normalize
    let norm: f64 = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    state.into_iter().map(|c| c / norm).collect()
}

/// Simple kernel using the simplified encoding (for testing)
pub fn quantum_kernel_simple(x_i: &[f64], x_j: &[f64]) -> f64 {
    let state_i = encode_features_simple(x_i);
    let state_j = encode_features_simple(x_j);

    let inner: Complex64 = state_i
        .iter()
        .zip(state_j.iter())
        .map(|(a, b)| a.conj() * b)
        .sum();

    inner.norm_sqr()
}

// =============================================================================
// TESTS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_features_simple() {
        let data = vec![0.5, 1.0, 1.5];
        let state = encode_features_simple(&data);

        // Check state is normalized (sum of |amplitude|² = 1)
        let norm_sqr: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm_sqr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_kernel_simple_identical() {
        // Identical vectors should have kernel ≈ 1.0
        let x = vec![0.5, 1.0, 1.5];
        let k = quantum_kernel_simple(&x, &x);
        assert!((k - 1.0).abs() < 1e-10, "Self-kernel should be 1.0");
    }

    #[test]
    fn test_quantum_kernel_simple_different() {
        // Different vectors should have kernel < 1.0
        let x = vec![0.5, 1.0, 1.5];
        let y = vec![1.5, 0.5, 0.0];
        let k = quantum_kernel_simple(&x, &y);
        assert!(k >= 0.0 && k <= 1.0, "Kernel should be in [0,1]");
        assert!(k < 1.0, "Different vectors should have kernel < 1.0");
    }

    #[test]
    fn test_quantum_kernel_simple_symmetric() {
        // Kernel should be symmetric: K(x,y) = K(y,x)
        let x = vec![0.5, 1.0];
        let y = vec![1.5, 0.5];
        let k_xy = quantum_kernel_simple(&x, &y);
        let k_yx = quantum_kernel_simple(&y, &x);
        assert!((k_xy - k_yx).abs() < 1e-10, "Kernel should be symmetric");
    }

    #[test]
    fn test_encode_features_qip() {
        // Test that encode_features produces a valid quantum state
        let data = vec![0.5, 1.0];
        let state = encode_features(&data);

        // State should have 2^n_qubits = 4 amplitudes
        assert_eq!(state.len(), 4, "2 qubits should give 4 amplitudes");

        // State should be normalized (sum of |amplitude|² = 1)
        let norm_sqr: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm_sqr - 1.0).abs() < 1e-10, "State should be normalized");
    }
}
