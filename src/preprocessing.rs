//! Preprocessing Module for QSVM Bot Detection
//!
//! This module handles data cleaning, normalization, dimensionality reduction (PCA),
//! and train/test splitting.

use crate::data_loader::Dataset;
use linfa::traits::{Fit, Predict};
use linfa_reduction::Pca;
use ndarray::{Array1, Array2, Axis, IndexLonger};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

// =============================================================================
// CHALLENGE 1: Implement normalize_minmax
// =============================================================================
/// Normalizes features to the range [0, 1] using Min-Max scaling.
///
/// # Formula
/// For each feature column:
/// ```text
/// x_normalized = (x - min) / (max - min)
/// ```
///
/// # Arguments
/// * `features` - 2D array of shape (n_samples, n_features)
///
/// # Returns
/// * Normalized features in range [0, 1]
///
/// # Steps
/// 1. For each column, find the min and max values
/// 2. Apply the formula: (x - min) / (max - min)
/// 3. Handle edge case: if max == min, set to 0.0
///
/// # Hints
/// - Use `features.column(i)` to get column i
/// - Use `.fold()` or iterate to find min/max
/// - Use `features.mapv()` for element-wise operations
pub fn normalize_minmax(features: &Array2<f64>) -> Array2<f64> {
    // TODO: Implement min-max normalization
    //
    // Step 1: Create output array with same shape
    // let mut normalized = features.clone();
    let mut normalized = features.clone();
    // Step 2: For each column (feature):
    // for col_idx in 0..features.ncols() {
    //     let column = features.column(col_idx);
    //     let min = ... // find minimum
    //     let max = ... // find maximum
    //     let range = max - min;
    //
    //     // Step 3: Normalize each value in this column
    //     for row_idx in 0..features.nrows() {
    //         if range > 0.0 {
    //             normalized[[row_idx, col_idx]] = (features[[row_idx, col_idx]] - min) / range;
    //         } else {
    //             normalized[[row_idx, col_idx]] = 0.0;
    //         }
    //     }
    // }
    for col_idx in 0..features.ncols() {
        let column = features.column(col_idx);
        let min = column.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        let max = column.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        let range = max - min;
        for row_idx in 0..features.nrows() {
            if range > 0.0 {
                normalized[[row_idx, col_idx]] = (features[[row_idx, col_idx]] - min) / range;
            } else {
                normalized[[row_idx, col_idx]] = 0.0;
            }
        }
    }
    //
    // Step 4: Return normalized array
    normalized

    // todo!("Implement normalize_minmax")
}

// =============================================================================
// CHALLENGE 2: Implement reduce_pca
// =============================================================================
/// Reduces dimensionality using Principal Component Analysis (PCA).
///
/// # What is PCA?
/// PCA finds the directions (principal components) of maximum variance
/// in the data and projects the data onto these directions.
///
/// # Arguments
/// * `features` - 2D array of shape (n_samples, n_features)
/// * `n_components` - Number of components to keep (e.g., 8)
///
/// # Returns
/// * Reduced features of shape (n_samples, n_components)
///
/// # Steps using linfa
/// 1. Convert ndarray to linfa Dataset format
/// 2. Create PCA transformer with n_components
/// 3. Fit and transform the data
/// 4. Extract the transformed array
///
/// # Hints
/// ```ignore
/// use linfa::traits::{Fit, Predict};
/// use linfa_reduction::Pca;
///
/// // Create PCA with n_components
/// let pca = Pca::params(n_components);
///
/// // Fit and transform
/// let dataset = linfa::DatasetBase::from(features.clone());
/// let pca_fitted = pca.fit(&dataset)?;
/// let transformed = pca_fitted.predict(&dataset);
/// ```
pub fn reduce_pca(features: &Array2<f64>, n_components: usize) -> Array2<f64> {
    // TODO: Implement PCA using linfa-reduction
    //
    // Step 2: Create linfa dataset from features
    // let dataset = linfa::DatasetBase::from(features.clone());
    let dataset = linfa::DatasetBase::from(features.clone());
    //
    // Step 3: Create and fit PCA
    // let pca = Pca::params(n_components)
    //     .fit(&dataset)
    //     .expect("PCA fitting failed");
    let pca = Pca::params(n_components)
        .fit(&dataset)
        .expect("PCA fitting failed");

    //
    // Step 4: Transform the data
    // let transformed = pca.predict(&dataset);
    let trasformed = pca.predict(&dataset);
    //
    // Step 5: Return the records (features) from transformed dataset
    // transformed.records().clone()
    trasformed

    // todo!("Implement reduce_pca")
}

// =============================================================================
// CHALLENGE 3: Implement train_test_split
// =============================================================================
/// Splits the dataset into training and testing sets.
///
/// # Arguments
/// * `dataset` - The full dataset to split
/// * `test_ratio` - Fraction for testing (e.g., 0.2 = 20% test)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// * Tuple of (train_dataset, test_dataset)
///
/// # Steps
/// 1. Generate shuffled indices
/// 2. Calculate split point
/// 3. Split features and targets
/// 4. Create two new Dataset structs
///
/// # Hints
/// - Use `rand::rngs::StdRng::seed_from_u64(seed)` for reproducibility
/// - Use `.shuffle(&mut rng)` on a Vec of indices
/// - Use ndarray slicing: `features.select(Axis(0), &indices)`
pub fn train_test_split(dataset: &Dataset, test_ratio: f64, seed: u64) -> (Dataset, Dataset) {
    // TODO: Implement train/test split
    //
    // Step 1: Create shuffled indices
    let n_samples = dataset.n_samples();
    let mut indices: Vec<usize> = (0..n_samples).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);
    //
    // Step 2: Calculate split point
    let test_size = (n_samples as f64 * test_ratio) as usize;
    let train_size = n_samples - test_size;
    //
    // Step 3: Split indices
    let train_indices: Vec<usize> = indices[..train_size].to_vec();
    let test_indices: Vec<usize> = indices[train_size..].to_vec();
    //
    // Step 4: Select rows for each split
    let train_features = dataset.features.select(Axis(0), &train_indices);
    let train_targets = dataset.targets.select(Axis(0), &train_indices);
    let test_features = dataset.features.select(Axis(0), &test_indices);
    let test_targets = dataset.targets.select(Axis(0), &test_indices);
    //
    // Step 5: Create new datasets
    let train = Dataset::new(train_features, train_targets, dataset.feature_names.clone());
    let test = Dataset::new(test_features, test_targets, dataset.feature_names.clone());

    (train, test)

    // todo!("Implement train_test_split")
}

// =============================================================================
// BONUS: Clean data function
// =============================================================================
/// Removes rows that have any NaN or infinite values.
///
/// # Arguments
/// * `dataset` - Dataset to clean
///
/// # Returns
/// * Cleaned dataset with invalid rows removed
pub fn clean_data(dataset: &Dataset) -> Dataset {
    // TODO: (BONUS) Remove rows with NaN or Inf values
    //
    // Step 1: Find valid row indices
    let mut valid_indices: Vec<usize> = Vec::new();
    for row_idx in 0..dataset.n_samples() {
        let row = dataset.features.row(row_idx);
        let is_valid = row.iter().all(|&x| x.is_finite());
        if is_valid {
            valid_indices.push(row_idx);
        }
    }
    //
    // Step 2: Select only valid rows
    let valid_features = dataset.features.select(Axis(0), &valid_indices);
    let valid_targets = dataset.targets.select(Axis(0), &valid_indices);
    // Step 3: Create new dataset
    let cleaned = Dataset::new(valid_features, valid_targets, dataset.feature_names.clone());
    cleaned

    // todo!("Implement clean_data (BONUS)")
}

// =============================================================================
// TESTS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_minmax() {
        let features =
            Array2::from_shape_vec((3, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap();

        let normalized = normalize_minmax(&features);

        // First column: [1,2,3] -> [0, 0.5, 1]
        assert!((normalized[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((normalized[[1, 0]] - 0.5).abs() < 1e-6);
        assert!((normalized[[2, 0]] - 1.0).abs() < 1e-6);

        // Second column: [10,20,30] -> [0, 0.5, 1]
        assert!((normalized[[0, 1]] - 0.0).abs() < 1e-6);
        assert!((normalized[[1, 1]] - 0.5).abs() < 1e-6);
        assert!((normalized[[2, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_train_test_split() {
        let features = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0,
            ],
        )
        .unwrap();
        let targets = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let names = vec!["f1".to_string(), "f2".to_string()];

        let dataset = Dataset::new(features, targets, names);
        let (train, test) = train_test_split(&dataset, 0.2, 42);

        // 20% of 10 = 2 test samples, 8 train samples
        assert_eq!(train.n_samples(), 8);
        assert_eq!(test.n_samples(), 2);
    }

    #[test]
    fn test_reduce_pca() {
        // Create 5 samples with 4 features - with INDEPENDENT variance
        // Each column should have different patterns to get 2+ components
        let features = Array2::from_shape_vec(
            (5, 4),
            vec![
                1.0, 10.0, 1.0, 5.0, // sample 1
                2.0, 8.0, 4.0, 3.0, // sample 2
                3.0, 6.0, 9.0, 7.0, // sample 3
                4.0, 4.0, 16.0, 2.0, // sample 4
                5.0, 2.0, 25.0, 9.0, // sample 5
            ],
        )
        .unwrap();

        let reduced = reduce_pca(&features, 2);

        // Should have 5 samples with 2 components
        assert_eq!(reduced.shape(), &[5, 2]);
    }
}
