//! Data Loader Module for QSVM Bot Detection
//!
//! This module handles loading CSV data using Polars and converting it to ndarray format.

use ndarray::{Array1, Array2};
use polars::prelude::*;
use std::error::Error;
use std::path::Path;

// =============================================================================
// Dataset struct - Already implemented ✅
// =============================================================================
#[derive(Debug, Clone)]
pub struct Dataset {
    pub features: Array2<f64>,
    pub targets: Array1<f64>,
    pub feature_names: Vec<String>,
}

impl Dataset {
    pub fn new(features: Array2<f64>, targets: Array1<f64>, feature_names: Vec<String>) -> Self {
        Self {
            features,
            targets,
            feature_names,
        }
    }

    pub fn n_samples(&self) -> usize {
        self.features.nrows()
    }

    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }

    pub fn features(&self) -> &Array2<f64> {
        &self.features
    }

    pub fn targets(&self) -> &Array1<f64> {
        &self.targets
    }
}

// =============================================================================
// CHALLENGE: Implement load_from_csv using Polars
// =============================================================================
/// Loads a dataset from a CSV file using Polars.
///
/// # Polars Key Concepts:
/// - `DataFrame`: Like a table with named columns
/// - `Series`: A single column of data
/// - `LazyFrame`: Lazy evaluation for optimized queries
///
/// # Polars API Quick Reference:
/// ```ignore
/// // Read CSV
/// let df = CsvReadOptions::default()
///     .with_has_header(true)
///     .try_into_reader_with_file_path(Some(path.into()))?
///     .finish()?;
///
/// // Get column names
/// let columns: Vec<String> = df.get_column_names_str().iter().map(|s| s.to_string()).collect();
///
/// // Get a column as Series
/// let series: &Series = df.column("column_name")?;
///
/// // Convert Series to Vec<f64>
/// let values: Vec<f64> = series.f64()?.into_no_null_iter().collect();
///
/// // Select specific columns
/// let selected = df.select(["col1", "col2"])?;
///
/// // Drop a column
/// let without_target = df.drop("target")?;
/// ```
pub fn load_from_csv<P: AsRef<Path>>(
    path: P,
    target_column: &str,
    feature_columns: Option<&[&str]>,
) -> Result<Dataset, Box<dyn Error>> {
    // =========================================================================
    // Step 1: Read CSV with Polars
    // =========================================================================
    // Configure CSV reader to treat "Unknown" as null values
    let null_values = NullValues::AllColumnsSingle("Unknown".into());

    let parse_options = CsvParseOptions::default().with_null_values(Some(null_values));

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_parse_options(parse_options)
        .try_into_reader_with_file_path(Some(path.as_ref().into()))?
        .finish()?;

    // Fill null values with 0.0 for all columns
    let df = df.fill_null(FillNullStrategy::Zero)?;

    println!("Loaded DataFrame with shape: {:?}", df.shape());

    // =========================================================================
    // Step 2: Extract target column
    // =========================================================================
    // TODO: Get the target column and convert to Vec<f64>
    // Hint:
    //   let target_series = df.column(target_column)?;
    //   let targets_vec: Vec<f64> = target_series.cast(&DataType::Float64)?
    //       .f64()?
    //       .into_no_null_iter()
    //       .collect();
    let target_series = df.column(target_column)?;
    let targets_vec: Vec<f64> = target_series
        .cast(&DataType::Float64)?
        .f64()?
        .into_no_null_iter()
        .collect();

    // =========================================================================
    // Step 3: Select feature columns
    // =========================================================================
    // TODO: Select either the specified columns or all columns except target
    // Hint for Option 1 (specific columns):
    //   let feature_df = df.select(feature_columns)?;
    // Hint for Option 2 (all except target):
    //   let feature_df = df.drop(target_column)?;
    let (feature_df, feature_names): (DataFrame, Vec<String>) = match feature_columns {
        Some(cols) => {
            // Convert &[&str] to Vec<String> for Polars select
            let col_names: Vec<String> = cols.iter().map(|s| s.to_string()).collect();
            let selected = df.select(col_names.iter().map(|s| s.as_str()))?;
            (selected, col_names)
        }
        None => {
            let dropped = df.drop(target_column)?;
            let names = dropped
                .get_column_names_str()
                .iter()
                .map(|s| s.to_string())
                .collect();
            (dropped, names)
        }
    };

    // =========================================================================
    // Step 4: Convert DataFrame to Array2<f64>
    // =========================================================================
    // TODO: Convert each column to f64 and build the feature matrix
    // Hint: Iterate through columns, cast to f64, collect into a flat Vec,
    //       then use Array2::from_shape_vec
    let n_samples = feature_df.height();
    let n_features = feature_df.width();

    let mut flat_data: Vec<f64> = Vec::with_capacity(n_samples * n_features);

    // Polars stores data column-major, but we need row-major for ndarray
    // So we need to transpose the data
    for row_idx in 0..n_samples {
        for col_idx in 0..n_features {
            let series = feature_df.get_columns()[col_idx].cast(&DataType::Float64)?;
            let value = series.f64()?.get(row_idx).unwrap_or(0.0);
            flat_data.push(value);
        }
    }

    let features = Array2::from_shape_vec((n_samples, n_features), flat_data)?;

    // =========================================================================
    // Step 5: Convert targets to Array1<f64>
    // =========================================================================
    let targets = Array1::from_vec(targets_vec);

    // =========================================================================
    // Step 6: Return the Dataset
    // =========================================================================
    Ok(Dataset::new(features, targets, feature_names))
}

// =============================================================================
// BONUS: More efficient conversion using Polars to_ndarray (requires feature)
// =============================================================================
// If you add `features = ["ndarray"]` to polars in Cargo.toml, you can use:
// let ndarray = df.to_ndarray::<Float64Type>()?;

// =============================================================================
// TESTS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_new() {
        let features = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let targets = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let names = vec!["f1".to_string(), "f2".to_string()];

        let dataset = Dataset::new(features, targets, names);

        assert_eq!(dataset.n_samples(), 3);
        assert_eq!(dataset.n_features(), 2);
    }

    #[test]
    #[ignore] // Remove when ready to test with real file
    fn test_load_from_csv_polars() {
        let result = load_from_csv(
            "bots_vs_users.csv",
            "target",
            Some(&["has_domain", "has_birth_date", "has_photo"]),
        );

        assert!(result.is_ok());
        let dataset = result.unwrap();
        assert!(dataset.n_samples() > 0);
        assert_eq!(dataset.n_features(), 3);
    }
}
