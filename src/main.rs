//! QSVM Bot Detection - Main Entry Point
//!
//! This is the main program that:
//! 1. Loads and preprocesses the bot detection dataset
//! 2. Trains both QSVM and Classical SVM
//! 3. Evaluates and compares both models
//!
//! Run with: cargo run --release

mod classical_svm;
mod data_loader;
mod evaluation;
mod export;
mod preprocessing;
mod qsvm;
mod quantum_kernel;

use classical_svm::ClassicalSVM;
use evaluation::{compare_models, measure_time, ConfusionMatrix, EvaluationResult};
use preprocessing::{normalize_minmax, reduce_pca, train_test_split};
use qsvm::QSVM;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║     QSVM Bot Detection - Quantum vs Classical SVM        ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // Step 1: Load the dataset
    // =========================================================================
    println!("📂 Loading dataset...");
    let dataset = data_loader::load_from_csv("bots_vs_users.csv", "target", None)
        .expect("Failed to load dataset");
    println!(
        "   ✅ Loaded {} samples with {} features",
        dataset.n_samples(),
        dataset.n_features()
    );

    // =========================================================================
    // Step 2: Preprocess the data
    // =========================================================================
    println!("\n🔧 Preprocessing data...");

    // 2a. Normalize features to [0, 1]
    let normalized_features = normalize_minmax(&dataset.features);
    println!("   ✅ Normalized features to [0, 1]");

    // 2b. Apply PCA to reduce from 60 features to 4 (for 4 qubits - much faster!)
    // 4 qubits = 16 amplitudes vs 8 qubits = 256 amplitudes (16x faster!)
    const N_COMPONENTS: usize = 4;
    let pca_features = reduce_pca(&normalized_features, N_COMPONENTS);
    println!("   ✅ Reduced to {} features using PCA", N_COMPONENTS);

    // 2c. Create new dataset with PCA features
    let pca_dataset = data_loader::Dataset::new(
        pca_features,
        dataset.targets.clone(),
        (0..N_COMPONENTS).map(|i| format!("PC{}", i)).collect(),
    );

    // 2d. Split: 200 samples for training, ALL remaining for testing
    // This gives us statistically significant test results
    const TRAIN_SAMPLES: usize = 500;

    // First shuffle all data
    let (shuffled_data, _) = train_test_split(&pca_dataset, 0.001, 42);

    // Split manually: first 200 for training, rest for testing
    let n_total = shuffled_data.n_samples();

    // Training set: first 200 samples
    let train_features = shuffled_data
        .features
        .slice(ndarray::s![..TRAIN_SAMPLES, ..])
        .to_owned();
    let train_targets_raw = shuffled_data
        .targets
        .slice(ndarray::s![..TRAIN_SAMPLES])
        .to_owned();
    let train_data = data_loader::Dataset::new(
        train_features,
        train_targets_raw,
        shuffled_data.feature_names.clone(),
    );

    // Test set: all remaining samples
    let test_features = shuffled_data
        .features
        .slice(ndarray::s![TRAIN_SAMPLES.., ..])
        .to_owned();
    let test_targets_raw = shuffled_data
        .targets
        .slice(ndarray::s![TRAIN_SAMPLES..])
        .to_owned();
    let test_data = data_loader::Dataset::new(
        test_features,
        test_targets_raw,
        shuffled_data.feature_names.clone(),
    );

    println!(
        "   ✅ Split: {} train, {} test samples",
        train_data.n_samples(),
        test_data.n_samples()
    );

    // Convert targets: 0 → -1, 1 → +1 (SVM format)
    let train_targets = train_data
        .targets
        .mapv(|t| if t == 0.0 { -1.0 } else { 1.0 });
    let test_targets = test_data
        .targets
        .mapv(|t| if t == 0.0 { -1.0 } else { 1.0 });

    // =========================================================================
    // Step 3: Train and evaluate QSVM
    // =========================================================================
    println!("\n🔬 Training QSVM (Quantum Support Vector Machine)...");
    println!("   ⚠️  This may take a while due to quantum kernel computation...\n");

    let (qsvm_model, train_time_qsvm) =
        measure_time(|| QSVM::fit(&train_data.features, &train_targets, 1.0, 0.001, 100));

    let (qsvm_predictions, pred_time_qsvm) =
        measure_time(|| qsvm_model.predict(&test_data.features));

    let qsvm_cm = ConfusionMatrix::from_predictions(&qsvm_predictions, &test_targets);
    let qsvm_result = EvaluationResult::new("QSVM", qsvm_cm, train_time_qsvm, pred_time_qsvm);
    qsvm_result.print();

    // =========================================================================
    // Step 4: Train and evaluate Classical SVM
    // =========================================================================
    println!("\n📊 Training Classical SVM (baseline)...");

    let (_, train_time_classical) = measure_time(|| {
        let mut classical = ClassicalSVM::new();
        classical.fit(&train_data.features, &train_targets, 0.001);
        classical
    });

    // Need to create the model again for prediction
    let mut classical_model = ClassicalSVM::new();
    classical_model.fit(&train_data.features, &train_targets, 0.001);

    let (classical_predictions, pred_time_classical) =
        measure_time(|| classical_model.predict(&test_data.features));

    let classical_cm = ConfusionMatrix::from_predictions(&classical_predictions, &test_targets);
    let classical_result = EvaluationResult::new(
        "Classical SVM",
        classical_cm,
        train_time_classical,
        pred_time_classical,
    );
    classical_result.print();

    // =========================================================================
    // Step 5: Compare the models
    // =========================================================================
    compare_models(&qsvm_result, &classical_result);

    // =========================================================================
    // Step 6: Export model and QASM
    // =========================================================================
    println!("\n📤 Exporting model and QASM...");

    // Export QASM for the first test sample
    let sample_data: Vec<f64> = test_data.features.row(0).to_vec();
    let qasm = export::generate_qasm(&sample_data);
    export::save_qasm_to_file(&qasm, "feature_map.qasm").expect("Failed to save QASM");

    // Export trained QSVM model as JSON
    let model_export = export::QSVMExport::from_qsvm(&qsvm_model);
    model_export
        .save_to_json("qsvm_model.json")
        .expect("Failed to save model");

    // Generate Qiskit loader script
    let qiskit_script = export::generate_qiskit_loader();
    std::fs::write("load_qasm.py", qiskit_script).expect("Failed to save Qiskit loader");
    println!("✅ Qiskit loader saved to: load_qasm.py");

    println!("\n✨ All complete! Files exported:");
    println!("   📄 feature_map.qasm - Quantum circuit in OpenQASM 2.0");
    println!("   📄 qsvm_model.json  - Trained model parameters");
    println!("   🐍 load_qasm.py     - Python script to run in Qiskit");
}
