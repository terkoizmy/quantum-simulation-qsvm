# QSVM Thesis Writing Guide

This guide helps you write your thesis and test the model on real quantum hardware.

---

## Part 1: Thesis Structure

### Suggested Chapter Outline

```
1. Introduction
   - Background on bot detection
   - Why quantum computing?
   - Research objectives

2. Literature Review
   - Classical SVM theory
   - Quantum computing basics
   - Quantum kernels and QSVM

3. Methodology
   - Dataset description
   - Preprocessing pipeline
   - Quantum feature map design
   - QSVM implementation

4. Implementation
   - Tools: Rust, qip, linfa
   - System architecture
   - Code explanation

5. Results & Analysis
   - Experimental setup
   - Accuracy comparison
   - Performance analysis
   - Discussion

6. Conclusion
   - Summary
   - Limitations
   - Future work
```

---

## Part 2: Key Sections to Write

### 3.1 Dataset (Methodology)

```
The dataset contains 5,874 social media accounts with 59 features.
Features include: follower count, following count, tweet frequency, etc.
Target: Binary classification (Bot = 1, Human = 0)

Preprocessing:
1. Min-max normalization to [0,1]
2. PCA dimensionality reduction (59 → 4 features)
3. Train/test split (300 train, ~5500 test)
```

### 3.2 Quantum Feature Map

```
We use the ZZ Feature Map circuit:

|0⟩ ─ H ─ RZ(x₀) ─ ●───
                   │
|0⟩ ─ H ─ RZ(x₁) ─ X───

Steps:
1. Hadamard gates create superposition
2. RZ rotations encode data into phases
3. CNOT gates create entanglement

This maps classical data into 2^n dimensional Hilbert space.
```

### 3.3 Quantum Kernel

```
The quantum kernel measures similarity:

K(xᵢ, xⱼ) = |⟨φ(xᵢ)|φ(xⱼ)⟩|²

- φ(x) is the quantum state from the feature map
- ⟨a|b⟩ is the inner product (overlap)
- Result is in [0, 1]: 1 = identical, 0 = orthogonal
```

### 5.1 Results Table

```
| Metric     | QSVM   | Classical SVM |
|------------|--------|---------------|
| Accuracy   | 92.5%  | 95.0%         |
| Precision  | 86.4%  | 90.5%         |
| Recall     | 100%   | 100%          |
| F1 Score   | 92.7%  | 95.0%         |
| Train Time | 263s   | 0.001s        |
```

### 5.2 Discussion Points

1. **Why Classical SVM performed slightly better:**
   - Limited qubit count (4 qubits = 16-dim space)
   - Simulation noise vs. exact classical computation
   - Small training set (300 samples)

2. **Potential quantum advantages:**
   - Larger feature spaces with more qubits
   - Complex non-linear patterns
   - Real quantum hardware may differ

3. **Limitations:**
   - Simulation is slow (O(2^n))
   - Real hardware has noise/errors
   - Current NISQ devices have limited qubits

---

## Part 3: Figures to Include

1. **System Architecture Diagram**
2. **Quantum Circuit Diagram** (from QASM)
3. **Confusion Matrix** (both models)
4. **Accuracy Comparison Bar Chart**
5. **Training Time Comparison**

---

## Part 4: Testing on Real Quantum Hardware

See the separate file: `qiskit_tutorial.py` (in docs folder)

You can run this on Google Colab to:
1. Load your QASM file
2. Visualize the circuit
3. Run on IBM Quantum simulator
4. (Optional) Run on real IBM quantum computer
