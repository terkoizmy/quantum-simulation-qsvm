# Understanding Quantum Kernel - A Deep Dive

This guide explains the quantum kernel code in your QSVM project.

---

## 1. What is a Quantum Kernel?

A **kernel** measures the **similarity** between two data points. In QSVM:

```
Classical Data → Quantum State → Measure Overlap → Similarity Score
```

### The Key Formula

```
K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²
```

| Symbol | Meaning |
|--------|---------|
| x_i, x_j | Two data points (feature vectors) |
| φ(x) | Quantum state encoding of data x |
| ⟨a\|b⟩ | Inner product (overlap) between states |
| \|...\|² | Squared magnitude (probability) |

**Result**: A number between 0 and 1:
- **1.0** = Identical data points
- **0.0** = Completely different (orthogonal states)

---

## 2. The Three Functions

### Function 1: `encode_features(data)` → Quantum State

This converts classical data into a quantum state using a **feature map circuit**.

```rust
pub fn encode_features(data: &[f64]) -> Vec<Complex64>
```

**Input**: `[0.5, 1.0]` (2 features)
**Output**: `[c₀, c₁, c₂, c₃]` (4 complex amplitudes for 2 qubits)

#### The Circuit

```
|0⟩ ─ H ─ RZ(0.5) ─ ●───
                    │
|0⟩ ─ H ─ RZ(1.0) ─ X───

Step 1: H (Hadamard) - Creates superposition
Step 2: RZ(x_i)     - Encodes data into phase
Step 3: CNOT        - Creates entanglement
```

#### Code Breakdown

```rust
// Step 1: Create qubits
let mut qubits: Vec<_> = (0..n_qubits)
    .map(|_| builder.qubit())  // Creates |0⟩ state
    .collect();

// Step 2: Apply Hadamard to each qubit
// H|0⟩ = (|0⟩ + |1⟩)/√2 = |+⟩ (superposition)
qubits = qubits.into_iter()
    .map(|q| builder.h(q))
    .collect();

// Step 3: Apply RZ rotation with data values
// RZ(θ)|+⟩ adds phase e^(iθ) based on the data
qubits = qubits.into_iter()
    .enumerate()
    .map(|(i, q)| builder.rz(q, data[i]))  // data[i] becomes the angle
    .collect();

// Step 4: CNOT entanglement (creates correlations between qubits)
// This is key for capturing feature interactions
for i in 0..(n_qubits - 1) {
    let (q_ctrl, q_targ) = builder.cnot(q_ctrl, q_targ).unwrap();
}

// Step 5: Get the statevector (all 2^n amplitudes)
let (state, _) = builder.calculate_state();
```

#### Visual: What Happens to the State

```
Start:          |00⟩

After H gates:  (|00⟩ + |01⟩ + |10⟩ + |11⟩) / 2

After RZ:       (e^(ix₀)|00⟩ + e^(ix₁)|01⟩ + ...) / 2
                Data is now encoded in the phases!

After CNOT:     Creates entanglement (complex correlations)
```

---

### Function 2: `quantum_kernel(x_i, x_j)` → Similarity Score

Computes the similarity between two data points.

```rust
pub fn quantum_kernel(x_i: &[f64], x_j: &[f64]) -> f64
```

**Input**: Two feature vectors
**Output**: Number in [0, 1]

#### Code Breakdown

```rust
// Step 1: Encode both data points into quantum states
let state_i = encode_features(x_i);  // e.g., [c₀, c₁, c₂, c₃]
let state_j = encode_features(x_j);  // e.g., [d₀, d₁, d₂, d₃]

// Step 2: Compute inner product ⟨φᵢ|φⱼ⟩
// This is: conj(c₀)*d₀ + conj(c₁)*d₁ + conj(c₂)*d₂ + conj(c₃)*d₃
let inner_product: Complex64 = state_i
    .iter()
    .zip(state_j.iter())
    .map(|(a, b)| a.conj() * b)  // conj(a) * b for each pair
    .sum();                        // Sum them all

// Step 3: Return |inner_product|² (squared magnitude)
inner_product.norm_sqr()
```

#### Why `conj(a) * b`?

In quantum mechanics, the inner product ⟨a|b⟩ uses the **conjugate transpose**:

```
⟨a|b⟩ = a₀* × b₀ + a₁* × b₁ + ...

where a* = complex conjugate of a
```

If `a = 3 + 4i`, then `conj(a) = 3 - 4i`

---

### Function 3: `build_kernel_matrix(data)` → N×N Matrix

Builds the full kernel matrix for all training samples.

```rust
pub fn build_kernel_matrix(data: &Array2<f64>) -> Array2<f64>
```

**Input**: N samples × D features
**Output**: N × N kernel matrix

#### The Kernel Matrix

```
       Sample 0   Sample 1   Sample 2
      ┌────────────────────────────┐
Samp 0│  K(0,0)    K(0,1)    K(0,2) │
Samp 1│  K(1,0)    K(1,1)    K(1,2) │
Samp 2│  K(2,0)    K(2,1)    K(2,2) │
      └────────────────────────────┘
```

**Properties**:
- Diagonal = 1.0 (self-similarity is perfect)
- Symmetric: K[i,j] = K[j,i]

#### Code Breakdown

```rust
let n_samples = data.nrows();
let mut kernel = Array2::zeros((n_samples, n_samples));

for i in 0..n_samples {
    for j in i..n_samples {  // Start from i (exploit symmetry!)
        let x_i = data.row(i).to_vec();
        let x_j = data.row(j).to_vec();
        let k_ij = quantum_kernel(&x_i, &x_j);
        
        kernel[[i, j]] = k_ij;
        kernel[[j, i]] = k_ij;  // Symmetric!
    }
}
```

---

## 3. Why Quantum?

### Classical Kernel (RBF)
```
K(x,y) = exp(-γ||x-y||²)
```
Simple distance-based.

### Quantum Kernel
```
K(x,y) = |⟨φ(x)|φ(y)⟩|²
```

The quantum state lives in a **2^n dimensional** space! For 8 qubits:
- Classical: 8 dimensions
- Quantum: 256 dimensions (2^8)

This "quantum feature space" can capture patterns that classical kernels miss.

---

## 4. Summary Diagram

```
┌─────────────┐
│ Data: [0.5] │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ encode_features  │
│                  │
│ |0⟩ → H → RZ →   │──┐
└──────────────────┘  │
                      │
┌─────────────┐       │
│ Data: [0.7] │       │
└──────┬──────┘       │
       │              │
       ▼              │
┌──────────────────┐  │
│ encode_features  │  │
│                  │  │
│ |0⟩ → H → RZ →   │──┤
└──────────────────┘  │
                      │
                      ▼
              ┌───────────────┐
              │quantum_kernel │
              │               │
              │ ⟨φ_i|φ_j⟩ → |.|² │
              └───────┬───────┘
                      │
                      ▼
                  K = 0.82
```

---

## 5. Test Your Understanding

1. **Q**: If `encode_features([1.0, 2.0])` returns 4 amplitudes, how many for `encode_features([1.0, 2.0, 3.0])`?
   
   **A**: 8 (because 2^3 = 8)

2. **Q**: What is `quantum_kernel(x, x)` for any x?
   
   **A**: 1.0 (perfect self-similarity)

3. **Q**: Why do we use `conj(a)` in the inner product?
   
   **A**: Quantum states are complex vectors; the proper inner product uses the conjugate.
