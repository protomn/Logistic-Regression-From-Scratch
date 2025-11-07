# Logistic Regression From Scratch

A complete implementation of logistic regression built from the ground up using only NumPy, with no machine learning libraries. This project demonstrates my understanding of the mathematical foundations and implementation details of binary classification.

## Project Highlights

- **97.37% accuracy** on the Wisconsin Breast Cancer dataset
- **Outperformed sklearn** (96.49%) on real medical data
- Built entirely from scratch with only NumPy
- Includes early stopping, gradient descent optimization, and comprehensive evaluation


## Results Summary

| Metric | Custom Implementation | Sklearn | Difference |
|--------|----------------------|---------|------------|
| **Test Accuracy** | 97.37% | 96.49% | +0.88% |
| **Training Time** | 0.98s | 0.10s | - |
| **Misclassifications** | 3/114 | 4/114 | -1 |
| **Implementation** | Pure Python + NumPy | Optimized C++ | - |

## Mathematical Foundation

### 1. The Sigmoid Function

The sigmoid function maps any real number to the range [0, 1], making it perfect for probability estimation:

```
σ(z) = 1 / (1 + e^(-z))
```

**Properties:**
- Domain: (-∞, +∞)
- Range: (0, 1)
- σ(0) = 0.5
- σ(+∞) → 1
- σ(-∞) → 0

### 2. The Model

For input features **X** and parameters **w** (weights) and **b** (bias):

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = w^T x + b
ŷ = σ(z) = 1 / (1 + e^(-w^T x - b))
```

Where ŷ is the predicted probability of the positive class.

### 3. Binary Cross-Entropy Loss

The loss function for a single example:

```
L(y, ŷ) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

For the entire dataset with m examples:

```
J(w, b) = -(1/m) Σ [y⁽ⁱ⁾·log(ŷ⁽ⁱ⁾) + (1-y⁽ⁱ⁾)·log(1-ŷ⁽ⁱ⁾)]
```

**Why not Mean Squared Error?**
- MSE creates a non-convex optimization landscape for logistic regression
- Multiple local minima make training unreliable
- Cross-entropy is convex with a single global minimum

### 4. Gradient Descent

Update rules for optimization:

```
w = w - α · ∂J/∂w
b = b - α · ∂J/∂b
```

Where α is the learning rate and the gradients are:

```
∂J/∂w = (1/m) · X^T · (ŷ - y)
∂J/∂b = (1/m) · Σ(ŷ - y)
```

## Implementation Details

### Core Components

1. **Sigmoid Function** (`sigmoid`)
   - Numerically stable implementation
   - Handles extreme values without overflow

2. **Forward Pass** (`forward_pass`)
   - Computes predictions for all examples
   - Vectorized for efficiency

3. **Loss Function** (`bin_cross_entropy`)
   - Binary cross-entropy with numerical stability
   - Clips predictions to avoid log(0)

4. **Gradient Computation** (`compute_grad`)
   - Calculates gradients for weights and bias
   - Fully vectorized using matrix operations

5. **Parameter Updates** (`update_parameters`)
   - Applies gradient descent step
   - Updates both weights and bias

6. **Training Loop** (`train`)
   - Orchestrates the learning process
   - Includes optional early stopping
   - Tracks loss history for visualization

7. **Prediction** (`predict`)
   - Makes binary predictions with threshold
   - Returns both predictions and probabilities

### Key Features

 **Early Stopping**
- Monitors validation loss
- Stops training when no improvement
- Saves best parameters automatically

 **Vectorization**
- All operations use NumPy array operations
- No explicit Python loops over data
- Efficient matrix multiplications

 **Numerical Stability**
- Clipping to prevent log(0)
- Careful handling of extreme sigmoid inputs


## Getting Started

### Basic Usage

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize features (important!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
w, b, losses = train(
    X_train_scaled, 
    y_train,
    learning_rate=0.1,
    epochs=5000,
    early_stop=True,
    patience=100,
    verbose=True
)

# Make predictions
y_pred, y_prob = predict(X_test_scaled, w, b)

# Evaluate
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
```

## Training Process

### Without Early Stopping

```python
w, b, losses = train(
    X_train, y_train,
    learning_rate=0.1,
    epochs=1000,
    early_stop=False
)
```

### With Early Stopping

```python
w, b, losses = train(
    X_train, y_train,
    learning_rate=0.1,
    epochs=5000,
    early_stop=True,
    patience=100,      # Stop after 100 epochs without improvement
    min_delta=1e-5,    # Minimum change to count as improvement
    verbose=True
)
```

## What I Learned

### Mathematical Concepts
- Sigmoid function and its properties
- Why cross-entropy is superior to MSE for classification
- Gradient descent optimization
- Partial derivatives and the chain rule
- Convex vs non-convex optimization

### Implementation Skills
- Vectorization with NumPy
- Numerical stability considerations
- Hyperparameter tuning (learning rate, epochs)
- Early stopping strategies
- Model evaluation and benchmarking

### Machine Learning Principles
- Train/test split importance
- Feature standardization necessity
- Overfitting prevention
- Model comparison methodologies

##  Experiments & Results

### Dataset: Wisconsin Breast Cancer

**Characteristics:**
- 569 samples
- 30 features (tumor measurements)
- Binary classification: Benign (0) vs Malignant (1)
- Class distribution: 357 benign, 212 malignant

**Preprocessing:**
- Train/test split: 80/20
- Feature standardization (zero mean, unit variance)
- No feature selection or engineering

**Final Model Performance:**

```
Confusion Matrix:
[[71  1]    # 71 benign correctly classified, 1 false positive
 [ 2 40]]   # 2 false negatives, 40 malignant correctly classified

Classification Report:
              precision    recall  f1-score   support
      Benign       0.97      0.99      0.98        72
   Malignant       0.98      0.95      0.96        42
    accuracy                           0.97       114
```

**Key Insights:**
- High precision and recall for both classes
- Only 3 total misclassifications
- Better than sklearn on this particular train/test split
- Low false negative rate (2/42) crucial for medical diagnosis

## Hyperparameter Tuning

### Learning Rate
- **Too low** (0.001): Slow convergence, many epochs needed
- **Too high** (1.0): Loss explodes, numerical instability
- **Optimal** (0.1): Fast convergence, stable training

### Early Stopping
- **Patience**: 50-100 epochs works well
- **Min Delta**: 1e-5 to 1e-4 prevents premature stopping
- Saved ~4000 epochs in testing

### Epochs
- Without early stopping: 5000+ epochs
- With early stopping: Typically stops at 500-1500 epochs

## Visualization Examples

### Loss Curve
Shows smooth descent of loss over training, demonstrating successful optimization.

### Decision Boundary
For 2D data, visualizes the linear separation learned by the model.

### Feature Importance
Weight magnitudes indicate which features matter most for classification.

## Comparison with Sklearn

### Advantages of Custom Implementation
Customizable for specific needs   
Achieved higher accuracy in this case

### Advantages of Sklearn
10x faster (optimized C++)  
More sophisticated solvers  
Built-in regularization  
More features (multi-class, regularization options)

## References & Resources

### Mathematical Background
- [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)
- [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
- [Cross-Entropy Loss](https://en.wikipedia.org/wiki/Cross_entropy)

### Implementation Guides
- [NumPy Documentation](https://numpy.org/doc/)
- [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Datasets
- [UCI Machine Learning Repository - Breast Cancer Wisconsin](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

## Contributing

This is a learning project, but suggestions and improvements are welcome! Feel free to:
- Report bugs or issues
- Suggest enhancements
- Share your own implementations

## License

This project is open source and available for educational purposes.

## Acknowledgments

- Built as a deep learning exercise from first principles
- Tested on real-world medical data
- Achieved production-level performance

---

**Final Accuracy: 97.37%** 

---

*Built with 💙, NumPy, and a lot of gradient descent*
