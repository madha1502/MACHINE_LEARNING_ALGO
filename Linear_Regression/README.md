# Linear Regression

> **Day:** Day-01  
> **Category:** Supervised Learning  
> **Date:** 2026-07-21  

## 📝 Concept & Notes

# Linear Regression from Scratch using Gradient Descent

## Objective

The goal of Linear Regression is to find the best-fit straight line that predicts the output (`Y`) from the input (`X`).

The mathematical equation is:

[
y = mx + b
]

Where:

* **m** = Slope (how much `y` changes when `x` changes)
* **b** = Intercept (value of `y` when `x = 0`)
* **x** = Input feature
* **y** = Predicted output

---

# Dataset

```python
X = [800, 1000, 1200, 1500, 1800]
Y = [150, 180, 210, 250, 300]
```

* `X` represents the input values (house sizes).
* `Y` represents the target values (house prices).

---

# Initial Parameters

```python
m = 0.1
b = 50
learning_rate = 0.0000001
epochs = 100
```

### Slope (`m`)

* Initial guess for the slope.
* Updated during training.

### Intercept (`b`)

* Initial guess for the intercept.
* Updated during training.

### Learning Rate

* Controls how large each update step is.
* Too small → slow learning.
* Too large → may overshoot the optimal solution.

### Epochs

* Number of times the entire dataset is used for training.

---

# Training Process

The algorithm repeats the following steps for every epoch:

1. Predict the output.
2. Calculate the error.
3. Compute the loss.
4. Calculate gradients.
5. Update the parameters.
6. Repeat until training finishes.

---

# Step 1: Prediction

```python
y_pred = m * x + b
```

Uses the current equation to predict the output.

Example:

```
m = 0.1
b = 50
x = 800

Prediction = 130
```

---

# Step 2: Error

```python
error = y - y_pred
```

Measures how far the prediction is from the actual value.

* Positive error → Prediction is too low.
* Negative error → Prediction is too high.

---

# Step 3: Loss

```python
loss = error ** 2
```

Uses Squared Error.

Reasons:

* Removes negative values.
* Penalizes large errors more heavily.
* Makes optimization easier.

---

# Step 4: Total Loss

```python
total_loss += loss
```

Adds the loss of every sample.

After all samples:

```python
cost = total_loss / n
```

This gives the **Mean Squared Error (MSE)**.

---

# Step 5: Compute Gradients

```python
gradient_m += x * error
gradient_b += error
```

These values determine how much the slope and intercept should change.

Average gradients:

```python
gradient_m = (-2 / n) * gradient_m
gradient_b = (-2 / n) * gradient_b
```

---

# Step 6: Update Parameters

```python
m = m - learning_rate * gradient_m
b = b - learning_rate * gradient_b
```

Gradient Descent moves the parameters in the direction that reduces the cost.

---

# Step 7: Repeat

The above steps are repeated for every epoch.

Example:

```
Epoch 1
↓

Epoch 2
↓

Epoch 3
↓

...

Epoch 100
```

The cost should decrease gradually as the model learns.

---

# Final Model

After training, the final equation becomes:

```
y = mx + b
```

where `m` and `b` are the learned values.

---

# Why Do We Initialize `m` and `b`?

Initially, we do **not know** the correct equation.

Therefore, we start with random or chosen guesses.

Example:

```python
m = 0.1
b = 50
```

Gradient Descent improves these values during training.

---

# Why Does the Learning Rate Stay Fixed?

The learning rate is a **hyperparameter** chosen before training.

It controls the update size.

Typical behavior:

* Small learning rate → Slow but stable learning.
* Large learning rate → Faster updates but may not converge.

Unlike `m` and `b`, it usually remains fixed during training.

---

# What Happens After the Model Converges?

Suppose the model predicts almost perfectly at Epoch 50.

If:

```python
epochs = 100
```

training still continues until Epoch 100 because the loop was instructed to run 100 times.

However:

* Error becomes nearly zero.
* Gradients become nearly zero.
* Updates become extremely small.
* `m` and `b` change very little.

---

# Early Stopping

Instead of always running for a fixed number of epochs, many machine learning libraries stop training automatically when the model has converged.

Example:

```python
if cost < threshold:
    break
```

or

```python
if no_improvement_for_10_epochs:
    stop_training()
```

This saves computation time and prevents unnecessary training.

---

# From Scratch vs Scikit-learn

## From Scratch

Advantages:

* Understands every mathematical step.
* Learns Gradient Descent.
* Understands cost and gradients.
* Great for beginners.

Used mainly for:

* Learning
* Interviews
* Academic understanding

---

## Scikit-learn

Advantages:

* Optimized implementation.
* Faster.
* Reliable.
* Used in real-world projects.

Example:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, Y)
```

Used mainly for:

* Industry projects
* Kaggle competitions
* Production systems

---

# Key Concepts Learned

* Linear Regression
* Slope (`m`)
* Intercept (`b`)
* Prediction
* Error
* Squared Error Loss
* Mean Squared Error (MSE)
* Gradient
* Gradient Descent
* Learning Rate
* Epoch
* Model Training
* Model Convergence
* Early Stopping
* Hyperparameters vs Model Parameters

---

# Learning Summary

* We begin with guessed values of `m` and `b`.
* The model predicts outputs using `y = mx + b`.
* The prediction error is calculated.
* Squared Error Loss measures prediction quality.
* Gradients determine how to update the parameters.
* Gradient Descent updates `m` and `b`.
* The process repeats over multiple epochs.
* The cost decreases as the model improves.
* After convergence, parameter updates become very small.
* In practice, libraries like **scikit-learn** automate these steps, but understanding the implementation from scratch builds a strong foundation for Machine Learning.

## 💻 Implementation

```python
# ===========================
# STEP 1 : Dataset
# ===========================

X = [800, 1000, 1200, 1500, 1800]
Y = [150, 180, 210, 250, 300]


# ===========================
# STEP 2 : Initialize Parameters
# ===========================

m = 0.1
b = 50

learning_rate = 0.0000001
epochs = 100

n = len(X)


# ===========================
# STEP 3 : Training Loop
# ===========================

for epoch in range(epochs):

    total_loss = 0
    gradient_m = 0
    gradient_b = 0

    # -----------------------
    # STEP 4 : Visit every sample
    # -----------------------

    for x, y in zip(X, Y):

        # Prediction
        y_pred = m * x + b

        # Error
        error = y - y_pred

        # Loss
        loss = error ** 2

        # Add loss
        total_loss += loss

        # Gradient for m
        gradient_m += x * error

        # Gradient for b
        gradient_b += error


    # -----------------------
    # STEP 5 : Cost
    # -----------------------

    cost = total_loss / n


    # -----------------------
    # STEP 6 : Average Gradient
    # -----------------------

    gradient_m = (-2 / n) * gradient_m
    gradient_b = (-2 / n) * gradient_b


    # -----------------------
    # STEP 7 : Update Parameters
    # -----------------------

    m = m - learning_rate * gradient_m
    b = b - learning_rate * gradient_b


    # -----------------------
    # STEP 8 : Print Progress
    # -----------------------

    print(
        f"Epoch {epoch+1:3d} | "
        f"Cost={cost:.2f} | "
        f"m={m:.5f} | "
        f"b={b:.5f}"
    )


# ===========================
# STEP 9 : Final Model
# ===========================

print("\nTraining Completed")
print(f"Final Equation : y = {m:.5f}x + {b:.5f}")
```

---
*Logged via [ML GitHub NoteSync Extension](https://github.com)*
