# Linear Regression Implementation (Day-01)
# Category: Supervised Learning

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
