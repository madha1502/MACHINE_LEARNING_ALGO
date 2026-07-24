# Logistic Regression

> **Day:** Day-02  
> **Category:** Supervised Learning  
> **Date:** 2026-07-24  

## 📝 Concept & Notes

# Logistic Regression Complete Study Notes

## 1. Fundamental Concept
* Purpose: A supervised learning algorithm used to predict the probability of a categorical dependent variable.
* Classification Tool: Despite the name "regression," it is used for classification by mapping continuous inputs to discrete outputs.
* Core Function: Uses the Sigmoid (Logistic) Function to squeeze any real-valued number into a range between 0 and 1.
* The Equation: σ(z) = 1 / (1 + e^-z)
* Linear Component: z = β0 + β1x1 + β2x2 + ... + βnxn

## 2. The Math: Odds & Log-Odds
* Odds Ratio: The ratio of the probability of success (p) to the probability of failure (1-p).
  Odds = p / (1 - p)
* Logit Function: The natural logarithm of the odds ratio. It linearises the relationship with the features.
  ln(p / (1 - p)) = β0 + β1x1 + ... + βnxn
* Coefficient Interpretation: For every 1-unit increase in x1, the log-odds of the outcome increase by β1. Alternatively, the odds multiply by e^β1.

## 3. Training & Optimisation
* Loss Function: Binary Cross-Entropy Loss (Log Loss). Regular Mean Squared Error (MSE) is not used because the sigmoid curve makes the MSE loss surface non-convex (leads to local minima).
  Cost = - [y * log(p) + (1 - y) * log(1 - p)]
* Optimisation: Uses Maximum Likelihood Estimation (MLE) via Gradient Descent to find the weights that maximise the likelihood of observing the training data.

## 4. Key Assumptions
* Binary Outcome: The dependent variable must be categorical (binary for standard logistic regression).
* Independence: Data observations must be independent of each other (no repeated measures).
* Linearity of Logit: The independent variables must be linearly related to the log-odds, not the raw probabilities.
* No Multicollinearity: Predictor variables should not be highly correlated with one another.
* Sample Size: Requires a relatively large sample size to achieve stable MLE convergence.

## 5. Variations
* Binary: Target has 2 classes (e.g., Spam vs Not Spam).
* Multinomial: Target has 3+ unordered classes (e.g., Red, Blue, Green). Uses Softmax instead of Sigmoid.
* Ordinal: Target has 3+ ordered classes (e.g., Low, Medium, High).

## 6. Pros & Cons

### Advantages:
* Interpretability: Highly explainable via coefficients and odds ratios.
* Probability Output: Provides a confidence score (probability), not just a hard classification label.
* Efficiency: Low computational overhead; fast to train and predict.
* Regularisation: Easily accepts L1 (Lasso) and L2 (Ridge) penalties to prevent overfitting.

### Disadvantages:
* Linear Boundaries: Can only construct linear decision boundaries unless features are manually transformed.
* Feature Vulnerability: Highly sensitive to multicollinearity and irrelevant features.
* Outlier Sensitive: Extreme outliers can significantly skew the decision boundary.

## 7. Performance Metrics
* Confusion Matrix: Table layout showing True Positives, True Negatives, False Positives, and False Negatives.
* Accuracy: Overall correct predictions; highly misleading on imbalanced datasets.
* Precision: True Positives / (True Positives + False Positives).
* Recall (Sensitivity): True Positives / (True Positives + False Negatives).
* F1-Score: Harmonic mean of Precision and Recall.
* ROC-AUC: Evaluates how well the model distinguishes between classes across all possible thresholds.

## 💻 Implementation

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Create dummy data (Hours Studied vs Pass/Fail)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)         # Outputs 0 or 1
y_prob = model.predict_proba(X_test)   # Outputs probabilities for [Class 0, Class 1]

# 5. Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
```

---
*Logged via [ML GitHub NoteSync Extension](https://github.com)*
