# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and load the dataset containing population and profit values.
2. Initialize the parameters (slope and intercept) and define the learning rate and number of iterations.
3. Apply the gradient descent algorithm to minimize the cost function and update the parameters.
4.Predict the profit using the trained linear regression model and display the regression line. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: V G SAIRAIMA
RegisterNumber:  212225040359
*/
import numpy as np

# Input Features
X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]
], dtype=float)

# Target
y = np.array([50, 45, 70, 80, 95], dtype=float)

# Initialize weights and bias
w = np.zeros(X.shape[1])
b = 0

# Hyperparameters
lr = 0.0001
epochs = 1000

# SGD Training
for epoch in range(epochs):

    for i in range(len(X)):

        # Prediction
        y_pred = np.dot(X[i], w) + b

        # Error
        error = y_pred - y[i]

        # Update weights and bias
        w = w - lr * error * X[i]
        b = b - lr * error

# Final weights
print("Weights:", w)
print("Bias:", b)

# Prediction
predictions = np.dot(X, w) + b

print("\nPredictions:")
print(predictions)
```

## Output:

![image](https://github.com/VGSAIRAIMA/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/main/Screenshot%202026-05-21%20192226.png)
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
