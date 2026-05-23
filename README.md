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

```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/acer/Downloads/50_Startups.csv")
x=data["R&D Spend"].values
y=data["Profit"].values


x_mean=np.mean(x)
x_std=np.std(x)
x=(x-x_mean)/x_std


w=0.0
b=0.0
alpha=0.01
epochs=100
n=len(x)

losses=[]


for i in range(epochs):
    y_hat=w*x+b
    loss=np.mean((y_hat-y)**2)
    losses.append(loss)
    
    dw=(2/n)*np.sum((y_hat-y)*x)
    db=(2/n)*np.sum(y_hat-y)
    
    w-=alpha*dw
    b-=alpha*db


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss(MSE)")
plt.title("Loss vs Iterations")

plt.subplot(1,2,2)
plt.scatter(x,y)
x_sorted=np.argsort(x)
plt.plot(x[x_sorted],(w*x+b)[x_sorted],color="red")
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression Fit")

plt.tight_layout()
plt.show()

print(f"Final weight (w): {w}")
print(f"Final bias (b): {b}")

```

```

## Output:
![Image](https://github.com/VGSAIRAIMA/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/main/Screenshot%202026-05-23%20081331.png)
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
