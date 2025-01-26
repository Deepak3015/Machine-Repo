import pandas as pd 
import  numpy as np
import matplotlib as plt



plt.style.use('./deeplearing.mplstyle')


x_train = np.array([1.0,2.0])
y_train = np.array([300.0],500.0)

print(f'x_train ={x_train}')
print(f'y_train = {y_train}')

# Number of training examples
m = x_train.shape[0]
print(f"Number of training examples: {m}")

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')

# Add labels and title
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')

# Show the plot
plt.show()


# Model parameters
w = 200  # Weight
b = 100  # Bias

print(f"w: {w}")
print(f"b: {b}")

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model.
    Args:
        x (ndarray (m,)): Input features (m examples).
        w (scalar): Weight parameter.
        b (scalar): Bias parameter.
    Returns:
        y (ndarray (m,)): Predicted values.
    """
    m = x.shape[0]  # Number of training examples
    y = np.zeros(m)  # Initialize an array to store predictions

    for i in range(m):
        y[i] = w * x[i] + b  # Compute the prediction for each example

    return y


# Compute model predictions
tmp_f_wb = compute_model_output(x_train, w, b)

# Plot the model predictions
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')

# Plot the actual data points
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

# Add labels, title, and legend
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()

# Show the plot
plt.show()


