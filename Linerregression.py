import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def gradient_descent(x, y, m, c, learning_rate):
    n = len(y)
    y_pred = m * x + c
    # Calculate gradients
    dm = (-2/n) * np.sum(x * (y - y_pred))
    dc = (-2/n) * np.sum(y - y_pred)
    # Update parameters
    m -= learning_rate * dm
    c -= learning_rate * dc
    return m, c

# Load your dataset
data = pd.read_csv("NNpython.csv")


x = data['office_size'].values
y = data['office_price'].values

# Placeholder for example (replace with actual data)
x = np.array([150, 200, 250, 300, 350])  # example feature values (office size)
y = np.array([2000, 2500, 3000, 3500, 4000])  # example target values (office price)

# Hyperparameters
learning_rate = 0.0001
epochs = 10
m, c = np.random.rand(), np.random.rand()

for epoch in range(epochs):
    y_pred = m * x + c
    mse = mean_squared_error(y, y_pred)
    m, c = gradient_descent(x, y, m, c, learning_rate)
    print(f"Epoch {epoch + 1}, MSE: {mse}")

# Plot the line of best fit after final epoch
plt.scatter(x, y, color="blue", label="Data points")
plt.plot(x, m * x + c, color="red", label="Line of best fit")
plt.xlabel("Office Size (sq. ft)")
plt.ylabel("Office Price")
plt.legend()
plt.show()

office_size = 100
predicted_price = m * office_size + c
print(f"Predicted price for 100 sq. ft.: {predicted_price}")
