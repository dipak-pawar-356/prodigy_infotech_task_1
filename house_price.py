# Importing necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generating some sample data (replace this with your dataset)
# Assuming X represents features (square footage, number of bedrooms, number of bathrooms) and y represents the target (house prices)
np.random.seed(0)
num_samples = 1000
X = np.random.rand(num_samples, 3)  # Assuming three features
y = 1000 * X[:, 0] + 300 * X[:, 1] + 500 * X[:, 2] + np.random.randn(num_samples) * 10000  # Random linear relationship with noise

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the testing set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Printing the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Generating house prices based on the test set features
random_house_idx = np.random.randint(0, len(X_test))
random_house_features = X_test[random_house_idx]
random_house_price = model.predict([random_house_features])
print("\nRandom House Features:", random_house_features)
print("Predicted House Price:", random_house_price)
