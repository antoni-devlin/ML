import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Replace 'your_data.csv' with your actual dataset
data = pd.read_csv('london_weather.csv')

# Replace 'target_column' with your target variable's name
X = data.drop('precipitation', axis=1)
y = data['precipitation']

# Impute missing values in the target variable with the mean
y.fillna(y.mean(), inplace=True)

# Handle missing values in features if any
X.fillna(0, inplace=True)  # Replace with your own strategy

# Encode categorical variables if any
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Measure the training time
start_time = time.time()

# Use a neural network model (MLPRegressor) instead of Linear Regression
model = MLPRegressor(hidden_layer_sizes=(100, 100, 50, 20), max_iter=1500, random_state=42)
model.fit(X_train, y_train)

end_time = time.time()
training_time = end_time - start_time

# Visualize convergence
plt.plot(model.loss_curve_)
plt.title('Training Loss over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Training Loss')
plt.show()

print(f'Training time: {training_time} seconds')

y_pred = model.predict(X_test)

# Evaluate using a regression metric (e.g., Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')
