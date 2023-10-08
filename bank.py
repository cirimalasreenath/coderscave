# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset (assuming you have a CSV file with mobile phone data)
# The dataset should have columns like 'RAM', 'Camera', 'Battery', 'Brand', 'Price'
# Replace 'dataset.csv' with the actual file name and path
data = pd.read_csv('dataset.csv')

# Define features (X) and target (y)
X = data[['RAM', 'Camera', 'Battery', 'Brand']]
y = data['Price']

# Convert categorical 'Brand' column to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['Brand'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
rmse = mse**0.5
print(f'Root Mean Squared Error: {rmse}')

# Now you can use the model to predict prices for new mobile phone specifications
# For example, if you have a new phone with 8GB RAM, 16MP Camera, 4000mAh Battery, and Brand 'Samsung':
new_data = {'RAM': [8], 'Camera': [16], 'Battery': [4000], 'Brand_Samsung': [1]}
new_phone = pd.DataFrame(new_data)
predicted_price = model.predict(new_phone)
print(f'Predicted Price: {predicted_price[0]}')