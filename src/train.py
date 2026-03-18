#1.import libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#2.load dataset
data = fetch_california_housing()
X = data.data
y = data.target

print("Dataset loaded successfully")
print("Shape of X:", X.shape)

#3.train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split completed")

#4.Scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("Feature scaling done")

#5.model training
model = LinearRegression()
model.fit(X_train_s, y_train)

print("Model training completed")

#6.predictions
y_pred = model.predict(X_test_s)

#7.Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Model Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("R2 score:", r2)