
# This program trains a Random Forest model on the Iris dataset to predict the species of each flower.
# It then prints the model’s accuracy and the predicted species for every test sample.


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()

# Get the features (measurements) and labels (flower types)
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a RandomForestClassifier (a machine learning model)
model = RandomForestClassifier(n_estimators=100)

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy * 100:.2f}%")

# Mapping numerical predictions to flower names
flower_names = iris.target_names  # This gives you an array: ['setosa', 'versicolor', 'virginica']

# Now, print each prediction as the flower name (not the number)
for i in range(len(y_pred)):
    predicted_flower = flower_names[y_pred[i]]  # Map prediction to flower name
    print(f"Test {i+1}: Predicted flower = {predicted_flower}")
