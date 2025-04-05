# iris-flower-classifier
This program uses a Random Forest model to classify the species of flowers in the Iris dataset. The dataset includes features such as sepal and petal length/width to predict the species of Iris flowers.

Steps:
Load the Iris Dataset: The program uses load_iris from sklearn.datasets to load the dataset.

Split Data: It splits the data into training and testing sets (70% for training, 30% for testing).

Train the Model: A Random Forest Classifier (RandomForestClassifier) is used to train the model.

Evaluate Accuracy: The model's accuracy on the test data is calculated and printed.

Predictions: The program outputs the predicted species for each test sample.

Requirements:
scikit-learn library

Usage:
Run the script, and it will train the model, evaluate its performance, and display predictions for each test sample.
