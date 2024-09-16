# IPL-Score-Prediction
This code builds an IPL score predictor using a neural network with two hidden layers. It preprocesses data using label encoding and scaling, trains with Huber loss, and evaluates performance with Mean Absolute Error. An interactive interface allows users to input values like teams and players to dynamically predict scores via `ipywidgets`.

# Import Libraries:

Standard libraries like Pandas, NumPy, Matplotlib, and Seaborn for data manipulation and visualization.
Sklearn preprocessing tools like LabelEncoder for encoding categorical variables and MinMaxScaler for scaling the input features.
Keras and TensorFlow for creating, compiling, and training a neural network model.

# Data Preparation:

The IPL data (ipl_data.csv) is loaded, and irrelevant features like 'date', 'runs', 'mid', etc., are dropped to focus on features directly influencing the score prediction.
The target variable (total) is separated from the features.

# Label Encoding:

Categorical features like venue, batting team, bowling team, batsman, and bowler are converted into numerical labels using LabelEncoder. This is essential because machine learning models need numerical inputs.

# Train-Test Split:

The dataset is split into training and testing sets using an 80-20 split (train_test_split).

# Feature Scaling:

Feature values are normalized to a range of [0, 1] using MinMaxScaler, which helps the neural network converge faster.

# Neural Network Model:

The neural network model has 2 hidden layers with 512 and 216 neurons, using ReLU activation.
The output layer has 1 unit and uses a linear activation function since this is a regression problem (predicting a continuous variable, the score).
The model is compiled with the Huber loss, which is less sensitive to outliers compared to MSE, making it a good choice for regression problems. The optimizer is Adam.

# Training the Model:

The model is trained for 50 epochs using a batch size of 64. During training, it also validates the model on the test data (validation_data).

# Visualization:

The model_losses DataFrame stores the training and validation losses during training, and plot() is used to visualize how the loss changes over epochs.

# Predictions:

After training, predictions are made on the test set, and the Mean Absolute Error (MAE) is calculated to evaluate the performance of the model.

# Interactive Score Prediction:

An interactive interface is created using ipywidgets to allow the user to select input values for venue, batting team, bowling team, striker, and bowler.
Once inputs are selected, a button click triggers the predict_score function that:
Encodes the selected input values.
Prepares and scales the input.
Passes the scaled input through the trained model to predict the score.
Displays the predicted score.

# Key Points to Note:
Label Encoding: The code first converts categorical features into numerical values before training the model. This is necessary as neural networks work with numerical data.

Huber Loss: This loss function combines elements of both mean squared error and mean absolute error, making it more robust to outliers, which is suitable for regression problems like score prediction.

Scaling: Feature scaling is crucial in neural networks for faster and more effective convergence. Here, MinMaxScaler ensures the input features are on the same scale.

Interactive Widgets: Using ipywidgets, a dropdown for user inputs allows for an interactive score prediction tool, which makes the model more accessible for non-technical users.

![image](https://github.com/user-attachments/assets/c6a3f910-3dfb-48e9-9751-140d2ea721e8)

