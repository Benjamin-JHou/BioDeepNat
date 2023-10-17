# 🚀 OptNCMiner Model Explanation 

The OptNCMiner is an advanced neural network model designed for a special kind of similarity and relation mining tasks. Here's a breakdown of its components and features:

## ✨ Main Features 

1.Flexible Network Structure: The model's architecture comprises two main sections - heads and bodies, each with customizable layers and dropout rates. 

2.Combination Modes: The model can combine features in various ways including subtraction, addition, cosine similarity, and concatenation. 

3.Training with Early Stopping: The model can be trained with an early stopping mechanism to prevent overfitting. 

4.Supports GPU Acceleration: Training can be expedited by leveraging GPU capabilities. 

5.Prediction Methods: Multiple prediction functions like simple predict, pairwise predict, and support-based predict. 

6.Model Save & Load: Easily save the trained model's state and load it back when required. 

## 🔧 Key Components 

This is the main deep learning model class. It is an `nn.Module` class, which means it is meant to be used with PyTorch.

#### Initialization (`__init__`) 🏗️:
- Defines a deep learning architecture with multiple heads and bodies.
- Initializes parameters and layers based on given shapes and dropout rates.

#### Network (`network`) 🌐:
- Processes two inputs (`left` and `right`) through the head layers.
- Combines the outputs based on a given mode (subtract, add, cosine similarity, or concatenation).

#### Forward (`forward`) ➡️:
- Computes the forward pass of the model.

#### Fit (`fit`) 🏋️:
- Trains the model using given inputs and targets.
- Utilizes early stopping and AdamW optimizer as default.

#### Predict (`predict`) 🎯:
- Evaluates the model on a given input.

#### Predict2 (`predict2`) 🎯:
- Another prediction function that takes separate left and right inputs.

#### Predict Support (`predict_support`) 🎯:
- Predicts using support network.

## ⚙️ Utility Functions 

#### saveModel (`saveModel`) 💾:
- Saves the current model state, including support vectors and parameters.

#### loadModel (`loadModel`) 📂:
- Loads a saved model from a file.

## Main Execution 🚦

When executed as a standalone script, it sets up logging for the module.


## 📝 Tips and Notes 

💡 Make sure the environment variable 'KMP_DUPLICATE_LIB_OK' is set to 'True' to avoid library issues.

💡 The network is flexible. You can easily change the shapes of heads and bodies to fit the complexity of your data.

💡 By default, the model uses the AdamW optimizer and binary cross-entropy loss. However, these can be customized based on your needs.

💡 Training logs are provided for better tracking and debugging.

## 🚀 Get Started! 
To start using this model:

📚 Import necessary libraries.

⚙️ Define data and parameters.

🎉 Initialize the OptNCMiner model.

🏋️ Train the model using the fit method.

🔮 Use the predict methods to make predictions on new data.

💼 Save trained model using saveModel and load it later using loadModel.

