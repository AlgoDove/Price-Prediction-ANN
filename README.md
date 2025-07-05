📱 Mobile Price Prediction Using ANN

This project is a simple implementation of an Artificial Neural Network (ANN) to classify mobile phones as high or low price range based on their specifications. The model is built using TensorFlow and trained on a real-world dataset.

Problem Statement

In a highly competitive smartphone market, it’s essential to predict a mobile phone’s price range based on its features. This model helps classify phones into two categories:
1 → High price
0 → Low price

Technologies Used

Python
Pandas
NumPy
Scikit-learn
TensorFlow / Keras
Google Colab

Dataset

File name: Mobile_Price_Classification.csv
Target column: price_range (0 or 1)
Features include: RAM, internal memory, battery power, etc.

Steps Performed

Loaded and explored the dataset

Preprocessed data (feature and label separation)

Scaled features using StandardScaler

Split the dataset into training (75 percent) and testing (25 percent)

Built an ANN model with:

Input Layer: 8 neurons

Hidden Layer: 4 neurons

Output Layer: 1 neuron with sigmoid activation

Compiled and trained the model using:

Loss function: binary_crossentropy

Optimizer: adam

Metric: accuracy

Trained for 100 epochs with batch size of 32

Evaluated model performance on the test set

Saved model weights for future use

Results

Achieved a good level of accuracy for binary classification (exact accuracy depends on dataset version)
Demonstrated effective use of ANN for tabular data classification

How to Run

Upload Mobile_Price_Classification.csv in your notebook environment

Run the notebook Mobile_Price_ANN.ipynb in Google Colab

Ensure required libraries are installed (TensorFlow, Pandas, etc.)

Weights will be saved as mobile_price_model_weights.h5

Learning Outcome

This project helped strengthen my understanding of:
Data preprocessing for neural networks
ANN architecture and hyperparameter tuning
Model training, evaluation, and weight saving in TensorFlow

License / Credits

Dataset inspired by mobile price classification challenges commonly found on platforms like Kaggle.
