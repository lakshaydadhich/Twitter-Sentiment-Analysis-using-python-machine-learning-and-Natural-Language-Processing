# Twitter-Sentiment-Analysis-using-python-machine-learning-and-Natural-Language-Processing



Sentiment Analysis on Twitter Data
This project focuses on performing sentiment analysis on Twitter data using multiple machine learning models. The goal is to predict whether a given tweet is positive or negative. The dataset contains tweets labeled as either positive (1) or negative (0), and the project utilizes Natural Language Processing (NLP) techniques to preprocess and analyze the text data.

Key Features
Data Preprocessing: The text data is cleaned and preprocessed by removing non-alphabetical characters, converting text to lowercase, and applying stemming to reduce words to their root form.
Feature Extraction: The text data is transformed into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique, which is commonly used in text classification tasks.
Machine Learning Models: The following models are trained and evaluated for sentiment classification:
Logistic Regression: A linear model for binary classification.
Random Forest: A powerful ensemble model that combines multiple decision trees.
Multinomial Naive Bayes: A probabilistic model based on Bayes' theorem for classification tasks.
Workflow
Data Collection: The dataset is loaded from a CSV file containing tweets with corresponding sentiment labels (positive or negative).
Text Preprocessing: The text is cleaned by removing punctuation and stopwords and then stemming the words to their root form.
Model Training: The dataset is split into training and testing sets, and the models are trained using the training data.
Evaluation: The performance of the models is evaluated using accuracy, classification report, and confusion matrix.
Usage
The trained models can be used to predict the sentiment of new tweets. The text is preprocessed using the same techniques as the training data before being passed through the model to predict whether the sentiment is positive or negative.

Example input:

text
Copy
Edit
"I love this product! It's amazing."
Example output:

text
Copy
Edit
Predicted Sentiment: Positive
Technologies Used
Python: The programming language used for this project.
NLP Libraries: nltk for text preprocessing tasks such as tokenization, stemming, and stopwords removal.
Machine Learning Libraries: sklearn for implementing machine learning models and evaluation metrics.
