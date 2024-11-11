
## Overview

This project implements a sentiment analysis classifier using machine learning techniques. The classifier is designed to analyze text data and predict the sentiment expressed in the text, categorizing it as positive, negative, or neutral. This tool can be beneficial for various applications, including social media monitoring, customer feedback analysis, and more.

## Features

  * Classifies text data into sentiment categories.
  * Supports preprocessing techniques to clean and standardize text.
  * Built using popular machine learning libraries.
  * Optimized for easy integration into applications.

## Technologies Used
* Python
* Scikit-learn
* Pandas
* NLTK / SpaCy
* Matplotlib / Seaborn
* Jupyter Notebook
* Streamlit
* Word Cloud

## Table of Contents
 * Installation
 * Usage
 * Model Training and Evaluation
 * Contributing
 * License

## Installation
 To set up this project, follow these steps:

 Clone this repository:

git clone https://github.com/MusinguziJoseph/Sentiment-analysis_Classifier.git
  cd sentiment-analysis-classifier
* Install the required dependencies:
 - Streamlit
 - pickle
 - SVC
pip install -r requirements.txt


Model Training and Evaluation
  * This classifier was trained on Flikpart product datast,  a public dataset available on kaggle, using Suport Vector Machine learning algorithm. You can retrain the model using the train.py script with a custom dataset.

* Evaluation metrics include accuracy, F1-score, precision, and recall, which are logged in the output after training.


## Contributing
  * We welcome contributions! If you’d like to improve this project, please fork the repository, make your changes, and create a pull request.


## License
   * This project is licensed under the MIT License. 
