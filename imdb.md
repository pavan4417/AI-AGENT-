## Project Title
IMDB Movie Reviews Sentiment Analysis

## Project Content
<a href="https://colab.research.google.com/github/ramyachandaluri/Ai-Agent/blob/main/IMDB.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Project Code
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from bs4 import BeautifulSoup          # BeautifulSoup is a useful library for extracting data from HTML and XML documents
from numpy import array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, GlobalMaxPooling1D, Dense, Embedding, LSTM, GRU
import pandas.testing as tm

```
```python

# Load the Dataset
movie_reviews = pd.read_csv("/content/IMDB Dataset.csv")
```
```python

# Check the shape of the data
movie_reviews.shape
```
```python

movie_reviews.head()
```
```python

# Check for null Values
movie_reviews.isnull().sum()

```

## Key Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- Natural Language Toolkit (NLTK)
- Jupyter Notebook
    

## Description

This project performs sentiment analysis on IMDB movie reviews. Using preprocessing techniques and machine learning models,
the project aims to classify reviews as positive or negative. It includes data cleaning, vectorization, model training and evaluation.
    

## Output

The model was trained and evaluated for accuracy. Based on the final evaluation metrics, the classifier was able to distinguish
between positive and negative reviews with significant accuracy (detailed metrics available in the notebook outputs).
    

## Further Research

- Apply deep learning models like LSTM or BERT for better performance.
- Explore deployment options using Flask or FastAPI.
- Improve preprocessing with advanced NLP techniques.
- Use larger or more diverse datasets.
    
