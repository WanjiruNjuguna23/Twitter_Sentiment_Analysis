# Twitter Sentiment Analysis Using RoBERTa

This project performs sentiment analysis on tweets using a pre-trained RoBERTa model. It classifies tweets into three categories: **Negative**, **Neutral**, and **Positive**. The project processes a dataset of tweets, applies preprocessing, predicts sentiment, and saves the results.

## Project Overview

- **Input**: A CSV file containing tweets (`Twitter Sentiments.csv`).
- **Model**: RoBERTa model (`cardiffnlp/twitter-roberta-base-sentiment`) from Hugging Face, fine-tuned for sentiment analysis on Twitter.
- **Output**: The predicted sentiment for each tweet, saved in a new CSV file (`Twitter_Sentiments_Predictions.csv`).

## Features

- Preprocesses tweets by replacing user mentions (`@user`) and URLs (`http`) with placeholders.
- Uses a pre-trained RoBERTa model to predict sentiment for each tweet.
- Saves the results in a new CSV file with predicted sentiments.

## Technologies Used

- **Python 3.6+**
- **Hugging Face Transformers**: For using the pre-trained RoBERTa model.
- **PyTorch**: For handling model inference.
- **pandas**: For loading and manipulating the dataset.
- **scipy**: For applying the softmax function to model outputs.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/WanjiruNjuguna23/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
