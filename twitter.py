import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# load dataset
df = pd.read_csv("Twitter Sentiments.csv")
print("dataset loaded successfully")
print(df.head())

#Roberta model
print("loading roberta model...")
roberta = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
print("model loaded successfully")

labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

# process tweets

def preprocess_tweet(tweet):
    tweet_words = []
    for word in tweet.split(" "):
        if word.startswith("@") and len(word) > 1:
            word = "user"
        if word.startswith("http"):
            word = "url"
        tweet_words.append(word)
    return " ".join(tweet_words)

processed_tweets = [preprocess_tweet(tweet) for tweet in df["tweet"]]

# predict sentiment

def predict_sentiment(tweet):
    inputs = tokenizer(tweet, return_tensors="pt")
    outputs = roberta(**inputs)
    logits = outputs.logits
    probabilities = softmax(logits.detach().numpy(), axis=1)
    sentiment = labels[torch.argmax(torch.tensor(probabilities)).item()]
    return sentiment

predicted_sentiments = [predict_sentiment(tweet) for tweet in processed_tweets]

df["Sentiment"] = predicted_sentiments

print(df.head())

df.to_csv("Twitter_Sentiments_Predictions.csv", index=False)
print("predictions saved to Twitter_Sentiments_Predictions.csv successfully")



