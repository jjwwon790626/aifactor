# -*- coding: utf-8 -*-
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm

def analyze_news_sentiment(file_path):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    df = pd.read_csv(file_path, parse_dates=['date'])
    tqdm.pandas(desc="Processing Sentiment")
    df['sentiment'] = df['headline'].progress_apply(lambda x: nlp(x[:512])[0]['label'])
    grouped = df.groupby('date')['sentiment'].value_counts(normalize=True).unstack()
    grouped.columns = [f"{col}_ratio" for col in grouped.columns]
    grouped['net_score'] = grouped.get('positive_ratio', 0) - grouped.get('negative_ratio', 0)
    grouped.reset_index(inplace=True)
    grouped.to_csv('output/daily_sentiment.csv', index=False)
    return grouped