import pandas
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

folder_path = "./news"
file_addresses = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return " ".join(filtered_text)


def sentiment_score(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

dates, news, sentiment_scores = [], [], []
for filename in file_addresses:
    with open(filename, 'r') as file:
        date, line = 1, 1
        while date and line:
            date, line = file.readline(), file.readline()
            dates.append(date.strip())
            line = remove_stopwords(clean_text(line.strip()))
            news.append(line)
            sentiment_scores.append(sentiment_score(line))

data = {'Date': dates, 'news': news, 'sentiment':sentiment_scores}
df = pandas.DataFrame(data)

df.to_csv('news.csv', index=False)

folder_path = "./shares"  # Replace this with the path to your folder
file_addresses = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

dfs = []

for filename in file_addresses:
    df = pandas.read_csv(filename)
    df.columns = [col + "_" + filename[-7:-4] if col != 'Date' else col for col in df.columns]
    dfs.append(df)

merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = pandas.merge(merged_df, df, on='Date', how='outer')

merged_df.sort_values(by='Date', inplace=True)
merged_df.reset_index(drop=True, inplace=True)

merged_df.to_csv('shares.csv', index=False)
