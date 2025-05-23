import abc
import os
import re

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sqlalchemy import select
from database import Quote, engine, Base, db_connection
import gensim
import gensim.downloader
import numpy as np
import nltk
from nltk.corpus import stopwords
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

from config import DEVICE, SEED


class BaseVectorizer(abc.ABC):
    @abc.abstractmethod
    def vectorize_init(self, texts):
        pass

    @abc.abstractmethod
    def vectorize(self, texts):
        pass


class BagOfWords(BaseVectorizer):
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def vectorize_init(self, texts):
        return self.vectorizer.fit_transform(texts).toarray()

    def vectorize(self, texts):
        return self.vectorizer.transform(texts).toarray()


class TFIDF(BaseVectorizer):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def vectorize_init(self, texts):
        return self.vectorizer.fit_transform(texts).toarray()

    def vectorize(self, texts):
        return self.vectorizer.transform(texts).toarray()


class Word2Vec(BaseVectorizer):
    def __init__(self):
        self.wv = gensim.downloader.load("word2vec-google-news-300")
        nltk.download("stopwords")
        self.stop_words = set(stopwords.words("english"))

    def vectorize(self, texts):
        vectors = []
        for text in texts:
            words = nltk.word_tokenize(text)
            word_vectors = [self.wv[word] for word in words if word in self.wv and word not in self.stop_words]
            vectors.append(np.mean(word_vectors, axis=0))
        return np.array(vectors)

    def vectorize_init(self, texts):
        return self.vectorize(texts)


class BERT(BaseVectorizer):
    def __init__(self, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(DEVICE)

    def vectorize(self, texts, batch_size=64):
        predictions = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_predictions = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            predictions.extend(batch_predictions)
        return np.array(predictions)

    def vectorize_init(self, texts):
        return self.vectorize(texts)


def preprocess_text(text):
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ImdbDataset:
    def __init__(self):
        self.dataset = load_dataset("imdb")
        self.labels = ["neg", "pos"]

    def split_train_test_classifier(self, train_size, test_size):
        train_sample = self.dataset["train"].shuffle(seed=SEED).select(range(train_size))
        test_sample = self.dataset["test"].shuffle(seed=SEED).select(range(test_size))
        train_texts = [preprocess_text(text) for text in train_sample["text"]]
        test_texts = [preprocess_text(text) for text in test_sample["text"]]
        y_train = train_sample["label"]
        y_test = test_sample["label"]
        return train_texts, test_texts, y_train, y_test

    def split_train_test_bert(self, train_size, test_size, tokenize):
        dataset = self.dataset.remove_columns("label")

        def preprocess(example):
            example["text"] = preprocess_text(example["text"])

        dataset = dataset.map(preprocess)
        dataset = dataset.map(tokenize, batched=True)
        train_sample = dataset["train"].shuffle(seed=SEED).select(range(train_size))
        test_sample = dataset["test"].shuffle(seed=SEED).select(range(test_size))
        return train_sample, test_sample


class QuotesSearch:
    @db_connection
    def __init__(self, session):
        self.vectorizer = Word2Vec()
        Base.metadata.create_all(engine)
        quotes = session.scalars(select(Quote)).all()
        if len(quotes) == 0:
            dataset = load_dataset("m-ric/english_historical_quotes", split="train")
            quotes = [Quote(quote=item["quote"], author=item["author"]) for item in dataset]
            session.add_all(quotes)
            session.commit()
        texts = [preprocess_text(quote.quote) for quote in quotes]
        self.corpus_vectors = self.vectorizer.vectorize_init(texts)
        self.quote_ids = [quote.id for quote in quotes]

    @db_connection
    def search(self, input_text, session, num_similar=3):
        if input_text is None or len(input_text) == 0:
            return []
        input_vector = self.vectorizer.vectorize([preprocess_text(input_text)])
        if len(input_vector.shape) != 2:
            return []
        similarities = cosine_similarity(input_vector, self.corpus_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:num_similar]
        similar_texts = [(session.get(Quote, self.quote_ids[i]), similarities[i]) for i in top_indices if
                         similarities[i] > 0.01]
        return similar_texts
