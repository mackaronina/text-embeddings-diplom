import abc
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gensim
import gensim.downloader
import numpy as np
import nltk
from nltk.corpus import stopwords
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import BertTokenizer, BertModel

from config import DEVICE


class BaseVectorizer(abc.ABC):
    @abc.abstractmethod
    def vectorize_init(self, texts):
        pass

    @abc.abstractmethod
    def vectorize(self, texts):
        pass


class BagOfWords(BaseVectorizer):
    def __init__(self):
        self._vectorizer = CountVectorizer()

    def vectorize_init(self, texts):
        return self._vectorizer.fit_transform(texts).toarray()

    def vectorize(self, texts):
        return self._vectorizer.transform(texts).toarray()


class TFIDF(BaseVectorizer):
    def __init__(self):
        self._vectorizer = TfidfVectorizer()

    def vectorize_init(self, texts):
        return self._vectorizer.fit_transform(texts).toarray()

    def vectorize(self, texts):
        return self._vectorizer.transform(texts).toarray()


class Word2Vec(BaseVectorizer):
    def __init__(self):
        self._wv = gensim.downloader.load("word2vec-google-news-300")
        nltk.download("stopwords")
        self._stop_words = set(stopwords.words("english"))

    def vectorize(self, texts):
        vectors = []
        for text in texts:
            words = nltk.word_tokenize(text)
            word_vectors = [self._wv[word] for word in words if word in self._wv and word not in self._stop_words]
            vectors.append(np.mean(word_vectors, axis=0))
        return np.array(vectors)

    def vectorize_init(self, texts):
        return self.vectorize(texts)


class BERT(BaseVectorizer):
    def __init__(self, model_name):
        self._tokenizer = BertTokenizer.from_pretrained(model_name)
        self._model = BertModel.from_pretrained(model_name).to(DEVICE)

    def vectorize(self, texts, batch_size=64):
        predictions = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self._tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            with torch.no_grad():
                outputs = self._model(**inputs)
                batch_predictions = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            predictions.extend(batch_predictions)
        return np.array(predictions)

    def vectorize_init(self, texts):
        return self.vectorize(texts)
