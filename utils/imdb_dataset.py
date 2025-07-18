from typing import Callable

from datasets import load_dataset, Dataset

from config import SEED
from utils.preprocess_text import preprocess_text


class ImdbDataset:
    def __init__(self):
        self._dataset = load_dataset('imdb')
        self.labels = ['neg', 'pos']

    def split_train_test_classifier(self, train_size: int, test_size: int) \
            -> tuple[list[str], list[str], list[int], list[int]]:
        train_sample = self._dataset['train'].shuffle(seed=SEED).select(range(train_size))
        test_sample = self._dataset['test'].shuffle(seed=SEED).select(range(test_size))
        train_texts = [preprocess_text(text) for text in train_sample['text']]
        test_texts = [preprocess_text(text) for text in test_sample['text']]
        y_train = train_sample['label']
        y_test = test_sample['label']
        return train_texts, test_texts, y_train, y_test

    def split_train_test_bert(self, train_size: int, test_size: int, tokenize: Callable) -> tuple[Dataset, Dataset]:
        dataset = self._dataset.remove_columns('label')

        def preprocess(example):
            example['text'] = preprocess_text(example['text'])

        dataset = dataset.map(preprocess)
        dataset = dataset.map(tokenize, batched=True)
        train_sample = dataset['train'].shuffle(seed=SEED).select(range(train_size))
        test_sample = dataset['test'].shuffle(seed=SEED).select(range(test_size))
        return train_sample, test_sample
