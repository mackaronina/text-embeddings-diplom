from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select

from database import Quote, engine, Base, db_connection
from utils.preprocess_text import preprocess_text
from utils.vectorizers import Word2Vec


class QuotesSearch:
    @db_connection
    def __init__(self, session):
        self._vectorizer = Word2Vec()
        Base.metadata.create_all(engine)
        quotes = session.scalars(select(Quote)).all()
        if len(quotes) == 0:
            dataset = load_dataset("m-ric/english_historical_quotes", split="train")
            quotes = [Quote(quote=item["quote"], author=item["author"]) for item in dataset]
            session.add_all(quotes)
            session.commit()
        texts = [preprocess_text(quote.quote) for quote in quotes]
        self._corpus_vectors = self._vectorizer.vectorize_init(texts)
        self._quote_ids = [quote.id for quote in quotes]

    @db_connection
    def search(self, input_text, session, num_similar=3):
        if input_text is None or len(input_text) == 0:
            return []
        input_vector = self._vectorizer.vectorize([preprocess_text(input_text)])
        if len(input_vector.shape) != 2:
            return []
        similarities = cosine_similarity(input_vector, self._corpus_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:num_similar]
        similar_texts = [(session.get(Quote, self._quote_ids[i]), similarities[i]) for i in top_indices if
                         similarities[i] > 0.01]
        return similar_texts
