import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from config import TRAIN_SIZES, TEST_SIZE
from utils import ImdbDataset, BagOfWords, TFIDF, Word2Vec, BERT


def train_classifier(x_train, x_test, y_train, y_test, labels):
    model = LogisticRegression(C=5, max_iter=1000)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_test = [labels[i] for i in y_test]
    y_pred = [labels[i] for i in y_pred]
    return (classification_report(y_test, y_pred, digits=4, labels=labels),
            f1_score(y_test, y_pred, average='weighted', labels=labels))


def evaluate_model(model_name, model, dataset):
    f1_scores = []
    times = []
    for train_size in TRAIN_SIZES:
        train_texts, test_texts, y_train, y_test = dataset.split_train_test_classifier(train_size, TEST_SIZE)
        x_train = model.vectorize_init(train_texts)
        start = time.time()
        x_test = model.vectorize(test_texts)
        end = time.time()
        times.append((end - start) / TEST_SIZE)
        report, f1 = train_classifier(x_train, x_test, y_train, y_test, dataset.labels)
        print(f"Report for {model_name} with train size {train_size}\n{report}\n")
        f1_scores.append(f1)
    time_per_text = np.mean(times)
    return f1_scores, time_per_text


def show_plot_results(results_f1):
    plt.figure(figsize=(10, 6))
    for label, (sizes, times) in results_f1.items():
        plt.plot(sizes, times, marker='o', label=label)
    plt.title("Comparison of text vectorization algorithms")
    plt.xlabel("Training set size")
    plt.ylabel("F1-score")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    results_f1 = {}
    results_times = {}
    dataset = ImdbDataset()
    models = {
        "Bag of Words": BagOfWords(),
        "TF-IDF": TFIDF(),
        "Word2Vec": Word2Vec(),
        "BERT": BERT('bert-base-uncased'),
        "fine-tuned BERT": BERT('./train_results/model'),
    }
    for model_name, model in models.items():
        f1_scores, time_per_text = evaluate_model(model_name, model, dataset)
        results_f1[model_name] = (TRAIN_SIZES, f1_scores)
        results_times[model_name] = time_per_text
    show_plot_results(results_f1)
    print("\n\nAverage calculation time for each text")
    for model_name, time_per_text in results_times.items():
        print(f"{model_name}: {time_per_text:.5f} seconds")


if __name__ == '__main__':
    main()
