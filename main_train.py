from transformers import BertTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BertForMaskedLM

from config import DEVICE, TEST_SIZE, SEED, BERT_TRAIN_SIZE
from utils.imdb_dataset import ImdbDataset


def save_logs(trainer: Trainer) -> None:
    with open('./train_results/log.txt', 'w', encoding='utf-8') as log_file:
        for line in trainer.state.log_history:
            log_file.write(f'{line}\n')


def main() -> None:
    dataset = ImdbDataset()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(DEVICE)

    def tokenize(example):
        return tokenizer(example['text'], padding='max_length', truncation=True).to(DEVICE)

    train_sample, test_sample = dataset.split_train_test_bert(BERT_TRAIN_SIZE, TEST_SIZE, tokenize)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    training_args = TrainingArguments(
        output_dir='./train_results/trainer_output',
        evaluation_strategy='steps',
        eval_steps=1000,
        logging_strategy='steps',
        logging_steps=1000,
        save_strategy='steps',
        save_steps=1000,
        save_total_limit=5,
        learning_rate=2e-5,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        seed=SEED,
        load_best_model_at_end=True,
        report_to='tensorboard'
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_sample,
        eval_dataset=test_sample
    )
    trainer.train()
    save_logs(trainer)
    model.save_pretrained('./train_results/model')
    tokenizer.save_pretrained('./train_results/model')


if __name__ == '__main__':
    main()
