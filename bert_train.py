import os
import sys
import logging
import datasets

import pandas as pd
import numpy as np

from transformers import BertTokenizerFast, BertForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

train = pd.read_csv("./Training_data/traindata.csv")

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train, val = train_test_split(train, test_size=.2)

    train_dict = {'label': train['Label'], 'text1': train['Statement'],'text2':train['Intervention']}
    val_dict = {'label': train['Label'], 'text1': train['Statement'],'text2':train['Intervention']}
    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


    def preprocess_function(examples):
        return tokenizer(examples['text1'],examples['text2'], truncation=True,padding=True)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    metric = datasets.load_metric("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(
        output_dir='./checkpoint',  # output directory
        num_train_epochs=10,  # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    prediction_outputs = trainer.predict(tokenized_val)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    # result_output = pd.DataFrame(data={"statement": train["Statement"], "label": test_pred})
    # result_output.to_csv("./result/bert_trainer_data.csv")
    # logging.info('result saved!')
