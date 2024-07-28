"""This file does sequence classification with BERT model training."""
import click
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from src.tf_idf_single_level import load_and_preprocess


class SigmaDs(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            tokenizer: BertTokenizer,
            max_len: int
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.iloc[index]['text']
        tags = self.df.iloc[index]['tags']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(tags, dtype=torch.float)
        }


def train(data, model_dir):
    # Process the tags
    data['tags'] = data['tags'].apply(lambda x: x.split(','))
    mlb = MultiLabelBinarizer()
    data['tags_binarise'] = mlb.fit_transform(data['tags']).tolist()

    # Split the data
    train_df, test_df = train_test_split(data, test_size=0.2, ramdom_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create dataset
    train_dataset = SigmaDs(train_df, tokenizer, max_len=128)
    val_dataset = SigmaDs(val_df, tokenizer, max_len=128)
    test_dataset = SigmaDs(test_df, tokenizer, max_len=128)

    # Initialize model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(mlb.classes_))

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    # Save the final model
    trainer.save_model(model_dir)

    results = trainer.evaluate(test_dataset)
    print(results)


@click.command()
@click.option(
    "--input-filepath",
    type=str,
    default="/Users/ananya.lahiri/PycharmProjects/SigmaProject/Users/ananya.lahiri/output_sigma/selected_fields_driver_load/rules/windows/driver_load/extracted_data.json",
    help="Location of input datafile",
)
@click.option(
    "--model-savedir",
    type=str,
    default='./results/final_model',
    help="Loc to save final model"
)
def main(
        input_filepath,
        model_savedir,
):
    data = load_and_preprocess(input_filepath)

    train(data=data, model_dir=model_savedir)


if __name__ == "__main__":
    main()
