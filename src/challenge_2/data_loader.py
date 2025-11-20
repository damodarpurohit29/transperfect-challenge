import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split


def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def calculate_edit_distance(source: str, target: str) -> float:
    from difflib import SequenceMatcher
    return 1.0 - SequenceMatcher(None, source, target).ratio()


def prepare_qe_datasets(config: Dict) -> Tuple:
    c2_conf = config['challenge_2']
    data_conf = c2_conf['data']
    model_name = c2_conf['model']['base']

    tokenizer = get_tokenizer(model_name)

    df = pd.read_excel(data_conf['dataset_path'])

    source_col = data_conf['columns']['source']
    mt_col = data_conf['columns']['mt']
    pe_col = data_conf['columns']['post_edit']

    df[source_col] = df[source_col].astype(str).fillna("")
    df[mt_col] = df[mt_col].astype(str).fillna("")
    df[pe_col] = df[pe_col].astype(str).fillna("")

    df = df[
        (df[source_col].str.strip() != "") &
        (df[mt_col].str.strip() != "") &
        (df[pe_col].str.strip() != "")
        ]

    print(f"Loaded {len(df)} QE samples")

    df['quality_score'] = df.apply(
        lambda row: float(calculate_edit_distance(
            row[mt_col],
            row[pe_col]
        )),
        axis=1
    )

    train_df, test_df = train_test_split(
        df,
        test_size=data_conf.get('test_size', 0.2),
        random_state=config['project']['random_seed'],
        shuffle=True
    )

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    max_length = data_conf.get('max_length', 128)

    def preprocess_function(examples):
        sources = examples[source_col]
        mt_outputs = examples[mt_col]
        quality_scores = examples['quality_score']

        inputs = [
            f"QE Source: {src} {tokenizer.sep_token} MT: {mt}"
            for src, mt in zip(sources, mt_outputs)
        ]

        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding=False
        )

        model_inputs["labels"] = [
            np.float32(score) for score in quality_scores
        ]

        return model_inputs

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    tokenized_test = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=test_dataset.column_names
    )

    print(f" Tokenization complete")
    print(f"Train samples: {len(tokenized_train)}")
    print(f"Test samples: {len(tokenized_test)}")

    return tokenizer, tokenized_train, tokenized_test, test_df


if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    tokenizer, train_ds, test_ds, test_df = prepare_qe_datasets(config)

    print(f"\n Dataset loaded!")
    print(f"Train: {len(train_ds)}")
    print(f"Test: {len(test_ds)}")
    print(f"First example keys: {train_ds[0].keys()}")
    print(f"First label type: {type(train_ds[0]['labels'])}")
    print(f"First label value: {train_ds[0]['labels']:.3f}")
