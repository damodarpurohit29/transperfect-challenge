import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_datasets(config):
    c1_conf = config['challenge_1']
    data_conf = c1_conf['data']
    model_name = c1_conf['encoder_decoder']['base_model']
    training_args = c1_conf['encoder_decoder']['training']

    tokenizer = get_tokenizer(model_name)

    train_ds_raw = load_dataset(
        data_conf['training_dataset_name'],
        data_conf['training_dataset_lang_pair'],
        split='train'
    )

    eval_ds_raw = load_dataset(
        data_conf['training_dataset_name'],
        data_conf['training_dataset_lang_pair'],
        split='validation'
    )

    lang_keys = train_ds_raw.features['translation'].languages
    source_lang, target_lang = lang_keys[0], lang_keys[1]

    def flatten_and_rename(batch):
        return {
            "source": [x[source_lang] for x in batch['translation']],
            "target": [x[target_lang] for x in batch['translation']]
        }

    train_ds_flat = train_ds_raw.map(flatten_and_rename, batched=True)
    eval_ds_flat = eval_ds_raw.map(flatten_and_rename, batched=True)

    train_dataset = train_ds_flat.shuffle(seed=config['project']['random_seed']).select(
        range(data_conf['training_sample_size']))
    eval_dataset = eval_ds_flat.shuffle(seed=config['project']['random_seed']).select(range(500))

    df_test = pd.read_excel(data_conf['domain_test_set_path'])
    source_col_name = data_conf['columns']['source']
    target_col_name = data_conf['columns']['target']
    df_test[source_col_name] = df_test[source_col_name].astype(str).fillna('')
    df_test[target_col_name] = df_test[target_col_name].astype(str).fillna('')
    test_dataset = Dataset.from_pandas(df_test)

    def preprocess_function(examples, source_key, target_key):
        inputs = examples[source_key]
        targets = examples[target_key]

        model_inputs = tokenizer(
            inputs,
            max_length=training_args['max_input_length'],
            truncation=True,
            padding="max_length"
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=training_args['max_target_length'],
                truncation=True,
                padding="max_length"
            )

        labels_input_ids = labels["input_ids"]

        processed_labels = []
        for label_row in labels_input_ids:
            row = [token_id if token_id != tokenizer.pad_token_id else -100 for token_id in label_row]
            processed_labels.append(row)

        model_inputs["labels"] = processed_labels
        return model_inputs

    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, "source", "target"),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    tokenized_eval = eval_dataset.map(
        lambda x: preprocess_function(x, "source", "target"),
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    tokenized_test = test_dataset.map(
        lambda x: preprocess_function(x, source_col_name, target_col_name),
        batched=True,
        remove_columns=test_dataset.column_names
    )

    return tokenizer, tokenized_train, tokenized_eval, tokenized_test