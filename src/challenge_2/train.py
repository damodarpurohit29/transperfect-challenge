import yaml
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.challenge_2.data_loader import prepare_qe_datasets


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()

    predictions = np.clip(predictions, 0.0, 1.0)

    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)

    pearson_corr, _ = pearsonr(labels, predictions)
    spearman_corr, _ = spearmanr(labels, predictions)

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson_corr,
        "spearman": spearman_corr,
    }


def main():
    try:
        config = load_config()
        project_conf = config['project']
        c2_conf = config['challenge_2']
        model_conf = c2_conf['model']
        training_conf = c2_conf['training']

        set_seed(project_conf['random_seed'])

        print(f"Loading tokenizer and datasets...")
        tokenizer, tokenized_train, tokenized_test, test_df = prepare_qe_datasets(config)

        print(f"Loading model: {model_conf['base']}")

        model_config = AutoConfig.from_pretrained(
            model_conf['base'],
            num_labels=1,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_conf['base'],
            config=model_config,
        )

        training_args = TrainingArguments(
            output_dir=model_conf['output_dir'],
            learning_rate=training_conf['learning_rate'],
            num_train_epochs=training_conf['num_train_epochs'],
            per_device_train_batch_size=training_conf['per_device_train_batch_size'],
            per_device_eval_batch_size=training_conf['per_device_eval_batch_size'],
            weight_decay=training_conf['weight_decay'],
            warmup_steps=training_conf.get('warmup_steps', 0),

            eval_strategy="no",

            save_steps=training_conf.get('save_steps', 500),
            save_strategy=training_conf.get('save_strategy', 'epoch'),
            save_total_limit=training_conf.get('save_total_limit', 1),

            logging_steps=training_conf['logging_steps'],

            fp16=False,
            report_to="none",
            push_to_hub=False,

            load_best_model_at_end=False,
        )

        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        print(f"--- Starting Fine-Tuning ---")
        print(f"Train samples: {len(tokenized_train)}")
        print(f"Test samples: {len(tokenized_test)}")
        print(f"Epochs: {training_conf['num_train_epochs']}")
        print(f"Batch size: {training_conf['per_device_train_batch_size']}")

        trainer.train()

        print("--- Fine-Tuning Complete ---")

        trainer.save_model()
        tokenizer.save_pretrained(model_conf['output_dir'])

        print(f" Model and tokenizer saved to {model_conf['output_dir']}")

        print("\n--- Running Final Evaluation ---")
        final_metrics = trainer.evaluate()

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"MAE (Mean Absolute Error):     {final_metrics.get('eval_mae', 0):.4f}")
        print(f"RMSE (Root Mean Squared Error): {final_metrics.get('eval_rmse', 0):.4f}")
        print(f"Pearson Correlation:            {final_metrics.get('eval_pearson', 0):.4f}")
        print(f"Spearman Correlation:           {final_metrics.get('eval_spearman', 0):.4f}")
        print("=" * 60)

    except Exception as e:
        print(f" Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
