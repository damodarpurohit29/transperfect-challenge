import yaml
import numpy as np
import evaluate
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.challenge_1.data_loader import prepare_datasets



def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    try:
        config = load_config()
        project_conf = config['project']
        c1_conf = config['challenge_1']['encoder_decoder']
        training_conf = c1_conf['training']
        eval_conf = c1_conf['evaluation']

        set_seed(project_conf['random_seed'])

        tokenizer, tokenized_train, tokenized_eval, _ = prepare_datasets(config)

        print(f"Loading model: {c1_conf['base_model']}")
        model = AutoModelForSeq2SeqLM.from_pretrained(c1_conf['base_model'])

        training_args = Seq2SeqTrainingArguments(
            output_dir=c1_conf['output_dir'],
            learning_rate=training_conf['learning_rate'],
            num_train_epochs=training_conf['num_train_epochs'],
            per_device_train_batch_size=training_conf['per_device_train_batch_size'],
            per_device_eval_batch_size=2,  # Reduced for memory
            gradient_accumulation_steps=training_conf['gradient_accumulation_steps'],
            optim=training_conf['optim'],
            weight_decay=training_conf['weight_decay'],
            warmup_steps=training_conf['warmup_steps'],

            eval_strategy="no",  # Disable eval to save memory

            save_steps=training_conf['save_steps'],
            save_strategy=training_conf['save_strategy'],
            save_total_limit=1,

            logging_steps=training_conf['logging_steps'],
            fp16=training_conf['fp16'],
            report_to="none",
            push_to_hub=False
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        print(f"--- Starting Fine-Tuning ---")
        print(f"Train samples: {len(tokenized_train)}")
        print(f"Epochs: {training_conf['num_train_epochs']}")
        print(f"Batch size: {training_conf['per_device_train_batch_size']}")
        print(f"Note: Evaluation disabled to save memory")

        trainer.train()

        print("--- Fine-Tuning Complete ---")

        trainer.save_model()
        tokenizer.save_pretrained(c1_conf['output_dir'])

        print(f" Model and tokenizer saved to {c1_conf['output_dir']}")
        print(f" Run evaluate.py to get BLEU score on test set")

    except Exception as e:
        print(f"\Error during training: {e}")
        raise


if __name__ == "__main__":
    main()

# import yaml
# import numpy as np
# import evaluate
# from transformers import (
#     AutoModelForSeq2SeqLM,
#     DataCollatorForSeq2Seq,
#     Seq2SeqTrainingArguments,
#     Seq2SeqTrainer,
#     set_seed,
# )
# import sys
# import os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# from src.challenge_1.data_loader import prepare_datasets
#
#
# def load_config(config_path="config.yaml"):
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)
#
#
# def make_compute_metrics(tokenizer):
#     def compute_metrics(eval_preds):
#         preds, labels = eval_preds
#         if isinstance(preds, tuple):
#             preds = preds[0]
#
#         decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#
#         labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#         decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#
#         decoded_preds = [pred.strip() for pred in decoded_preds]
#         decoded_labels = [[label.strip()] for label in decoded_labels]
#
#         metric = evaluate.load("sacrebleu")
#         result = metric.compute(predictions=decoded_preds, references=decoded_labels)
#
#         prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
#         result["gen_len"] = np.mean(prediction_lens)
#
#         return {"bleu": result["score"]}
#
#     return compute_metrics
#
#
# def main():
#     try:
#         config = load_config()
#         project_conf = config['project']
#         c1_conf = config['challenge_1']['encoder_decoder']
#         training_conf = c1_conf['training']
#         eval_conf = c1_conf['evaluation']
#
#         set_seed(project_conf['random_seed'])
#
#         tokenizer, tokenized_train, tokenized_eval, _ = prepare_datasets(config)
#
#         print(f"Loading model: {c1_conf['base_model']}")
#         model = AutoModelForSeq2SeqLM.from_pretrained(c1_conf['base_model'])
#
#         training_args = Seq2SeqTrainingArguments(
#             output_dir=c1_conf['output_dir'],
#             learning_rate=training_conf['learning_rate'],
#             num_train_epochs=training_conf['num_train_epochs'],
#             per_device_train_batch_size=training_conf['per_device_train_batch_size'],
#             per_device_eval_batch_size=training_conf['per_device_eval_batch_size'],
#             gradient_accumulation_steps=training_conf['gradient_accumulation_steps'],
#             optim=training_conf['optim'],
#             weight_decay=training_conf['weight_decay'],
#             warmup_steps=training_conf['warmup_steps'],
#
#             eval_strategy=training_conf['evaluation_strategy'],  # FIX
#             eval_steps=training_conf['eval_steps'],
#             save_steps=training_conf['save_steps'],
#             save_strategy=training_conf['save_strategy'],
#
#             save_total_limit=training_conf['save_total_limit'],
#             load_best_model_at_end=training_conf['load_best_model_at_end'],
#             metric_for_best_model=training_conf['metric_for_best_model'],
#             greater_is_better=training_conf['greater_is_better'],
#
#             logging_steps=training_conf['logging_steps'],
#             fp16=training_conf['fp16'],
#             report_to="none",
#             push_to_hub=False
#         )
#
#         data_collator = DataCollatorForSeq2Seq(
#             tokenizer=tokenizer,
#             model=model,
#             label_pad_token_id=-100
#         )
#
#         trainer = Seq2SeqTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=tokenized_train,
#             eval_dataset=tokenized_eval,
#             tokenizer=tokenizer,
#             data_collator=data_collator,
#             compute_metrics=make_compute_metrics(tokenizer),
#         )
#
#         print(f"--- Starting Fine-Tuning ---")
#         print(f"Train samples: {len(tokenized_train)}")
#         print(f"Eval samples: {len(tokenized_eval)}")
#         print(f"Epochs: {training_conf['num_train_epochs']}")
#         print(f"Batch size: {training_conf['per_device_train_batch_size']}")
#
#         trainer.train()
#
#         print("--- Fine-Tuning Complete ---")
#
#         trainer.save_model()
#         tokenizer.save_pretrained(c1_conf['output_dir'])
#
#         print(f" Model and tokenizer saved to {c1_conf['output_dir']}")
#
#         final_metrics = trainer.evaluate()
#         bleu_score = final_metrics.get('eval_bleu', final_metrics.get('bleu', 0.0))
#         print(f" Final BLEU: {bleu_score:.2f}")
#
#     except FileNotFoundError as e:
#         print(f"❌ Config or data file not found: {e}")
#         raise
#     except Exception as e:
#         print(f"❌ Error during training: {e}")
#         raise
#
#
# if __name__ == "__main__":
#     main()
