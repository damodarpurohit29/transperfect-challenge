import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
import sacrebleu

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_test_data(config):
    c1_conf = config['challenge_1']
    data_conf = c1_conf['data']

    df = pd.read_excel(data_conf['domain_test_set_path'])

    source_col = data_conf['columns']['source']
    target_col = data_conf['columns']['target']

    df[source_col] = df[source_col].astype(str).fillna("")
    df[target_col] = df[target_col].astype(str).fillna("")

    df = df[
        (df[source_col].str.strip() != "") &
        (df[target_col].str.strip() != "")
        ]

    sources = df[source_col].tolist()
    references = df[target_col].tolist()

    return sources, references, df


def generate_translations(
        model,
        tokenizer,
        sources: List[str],
        batch_size: int = 16,
        max_length: int = 256,
        num_beams: int = 5,
        device: str = "cpu"
) -> List[str]:
    model.eval()
    model.to(device)

    all_translations = []

    for i in tqdm(range(0, len(sources), batch_size), desc="Translating"):
        batch = sources[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_translations.extend(translations)

    return all_translations


def compute_all_metrics(predictions: List[str], references: List[str]) -> Dict:
    """
    Compute BLEU, chrF, and TER metrics.

    Args:
        predictions: List of predicted translations
        references: List of reference translations

    Returns:
        Dictionary with metric scores
    """
    predictions_clean = [pred.strip() for pred in predictions]
    references_clean = [ref.strip() for ref in references]

    metrics = {}

    print("\nComputing BLEU...")
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU()
        bleu_result = bleu.corpus_score(predictions_clean, [references_clean])
        metrics['bleu'] = bleu_result.score
        print(f"✅ BLEU: {bleu_result.score:.2f}")
    except Exception as e:
        print(f"❌ BLEU failed: {e}")
        metrics['bleu'] = 0.0

    print("\nComputing chrF...")
    try:
        from sacrebleu.metrics import CHRF
        chrf = CHRF()
        chrf_result = chrf.corpus_score(predictions_clean, [references_clean])
        metrics['chrf'] = chrf_result.score
        print(f"✅ chrF: {chrf_result.score:.2f}")
    except Exception as e:
        print(f"⚠️ chrF skipped: {e}")
        metrics['chrf'] = None

    print("\nComputing TER...")
    try:
        from sacrebleu.metrics import TER
        ter = TER()
        ter_result = ter.corpus_score(predictions_clean, [references_clean])
        metrics['ter'] = ter_result.score
        print(f"✅ TER: {ter_result.score:.2f}")
    except Exception as e:
        print(f"⚠️ TER skipped: {e}")
        metrics['ter'] = None

    # COMET requires source texts and special setup
    metrics['comet'] = None

    return metrics


def calculate_additional_stats(predictions: List[str], references: List[str]) -> Dict:
    stats = {}

    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]

    stats['avg_pred_length'] = np.mean(pred_lengths)
    stats['avg_ref_length'] = np.mean(ref_lengths)
    stats['length_ratio'] = stats['avg_pred_length'] / stats['avg_ref_length']

    return stats


def save_results(
        config,
        predictions: List[str],
        references: List[str],
        sources: List[str],
        metrics: Dict,
        stats: Dict,
        df_original: pd.DataFrame
):
    results_dir = Path(config['project']['results_dir'])
    results_dir.mkdir(exist_ok=True)

    output_df = pd.DataFrame({
        'Source': sources,
        'Reference': references,
        'Prediction': predictions
    })

    output_file = results_dir / "challenge1_test_results.xlsx"
    output_df.to_excel(output_file, index=False)
    print(f"\n Predictions saved to: {output_file}")

    metrics_file = results_dir / "challenge1_metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Challenge 1 - Translation Evaluation Results\n")
        f.write("=" * 60 + "\n\n")

        f.write("Model: Helsinki-NLP/opus-mt-en-nl (fine-tuned)\n")
        f.write(f"Test samples: {len(predictions)}\n\n")

        f.write("-" * 60 + "\n")
        f.write("METRICS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"BLEU:  {metrics.get('bleu', 0):.2f}\n")
        if metrics.get('chrf') is not None:
            f.write(f"chrF:  {metrics.get('chrf', 0):.2f}\n")
        if metrics.get('ter') is not None:
            f.write(f"TER:   {metrics.get('ter', 0):.2f}\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write("STATISTICS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Avg prediction length: {stats['avg_pred_length']:.1f} words\n")
        f.write(f"Avg reference length:  {stats['avg_ref_length']:.1f} words\n")
        f.write(f"Length ratio:          {stats['length_ratio']:.2f}\n")

    print(f" Metrics saved to: {metrics_file}")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"BLEU:  {metrics.get('bleu', 0):.2f}")
    if metrics.get('chrf') is not None:
        print(f"chrF:  {metrics.get('chrf', 0):.2f}")
    if metrics.get('ter') is not None:
        print(f"TER:   {metrics.get('ter', 0):.2f}")
    print("=" * 60)


def main():
    try:
        print("Loading configuration...")
        config = load_config()

        c1_conf = config['challenge_1']['encoder_decoder']
        eval_conf = c1_conf['evaluation']

        model_path = c1_conf['output_dir']

        print(f"Loading model from: {model_path}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        print("\nLoading test data...")
        sources, references, df_original = load_test_data(config)
        print(f"Loaded {len(sources)} test samples")

        print("\nGenerating translations...")
        predictions = generate_translations(
            model=model,
            tokenizer=tokenizer,
            sources=sources,
            batch_size=eval_conf.get('batch_size', 16),
            max_length=eval_conf.get('max_length', 256),
            num_beams=eval_conf.get('num_beams', 5),
            device=device
        )

        print("\nComputing metrics...")
        metrics = compute_all_metrics(predictions, references)

        print("\nComputing statistics...")
        stats = calculate_additional_stats(predictions, references)

        print("\nSaving results...")
        save_results(
            config,
            predictions,
            references,
            sources,
            metrics,
            stats,
            df_original
        )

        print("\n Evaluation complete!")

    except FileNotFoundError as e:
        print(f" File not found: {e}")
        raise
    except Exception as e:
        print(f" Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
