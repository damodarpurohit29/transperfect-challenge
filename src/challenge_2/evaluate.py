"""
Challenge 2 - Quality Estimation Evaluation Script
Evaluates fine-tuned QE model on test set
"""

import yaml
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_edit_distance(mt: str, pe: str) -> float:
    """Calculate edit distance as quality score."""
    from difflib import SequenceMatcher
    return 1.0 - SequenceMatcher(None, mt, pe).ratio()


def load_test_data(config):
    """Load and prepare test data."""
    c2_conf = config['challenge_2']
    data_conf = c2_conf['data']

    # Load dataset
    df = pd.read_excel(data_conf['dataset_path'])

    source_col = data_conf['columns']['source']
    mt_col = data_conf['columns']['mt']
    pe_col = data_conf['columns']['post_edit']

    # Clean data
    df[source_col] = df[source_col].astype(str).fillna("")
    df[mt_col] = df[mt_col].astype(str).fillna("")
    df[pe_col] = df[pe_col].astype(str).fillna("")

    df = df[
        (df[source_col].str.strip() != "") &
        (df[mt_col].str.strip() != "") &
        (df[pe_col].str.strip() != "")
        ]

    # Calculate true quality scores
    df['true_quality'] = df.apply(
        lambda row: calculate_edit_distance(
            row[mt_col],
            row[pe_col]
        ),
        axis=1
    )

    # Split same way as training
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(
        df,
        test_size=data_conf.get('test_size', 0.2),
        random_state=config['project']['random_seed'],
        shuffle=True
    )

    print(f"Loaded {len(test_df)} test samples")

    return test_df, source_col, mt_col, pe_col


def predict_quality(model, tokenizer, sources, mt_outputs, batch_size=8, device="cpu"):
    """Generate quality predictions."""
    model.eval()
    model.to(device)

    all_predictions = []

    print(f"Predicting quality for {len(sources)} samples...")

    for i in range(0, len(sources), batch_size):
        batch_sources = sources[i:i + batch_size]
        batch_mts = mt_outputs[i:i + batch_size]

        # Format inputs same as training
        inputs = [
            f"QE Source: {src} {tokenizer.sep_token} MT: {mt}"
            for src, mt in zip(batch_sources, batch_mts)
        ]

        encoded = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            predictions = outputs.logits.squeeze().cpu().numpy()

        # Handle single sample case
        if predictions.ndim == 0:
            predictions = np.array([predictions])

        # Clip to valid range [0, 1]
        predictions = np.clip(predictions, 0.0, 1.0)
        all_predictions.extend(predictions)

    return np.array(all_predictions)


def compute_metrics(predictions, true_labels):
    """Compute all evaluation metrics."""

    # Regression metrics
    mae = mean_absolute_error(true_labels, predictions)
    mse = mean_squared_error(true_labels, predictions)
    rmse = np.sqrt(mse)

    # Correlation metrics
    pearson_corr, pearson_pval = pearsonr(true_labels, predictions)
    spearman_corr, spearman_pval = spearmanr(true_labels, predictions)

    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'pearson': pearson_corr,
        'pearson_pval': pearson_pval,
        'spearman': spearman_corr,
        'spearman_pval': spearman_pval,
    }

    return metrics


def create_visualizations(predictions, true_labels, results_dir):
    """Create evaluation plots."""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Scatter plot - Predictions vs True
    ax1 = axes[0]
    ax1.scatter(true_labels, predictions, alpha=0.6, s=80, edgecolors='navy', linewidth=1.5)
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
    ax1.set_xlabel('True Quality Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Quality Score', fontsize=12, fontweight='bold')
    ax1.set_title('Predictions vs True Labels', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)

    # Plot 2: Error distribution
    ax2 = axes[1]
    errors = predictions - true_labels
    ax2.hist(errors, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax2.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(errors):.3f}')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = results_dir / "challenge2_evaluation_plots.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f" Plots saved to: {plot_file}")
    plt.close()


def save_results(config, test_df, predictions, metrics, source_col, mt_col, pe_col):
    """Save all results to files."""
    results_dir = Path(config['project']['results_dir'])
    results_dir.mkdir(exist_ok=True)

    # Create results DataFrame
    output_df = pd.DataFrame({
        'Source': test_df[source_col].values,
        'MT_Output': test_df[mt_col].values,
        'Post_Edit': test_df[pe_col].values,
        'True_Quality': test_df['true_quality'].values,
        'Predicted_Quality': predictions,
        'Error': predictions - test_df['true_quality'].values,
        'Abs_Error': np.abs(predictions - test_df['true_quality'].values)
    })

    # Sort by error for easy review
    output_df = output_df.sort_values('Abs_Error', ascending=False)

    # Save to Excel
    output_file = results_dir / "challenge2_test_results.xlsx"
    output_df.to_excel(output_file, index=False)
    print(f" Results saved to: {output_file}")

    # Save metrics to text file
    metrics_file = results_dir / "challenge2_metrics.txt"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Challenge 2 - Quality Estimation Evaluation Results\n")
        f.write("=" * 70 + "\n\n")

        f.write("Model: xlm-roberta-base (fine-tuned for QE)\n")
        f.write(f"Test samples: {len(predictions)}\n")
        f.write(f"Task: Predict MT quality (edit distance)\n\n")

        f.write("-" * 70 + "\n")
        f.write("REGRESSION METRICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"MAE (Mean Absolute Error):      {metrics['mae']:.4f}\n")
        f.write(f"MSE (Mean Squared Error):       {metrics['mse']:.4f}\n")
        f.write(f"RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}\n\n")

        f.write("-" * 70 + "\n")
        f.write("CORRELATION METRICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Pearson Correlation:  {metrics['pearson']:.4f} ")
        f.write(f"(p-value: {metrics['pearson_pval']:.4f})\n")
        f.write(f"Spearman Correlation: {metrics['spearman']:.4f} ")
        f.write(f"(p-value: {metrics['spearman_pval']:.4f})\n\n")

        f.write("-" * 70 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Average prediction error: ±{metrics['mae']:.4f}\n")

        if metrics['pearson'] > 0.7:
            f.write(" Strong positive correlation with true quality\n")
        elif metrics['pearson'] > 0.5:
            f.write(" Moderate positive correlation with true quality\n")
        else:
            f.write(" Weak correlation with true quality\n")

        if metrics['pearson_pval'] < 0.05:
            f.write(" Correlation is statistically significant (p < 0.05)\n")
        else:
            f.write("⚠ Correlation is NOT statistically significant\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("QUALITY ASSESSMENT:\n")
        f.write("-" * 70 + "\n")

        if metrics['mae'] < 0.10:
            f.write(" Excellent prediction accuracy (MAE < 0.10)\n")
        elif metrics['mae'] < 0.15:
            f.write(" Good prediction accuracy (MAE < 0.15)\n")
        elif metrics['mae'] < 0.20:
            f.write("⚠ Acceptable prediction accuracy (MAE < 0.20)\n")
        else:
            f.write(" Prediction accuracy needs improvement (MAE ≥ 0.20)\n")

    print(f" Metrics saved to: {metrics_file}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"MAE:               {metrics['mae']:.4f}")
    print(f"RMSE:              {metrics['rmse']:.4f}")
    print(f"Pearson:           {metrics['pearson']:.4f} (p={metrics['pearson_pval']:.4f})")
    print(f"Spearman:          {metrics['spearman']:.4f} (p={metrics['spearman_pval']:.4f})")
    print("=" * 70)


def main():
    try:
        print("=" * 70)
        print("Challenge 2 - Quality Estimation Evaluation")
        print("=" * 70 + "\n")

        print("Loading configuration...")
        config = load_config()

        c2_conf = config['challenge_2']
        model_path = c2_conf['model']['output_dir']

        print(f"Loading model from: {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}\n")

        print("Loading test data...")
        test_df, source_col, mt_col, pe_col = load_test_data(config)

        sources = test_df[source_col].tolist()
        mt_outputs = test_df[mt_col].tolist()
        true_labels = test_df['true_quality'].values

        print(f"\nGenerating quality predictions...")
        predictions = predict_quality(
            model=model,
            tokenizer=tokenizer,
            sources=sources,
            mt_outputs=mt_outputs,
            batch_size=8,
            device=device
        )

        print("\nComputing evaluation metrics...")
        metrics = compute_metrics(predictions, true_labels)

        print("\nCreating visualizations...")
        create_visualizations(
            predictions,
            true_labels,
            config['project']['results_dir']
        )

        print("\nSaving results...")
        save_results(
            config,
            test_df,
            predictions,
            metrics,
            source_col,
            mt_col,
            pe_col
        )

        print("\n" + "=" * 70)
        print(" Evaluation complete!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f" File not found: {e}")
        print("Make sure Challenge 2 training is complete!")
        raise
    except Exception as e:
        print(f" Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
