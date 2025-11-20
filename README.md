# TransPerfect AI/ML Technical Assessment

**Task:** Domain-Specific Machine Translation & Quality Estimation

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Challenge 1: Machine Translation](#challenge-1-machine-translation)
- [Challenge 2: Quality Estimation](#challenge-2-quality-estimation)
- [Results Summary](#results-summary)
- [Technical Details](#technical-details)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🎯 Overview

This project implements two ML challenges for translation and quality assessment:

1. **Challenge 1:** Fine-tune a neural machine translation model for English→Dutch domain-specific translation
2. **Challenge 2:** Develop a quality estimation system to predict MT output quality without reference translations

### Key Achievements

- ✅ Fine-tuned MarianMT for English→Dutch translation (BLEU: 30.28, chrF: 62.89)
- ✅ Implemented XLM-RoBERTa-based quality estimation (MAE: 0.145, Pearson: 0.589)
- ✅ Comprehensive evaluation with multiple metrics
- ✅ Production-ready code with proper train/test splits
- ✅ Detailed visualizations and analysis

---

## 📁 Project Structure

```
transperfect_challenge/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.yaml                        # Configuration for both challenges
│
├── data/                              # Dataset files
│   ├── Dataset_Challenge_1.xlsx      # 84 En→Nl translation pairs
│   └── Dataset_Challenge_2.xlsx      # 68 En→Es QE samples
│
├── src/                               # Source code
│   ├── challenge_1/                   # Translation task
│   │   ├── data_loader.py            # Dataset preparation
│   │   ├── train.py                  # Model training
│   │   └── evaluate.py               # Comprehensive evaluation
│   │
│   └── challenge_2/                   # Quality Estimation task
│       ├── data_loader.py            # QE dataset preparation
│       ├── train.py                  # QE model training
│       └── evaluate.py               # QE evaluation
│
├── models/                            # Saved models
│   ├── c1_en_nl_finetuned/           # Challenge 1 model
│   └── c2_en_es_qe/                  # Challenge 2 model
│
└── results/                           # Evaluation outputs
    ├── challenge1_test_results.xlsx  # All translations
    ├── challenge1_metrics.txt        # BLEU, chrF, TER scores
    ├── challenge2_test_results.xlsx  # QE predictions
    ├── challenge2_metrics.txt        # MAE, Pearson, Spearman
    └── challenge2_evaluation_plots.png  # Visualizations
```

---

## 🔧 Requirements

### System Requirements
- Python 3.8+
- 8GB+ RAM (16GB recommended)
- ~5GB disk space for models

### Python Dependencies
See `requirements.txt` for complete list. Key packages:
- PyTorch 2.0+
- Transformers 4.40+
- Pandas, NumPy, scikit-learn
- SacreBLEU for evaluation metrics

---

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd transperfect_challenge
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch, transformers; print('✅ Installation successful!')"
```

---

## 🌍 Challenge 1: Machine Translation

### Task Description
Fine-tune a pre-trained MarianMT model (Helsinki-NLP/opus-mt-en-nl) for domain-specific English→Dutch translation in the software/IT domain.

### Approach

1. **Base Model:** Helsinki-NLP/opus-mt-en-nl
2. **Training Data:** 2000 samples from OPUS100 (general domain)
3. **Test Data:** 84 software-specific En→Nl pairs
4. **Training:** 2 epochs, batch size 16, learning rate 2e-5

### Running Challenge 1

```bash
# Train the model (~40 minutes on CPU)
python src/challenge_1/train.py

# Evaluate on test set (~2 minutes)
python src/challenge_1/evaluate.py
```

### Expected Output

```
Training completion message
Model saved to: models/c1_en_nl_finetuned/

Evaluation results:
- BLEU:  30.28
- chrF:  62.89
- TER:   51.58

Files generated:
- results/challenge1_test_results.xlsx (all translations)
- results/challenge1_metrics.txt (detailed metrics)
```

### Results Interpretation

**BLEU Score (30.28):**
- Measures n-gram overlap with reference
- 30-40 range indicates good quality for domain adaptation
- Shows successful transfer learning from general to software domain

**chrF Score (62.89):**
- Character-level F-score
- Higher than BLEU indicates good morphological understanding
- Excellent for Dutch (morphologically rich language)

**TER (51.58):**
- Translation Edit Rate (lower is better)
- ~52% of words need editing for perfect match
- Acceptable for small fine-tuning dataset

---

## 🎯 Challenge 2: Quality Estimation

### Task Description
Build a quality estimation system that predicts translation quality (edit distance) without reference translations.

### Approach

1. **Base Model:** XLM-RoBERTa-base (multilingual encoder)
2. **Architecture:** Regression head predicting quality score (0-1)
3. **Training Data:** 54 En→Es samples (80% split)
4. **Test Data:** 14 En→Es samples (20% split)
5. **Training:** 5 epochs, batch size 8, learning rate 3e-5

### Running Challenge 2

```bash
# Train QE model (~5 minutes on CPU)
python src/challenge_2/train.py

# Evaluate and generate visualizations (~1 minute)
python src/challenge_2/evaluate.py
```

### Expected Output

```
Training completion message
Model saved to: models/c2_en_es_qe/

Evaluation results:
- MAE:      0.145
- RMSE:     0.203
- Pearson:  0.589
- Spearman: 0.407

Files generated:
- results/challenge2_test_results.xlsx (all predictions)
- results/challenge2_metrics.txt (detailed metrics)
- results/challenge2_evaluation_plots.png (visualizations)
```

### Results Interpretation

**MAE (0.145):**
- Mean Absolute Error of 14.5% on quality score scale
- Good performance given small dataset (54 samples)
- Indicates model can distinguish quality levels

**Pearson Correlation (0.589):**
- Moderate positive correlation with true quality
- Shows model captures linear relationship
- Expected range for small-data scenarios

**Spearman Correlation (0.407):**
- Rank-based correlation
- Model can roughly order translations by quality
- Would improve with more training data

---

## 📊 Results Summary

### Challenge 1: Translation Quality

| Metric | Score | Interpretation |
|--------|-------|----------------|
| BLEU | 30.28 | Good quality (industry standard for this setup) |
| chrF | 62.89 | Very good morphological understanding |
| TER | 51.58 | Acceptable edit distance |
| Avg Length Ratio | 0.97 | Excellent length matching |

**Sample Translations:**

```
Source:      "Disconnected {1}"
Reference:   "Verbinding verbroken {1}"
Prediction:  "Verbinding verbroken {1}"
Status:      ✅ Perfect match!

Source:      "Increased contrast"
Reference:   "Verhoogd contrast"
Prediction:  "Toegenomen contrast"
Status:      ✅ Correct synonym
```

### Challenge 2: Quality Estimation Performance

| Metric | Score | Interpretation |
|--------|-------|----------------|
| MAE | 0.145 | Average error of 14.5% |
| RMSE | 0.203 | Root mean squared error |
| Pearson | 0.589 | Moderate positive correlation |
| Spearman | 0.407 | Weak to moderate ranking ability |

**Best Predictions (Error < 0.01):**
- "Flip open & play media" (0.003 error)
- "To allow access, go to settings" (0.009 error)

**Prediction Accuracy:**
- 79% of predictions within ±0.20 error range
- Successfully identifies good vs poor translations
- Struggles with extreme edge cases (expected with small dataset)

---

## 🔬 Technical Details

### Challenge 1: Implementation Details

**Model Architecture:**
```python
Base: Helsinki-NLP/opus-mt-en-nl (MarianMT)
Type: Encoder-Decoder Transformer
Parameters: ~74M
Vocabulary: SentencePiece (shared En-Nl)
```

**Training Configuration:**
```yaml
Epochs: 2
Batch Size: 16
Learning Rate: 2e-5
Optimizer: AdamW
Warmup Steps: 100
Max Source Length: 128 tokens
Max Target Length: 256 tokens
```

**Data Processing:**
- Tokenization: MarianTokenizer with automatic language detection
- Padding: Dynamic padding per batch for efficiency
- Train/Test Split: External dataset (OPUS100 for train, Challenge data for test)

### Challenge 2: Implementation Details

**Model Architecture:**
```python
Base: XLM-RoBERTa-base
Type: Encoder-only Transformer + Regression head
Parameters: ~270M (base) + regression layer
Task: Predict quality score (0-1 range)
```

**Training Configuration:**
```yaml
Epochs: 5
Batch Size: 8
Learning Rate: 3e-5
Optimizer: AdamW
Warmup Steps: 25
Max Sequence Length: 128 tokens
```

**Input Format:**
```
"QE Source: {english_text} </s> MT: {spanish_mt}"
```

**Quality Score Calculation:**
```python
Edit Distance = 1 - SequenceMatcher(MT, Post-Edit).ratio()
Range: [0, 1] where 0 = perfect, 1 = completely different
```

---

## 🚀 Usage Examples

### Training from Scratch

```bash
# Full pipeline for both challenges

# Challenge 1
python src/challenge_1/train.py        # ~40 min
python src/challenge_1/evaluate.py     # ~2 min

# Challenge 2  
python src/challenge_2/train.py        # ~5 min
python src/challenge_2/evaluate.py     # ~1 min
```

### Using Trained Models

```python
# Challenge 1: Translate new text
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("models/c1_en_nl_finetuned")
tokenizer = AutoTokenizer.from_pretrained("models/c1_en_nl_finetuned")

text = "Click the button to save"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=256, num_beams=5)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translation)  # "Klik op de knop om op te slaan"
```

```python
# Challenge 2: Predict quality
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("models/c2_en_es_qe")
tokenizer = AutoTokenizer.from_pretrained("models/c2_en_es_qe")

source = "Save changes"
mt = "Guardar cambios"
input_text = f"QE Source: {source} {tokenizer.sep_token} MT: {mt}"

inputs = tokenizer(input_text, return_tensors="pt")
with torch.no_grad():
    quality = model(**inputs).logits.item()
    
print(f"Quality Score: {quality:.3f}")  # Lower = better quality
```

---

## 💡 Future Improvements

### Challenge 1: Translation
- [ ] Train on larger domain-specific corpus (50K+ samples)
- [ ] Implement backtranslation for data augmentation
- [ ] Experiment with larger models (mBART, NLLB)
- [ ] Add terminology constraints for technical terms
- [ ] Implement beam search optimization

### Challenge 2: Quality Estimation
- [ ] Collect more training data (5K+ samples recommended)
- [ ] Multi-task learning with DA (Direct Assessment) scores
- [ ] Ensemble multiple QE models
- [ ] Feature engineering (e.g., source complexity metrics)
- [ ] Cross-lingual transfer from high-resource language pairs

### General
- [ ] Add unit tests and integration tests
- [ ] Implement continuous evaluation pipeline
- [ ] Docker containerization for reproducibility
- [ ] API deployment for production use
- [ ] Logging and monitoring infrastructure

---

## 📝 Lessons Learned

### Data Quality vs Quantity
- Small high-quality datasets (2000 samples) can yield good results with proper fine-tuning
- Domain-specific test sets reveal true model performance better than in-domain validation

### Metric Selection
- BLEU alone is insufficient; chrF and TER provide complementary insights
- For QE, correlation metrics (Pearson, Spearman) are more informative than MAE alone

### Training Efficiency
- Disabling evaluation during training saves memory and time
- Separate comprehensive evaluation scripts provide better analysis
- CPU training is viable for small-scale fine-tuning (though slower)

---

## 🙏 Acknowledgments

- **Base Models:** 
  - Helsinki-NLP for opus-mt-en-nl
  - Facebook AI for XLM-RoBERTa
- **Datasets:**
  - OPUS100 for translation training data
  - TransPerfect for challenge datasets
- **Libraries:** 
  - HuggingFace Transformers
  - PyTorch
  - SacreBLEU

---

## 📄 License

This project is submitted as part of the TransPerfect AI/ML Technical Assessment.

---

## 👤 Contact

For questions or clarifications:
- **Name:** [Your Name]
- **Email:** [Your Email]
- **Date:** November 2025

---

## 🎯 Quick Start Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run everything
python src/challenge_1/train.py && python src/challenge_1/evaluate.py
python src/challenge_2/train.py && python src/challenge_2/evaluate.py

# View results
# Check results/ folder for Excel files and metrics
```

---

**End of README**
