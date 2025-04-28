# 🧠 BERT implementation from from Scratch & Training

This project is a reimplementation of the **BERT** (Bidirectional Encoder Representations from Transformers) architecture based on the encoder block from the original Transformer model described in [**"paper"**](https://arxiv.org/abs/1810.04805).

The goal is to create a minimal but modern version of BERT, pre-trained on the **IMDB Movie Review Dataset** for sentiment analysis.

---

## 📦 Dataset: IMDB Movie Review Dataset

To download the dataset, use `kagglehub`:

```python
import kagglehub

# Login using your Kaggle API credentials
kagglehub.login()

# Download the IMDB dataset
kagglehub.dataset_download("yasserh/imdb-movie-ratings-sentiment-analysis")
```

Once downloaded, place the contents in the `data/` directory.

---

## 🏋️ Training Setup

- **Model**: BERT Transformer (Encoder-only architecture)
- **Dropout**: Not used
- **Dataset**: IMDB Movie Review Dataset
- **Epochs**: 3
- **Framework**: PyTorch
- **Tokenizer**: Pretrained `bert-base`

Despite using a simplified model and limited tuning, the model was able to converge and produce meaningful learning dynamics.

---

## 🚀 Quickstart

```bash
# Run training
python train.py
```

---

## 📊 Test Set Evaluation

```
=== Test Set Evaluation ===
Accuracy : 0.4996
Precision: 0.4996
Recall   : 1.0000
F1 Score : 0.6663
```

---

## 📈 Training & Validation Curves

### 🟦 Training Curves
- **Training Loss**
![image](https://github.com/user-attachments/assets/64aff096-54f2-4f7f-aa1b-90cf1188e348)

- **Training Accuracy**
![image](https://github.com/user-attachments/assets/138a98b7-07e5-456c-a486-966fe4032991)

