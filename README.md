# Spam SMS Detection

Hệ thống phát hiện tin nhắn spam/lừa đảo sử dụng Machine Learning và BERT.

## Models

| Model | Approach | Expected F1 (spam) |
|---|---|---|
| Naive Bayes | TF-IDF + MultinomialNB | ~0.94 |
| SVM | TF-IDF + LinearSVC | ~0.97 |
| Logistic Regression | TF-IDF + LR | ~0.96 |
| **BERT** | bert-base-uncased fine-tuned | **~0.98** |

## Dataset

UCI SMS Spam Collection — 5,574 tin nhắn (4,827 ham / 747 spam).

**Tải dataset:**
1. Vào https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Tải file `spam.csv`
3. Đặt vào thư mục `data/spam.csv`

## Sử dụng BERT (không cần train lại)

Model BERT đã được train sẵn và push lên HuggingFace. Load trực tiếp:

```python
from src.bert_model import load_bert, predict_bert

model, tokenizer = load_bert()  # tự động tải từ HuggingFace lần đầu

label, confidence = predict_bert(model, tokenizer, "Congratulations! You won a free iPhone!")
print(label, confidence)  # spam 0.9980
```

> Lần đầu chạy sẽ cache model về `~/.cache/huggingface/`, các lần sau load nhanh hơn.

## API

**Chạy server:**
```bash
uvicorn api:app --reload
```

Server mặc định tại `http://localhost:8000`. Docs tự động tại `http://localhost:8000/docs`.

### Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/health` | Kiểm tra server + models đã load |
| GET | `/models` | Danh sách models có sẵn |
| POST | `/predict` | Phân loại tin nhắn spam/ham |

### POST /predict

**Request:**
```json
{
  "text": "Congratulations! You won a free iPhone!",
  "model": "bert"
}
```

`model` có thể là: `bert` (mặc định), `NaiveBayes`, `SVM`, `LogisticRegression`

**Response:**
```json
{
  "text": "Congratulations! You won a free iPhone!",
  "label": "spam",
  "confidence": 0.998,
  "model": "bert"
}
```

---

```bash
pip install -r requirements.txt
```

> BERT yêu cầu PyTorch. Nếu có GPU:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```

## Chạy

**Toàn bộ pipeline (ML + BERT):**
```bash
python train.py
```

**Chỉ ML models** (không cần GPU, ~2 phút):
```bash
python train.py --skip-bert
```

## Output

Sau khi train xong, toàn bộ kết quả lưu trong `models/`:

```
models/
├── NaiveBayes.pkl          # Saved ML models
├── SVM.pkl
├── LogisticRegression.pkl
├── metrics.json            # Bảng số liệu tất cả models
├── eda.png                 # Class distribution, message length
├── correlation.png         # Feature correlation heatmap
├── roc_curves.png          # ROC curves so sánh tất cả models
├── confusion_matrices.png  # Confusion matrices
├── metrics_comparison.png  # Bar chart so sánh metrics
├── bert_history.png        # BERT training curves
└── bert_eval.png           # BERT confusion matrix + ROC curve
```

> BERT weights không lưu local — load tự động từ [HuggingFace](https://huggingface.co/qhuyisthebest/bert-spam).

## Cấu trúc project

```
Spam-AI/
├── data/
│   └── spam.csv            ← đặt dataset vào đây
├── src/
│   ├── preprocess.py       ← load, clean text, EDA, train/test split
│   ├── ml_models.py        ← TF-IDF pipelines + GridSearchCV
│   ├── bert_model.py       ← BERT fine-tuning với PyTorch
│   └── evaluate.py         ← benchmark comparison, plots
├── models/                 ← output (auto-generated)
├── train.py                ← entrypoint
└── requirements.txt
```

## Chi tiết kỹ thuật

**Preprocessing:**
- Lowercase, loại bỏ URL, số, dấu câu
- Lemmatization + stopword removal (NLTK)
- Feature engineering: char count, word count, digit count, uppercase count

**ML Models:**
- TF-IDF với `max_features=10000`, `ngram_range=(1,2)`, `sublinear_tf=True`
- GridSearchCV 5-fold để tune hyperparameters
- `class_weight='balanced'` cho SVM và LR để xử lý class imbalance

**BERT:**
- `bert-base-uncased`, fine-tuned trên Kaggle (GPU T4 x2), 8 epochs với early stopping (patience=2)
- Max sequence length: 128 tokens, batch size: 32
- Learning rate: 2e-5 với linear warmup (10%)
- Weighted CrossEntropyLoss để xử lý class imbalance
- Model đã train sẵn trên HuggingFace: [qhuyisthebest/bert-spam](https://huggingface.co/qhuyisthebest/bert-spam)
