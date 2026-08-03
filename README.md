# 📄 Resume Screening Classifier

This project builds an end-to-end NLP pipeline to classify resumes into 25 job categories. It uses text cleaning, TF‑IDF vectorization with bigrams, and a comparison of six classic classifiers, with Linear SVM (tuned via GridSearchCV) as the final model. The focus is on a clean, reproducible multi-class text classification workflow: from EDA and preprocessing to model evaluation, hyperparameter tuning, and auditable predictions.

---

## ✨ What I Did and Why

- **Text Cleaning (`Utility.py`):** Lowercased text, stripped URLs, HTML tags, punctuation, and digits, removed NLTK stopwords, and dropped tokens shorter than 2 characters. Resume text is noisy — raw symbols and common words like "the" and "and" add no category signal, so removing them lets the model focus on meaningful terms like tools, frameworks, and domain keywords.

- **TF-IDF Vectorization:** Converted cleaned resume text into numerical features using Term Frequency-Inverse Document Frequency with bigrams (`ngram_range=(1,2)`) and a 20,000 feature cap. TF-IDF down-weights words that appear across all resumes (like "experience" or "work") and up-weights words that are distinctive to a category. Bigrams let the model pick up short phrases like "machine learning", "data analysis", or "project management" that single words miss.

- **Six Classifiers Compared:** Trained and evaluated six classic ML classifiers — Linear SVM, Passive Aggressive, Ridge Classifier, Complement NB, Logistic Regression, and Multinomial NB — on the same TF-IDF features. Running multiple baselines before tuning is good practice: it shows which model family suits the data structure before spending time on hyperparameter search.

- **Train/Test Split with Stratification:** Evaluated on a held-out 20% test set (`random_state=42`) with stratified sampling to preserve the class distribution across 25 job categories. Raw text was preserved separately (`X_test_raw`) before TF-IDF transformation so the prediction cell could correctly pass new strings through the vectorizer.

- **GridSearchCV Tuning on Linear SVM:** Fine-tuned the best model (Linear SVM) using 3-fold cross-validated grid search over `C ∈ {0.01, 0.1, 1, 10}`, optimising for weighted F1. Best `C = 10`, achieving **100% accuracy and F1 on the test set**.

- **Predictions CSV:** Exported all 193 test set predictions to `test_predictions.csv` with columns for resume text, true label, predicted label, and a `correct` flag — making results reproducible and auditable without re-running the notebook.

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Data Analytics:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn (`TfidfVectorizer`, `LinearSVC`, `GridSearchCV`, `train_test_split`, `accuracy_score`, `f1_score`, `confusion_matrix`)
- **NLP:** NLTK (stopwords), custom `Utility.py` for cleaning, model registry, evaluation, and confusion matrix plotting
- **Dataset:** `ResumeDataset.csv` — 962 resumes across **25 job categories**

---

## 🚀 Setup

1. Clone the repo.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn nltk
   ```
3. Download NLTK stopwords (once):
   ```python
   import nltk
   nltk.download('stopwords')
   ```
4. Place `ResumeDataset.csv` in the repo root.
5. Run the notebooks in order:
   - `01_EDA.ipynb` → explore the data
   - `02_Data_Cleaning.ipynb` → generate `resume_cleaned.csv`
   - `03_Model_Training.ipynb` → train, evaluate, tune, and export predictions

---

## 📊 Results

**Dataset:** 962 resumes · 25 job categories · 80/20 train-test split

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Linear SVM** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Passive Aggressive | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Ridge Classifier | 0.9948 | 0.9957 | 0.9948 | 0.9949 |
| Complement NB | 0.9948 | 0.9956 | 0.9948 | 0.9947 |
| Logistic Regression | 0.9896 | 0.9907 | 0.9896 | 0.9895 |
| Multinomial NB | 0.9793 | 0.9817 | 0.9793 | 0.9762 |

**After GridSearchCV tuning (Linear SVM, best C = 10):**
- Test Accuracy: **100%** · Weighted F1: **100%**
- All 193 test predictions correct (`test_predictions.csv`)

**Key observation:** Linear SVM and Passive Aggressive both hit perfect scores on this dataset. Linear SVM was selected as the tuning target because it has a well-understood regularisation parameter (`C`) and is the standard strong baseline for high-dimensional sparse text features.

<img width="930" height="790" alt="image" src="https://github.com/user-attachments/assets/784f13d5-fd26-4282-9e64-1fe7377402b0" />


---

## ⚠️ Limitations

- **Perfect test accuracy is suspicious on small data.** 962 samples across 25 categories averages ~38 per class. With such small per-class counts and TF-IDF on domain-specific vocabulary, the model may be memorising category-specific jargon rather than learning generalisable patterns. Performance on truly unseen resumes from the wild may be lower.
- **No lemmatization or stemming.** "Developed", "developer", and "developing" are treated as three separate tokens. Adding lemmatization could reduce noise and improve generalisation.
- **TF-IDF loses word order entirely.** "Not experienced in Python" and "experienced in Python" produce near-identical feature vectors.
- **Fixed vocabulary.** The vectorizer was fit on training data. Resumes containing tools or skills not seen during training will have those terms ignored entirely.
- **Dataset age and source.** Resume phrasing, in-demand tools, and job titles change over time. A model trained on this dataset may not reflect current industry language.
