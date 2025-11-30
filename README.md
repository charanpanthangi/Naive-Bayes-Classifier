# Naive Bayes Classifier (Gaussian, Multinomial, Bernoulli)

A beginner-friendly tutorial and template for three common Naive Bayes variants using the classic Iris dataset.

## What is Naive Bayes?
Naive Bayes classifiers apply Bayes' theorem: we update a prior belief with evidence to get a posterior belief. The "naive" part assumes that features are conditionally independent given the class. While this assumption is rarely perfect, the models remain fast, simple, and surprisingly strong when the assumption roughly holds.

### When to use each variant
- **GaussianNB**: continuous numeric features (e.g., measurements, sensor data).
- **MultinomialNB**: counts or frequencies (e.g., bag-of-words text counts, event counts).
- **BernoulliNB**: binary indicators (e.g., word present/absent, yes/no features).

## Dataset
The repository uses scikit-learn's Iris dataset (150 flower samples, 4 numeric features, 3 species). It is small enough for quick experimentation and clear visualizations.

## Project structure
```
<repo-root>/
├── app/
│   ├── data.py          # Load the Iris dataset
│   ├── preprocess.py    # Train/test split, scaling, binarization
│   ├── model.py         # Three Naive Bayes model constructors
│   ├── evaluate.py      # Accuracy, precision, recall, F1, confusion matrix
│   ├── visualize.py     # Confusion matrix + PCA plots saved as SVG
│   └── main.py          # End-to-end pipeline
├── notebooks/
│   └── demo_naive_bayes_classifier.ipynb
├── tests/               # Lightweight pytest suite
├── examples/
│   └── README_examples.md
├── requirements.txt
├── Dockerfile
├── .gitignore
└── LICENSE
```

## How the pipeline works
1. **Load data** with `app/data.py`.
2. **Split and preprocess** using `train_test_split`, optional scaling for GaussianNB, and Min-Max + binarization for Multinomial/Bernoulli.
3. **Train models** for all three variants.
4. **Evaluate** accuracy, precision, recall, F1, and confusion matrices.
5. **Visualize** confusion matrices and PCA projections of predictions (saved to `visualizations/`).

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/main.py
```

### Run tests
```bash
pytest
```

### Use Docker
```bash
docker build -t naive-bayes-demo .
docker run --rm naive-bayes-demo
```

## Future improvements
- Add a text classification example with bag-of-words features.
- Use TF-IDF with MultinomialNB for documents.
- Compare results with Logistic Regression as a baseline.

## License
This project is licensed under the MIT License.
