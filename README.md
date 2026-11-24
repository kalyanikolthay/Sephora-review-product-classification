# Predicting Product Success in Beauty Retail Using Customer Reviews

MSc Business Analytics dissertation, Queen's Business School, Queen's University Belfast (2026)

## Overview

Large retailers carry thousands of products, which makes it difficult to judge which ones are genuinely succeeding with customers. A star rating alone rarely explains why customers respond well to a product, and in skincare the wording of a review often says more than the rating attached to it.

This project tests whether customer review text can be turned into reliable signals of product success. Using Sephora's public skincare catalogue and customer reviews, I extracted five customer-perception concepts from review text, combined them with structured product data, and compared how well six classifiers predicted whether a product is a High or Low performer with and without these features.

## Business Question

Can the language customers use in reviews improve a retailer's ability to identify its strongest products, and which aspects of customer perception matter most?

The project addresses three research questions:

1. Do NLP-derived concept features significantly improve predictive performance over a model built on structured product data alone?
2. Which customer-perception concepts are most strongly associated with High classification, and in which direction?
3. To what extent can changes in concept-level sentiment shift a product's predicted classification from Low to High?

## Data

- **Source:** [Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews) (Kaggle, 2023)
- **Scope:** 2,351 skincare products with customer reviews, out of a full catalogue of 8,494 products
- **Unit of analysis:** the product, with review-level data aggregated to product level

The dataset is not included in this repository. See How to Run below.

## Methodology

### 1. Target variable

The dataset has no "product success" label, so I built one from two measures of customer response:

- **Rating:** an explicit evaluation by customers
- **Loves count:** a behavioural measure of engagement (log-transformed to reduce skew)

Both were standardised and combined with equal weighting. Products in the top quartile were classified as High (25%) and the rest as Low (75%). A sensitivity analysis confirmed that the classification remained stable under alternative weightings. All statistics used to define the target were calculated on the training set only, to prevent data leakage.

### 2. Concept extraction from reviews

I extracted five customer-perception concepts from review text: **effectiveness, irritation, texture, fragrance and value**.

- A zero-shot natural language inference model (`facebook/bart-large-mnli`) first detected whether each concept was mentioned, then classified the sentiment of each mention
- Up to 30 reviews per product were processed, with a mention threshold of 0.8
- The pipeline was validated against 89 manually annotated reviews, selected by stratified random sampling
- Extraction performance was strongest for fragrance (macro F1 = 0.921) and weakest for effectiveness (macro F1 = 0.357)

### 3. Predictive modelling

Six classifiers were trained on two feature sets: structured product data only, and structured data enriched with the concept features.

- Logistic regression
- Elastic net
- Support vector machine (radial kernel)
- K-nearest neighbours
- Random forest
- XGBoost

Each model was tuned with repeated cross-validation, optimising ROC-AUC. Because the High class is the minority, models were also evaluated with threshold tuning (Youden's index), and PR-AUC was used to select the best model.

### 4. Evaluation and interpretation

- **DeLong test:** significance of ROC-AUC differences between structured and enriched models
- **Paired bootstrap:** significance of PR-AUC differences
- **SHAP, model coefficients and permutation importance:** to confirm which concepts drive predictions across several methods, not just one
- **Counterfactual analysis:** to test how far increasing concept sentiment shifts a product's predicted classification, with all structural features held fixed

## Results

### Model comparison

| Model | ROC-AUC (Structured) | ROC-AUC (Enriched) | DeLong p-value | PR-AUC (Structured) | PR-AUC (Enriched) | Bootstrap p-value |
|---|---|---|---|---|---|---|
| Logistic regression | 0.804 | 0.859 | 0.001 | 0.633 | 0.694 | 0.021 |
| Elastic net | 0.819 | 0.866 | 0.005 | 0.640 | 0.712 | 0.003 |
| SVM | 0.829 | 0.870 | 0.016 | 0.636 | 0.700 | 0.008 |
| KNN | 0.758 | 0.779 | 0.328 | 0.604 | 0.578 | 0.453 |
| Random forest | 0.855 | 0.901 | < 0.001 | 0.707 | 0.767 | 0.027 |
| **XGBoost** | **0.856** | **0.913** | **< 0.001** | **0.699** | **0.801** | **0.001** |

### Key findings

1. **Review text adds predictive value.** Concept features significantly improved prediction in five of the six models. KNN was the only exception, which shows that the benefit depends on the type of model used.
2. **XGBoost was the best model.** With concept features, it reached a ROC-AUC of 0.913 and a PR-AUC of 0.801, the largest PR-AUC gain in the study (+0.102).
3. **Effectiveness and fragrance matter most.** These two concepts were the strongest signals of High classification, consistently across SHAP, model coefficients and permutation importance.
4. **Borderline products are the most sensitive.** In the counterfactual analysis, 71% of borderline products (predicted probability 0.35 to 0.5) moved from Low to High when effectiveness and fragrance sentiment increased, compared with 7.5% of products with very low predicted probability.

## Business Recommendations

- Retail analytics or merchandising teams could run this classification periodically to give managers an evidence-based view of which existing products are performing well with customers
- The framework should be used as a first input to assortment reviews, not as the basis for investment or delisting decisions on its own, since managers hold context the model cannot see, such as supply chain issues or external trends
- Products near the classification boundary deserve the closest attention, as they are the most responsive to changes in customer perception

## Limitations

- The dataset is a 2023 snapshot of a single retailer and a single category, which limits how far the findings generalise
- Reviews were obtained through third-party scraping and include incentivised reviews
- The analysis identifies associations, not causes: it shows which concepts are linked to High classification, not what makes a product succeed
- Extraction reliability for effectiveness was relatively low, and the manual annotation was validated by a single rater

## Repository Structure

```
├── R/
│   └── sephora_analysis.R   Full analysis: data preparation, target construction,
│                            concept extraction, modelling, evaluation, SHAP and
│                            counterfactual analysis
├── report/                  Research report and technical report (PDF)
└── README.md
```

## How to Run

1. Download the [Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews) dataset and place `product_info.csv` and the `reviews_*.csv` files in a `data/` folder at the root of the repository
2. Install the R packages listed under Tools
3. For the zero-shot extraction, set up a Python environment with `transformers` and `torch` installed, and set `RETICULATE_PYTHON` to the path of that Python executable
4. Run `R/sephora_analysis.R` from the root of the repository

## Tools

**R:** tidyverse, caret, glmnet, kernlab, kknn, randomForest, xgboost, SHAPforxgboost, pROC, MLmetrics, irr, tidytext, topicmodels, reticulate

**Python (via reticulate):** PyTorch, Hugging Face Transformers

## Author

**Kalyani Kolthay**
MSc Business Analytics (Distinction), Queen's University Belfast
[LinkedIn](https://www.linkedin.com/in/kalyani-sanjay-kolthay)
