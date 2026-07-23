# 🎓 Capstone: Define and Solve an ML Problem

A supervised machine learning project that predicts whether an individual's annual income exceeds $50K using 1994 U.S. Census demographic and employment data. The project walks through the full ML lifecycle — from problem definition to model comparison — using both a traditional model (Decision Tree) and a neural network.

## 📌 Project Overview

This capstone tackles a **binary classification problem**: predicting `income_binary` (`<=50K` vs `>50K`) from features like age, education, occupation, marital status, hours worked per week, and capital gains/losses.

The project is split into two phases:

- **🌳 Part A — Traditional ML:** Data exploration, cleaning, feature engineering, and training a `DecisionTreeClassifier`, tuned via cross-validation and grid search.
- **🧠 Part B — Deep Learning:** Building, training, and evaluating a `Keras` neural network on the same problem, then comparing it head-to-head against the Decision Tree.

## 🗂️ Workflow

1. **Data Exploration (EDA)** — checked class balance, distributions, correlations, and missing values
2. **Data Preparation** — dropped irrelevant/redundant columns, imputed missing values, winsorized outliers, one-hot encoded categoricals
3. **Model Training & Tuning** — trained a baseline Decision Tree, addressed class imbalance with `class_weight='balanced'`, then used `GridSearchCV` to tune `max_depth` and `min_samples_leaf`
4. **Neural Network** — scaled features with `StandardScaler`, built a 3-hidden-layer `Sequential` Keras model (ReLU activations, sigmoid output), trained with SGD and binary cross-entropy loss
5. **Evaluation & Comparison** — compared both models on **accuracy** and **F1 score**, and reflected on whether the added complexity of the neural network was worth it

## 📊 Results

| Metric | Decision Tree | Neural Network |
|---|---|---|
| Accuracy | ~equal | ~equal (+0.0001) |
| F1 Score | lower | ~0.10 higher |

**Takeaway:** The neural network edged out the Decision Tree on F1 score, but the improvement was modest relative to the added complexity and reduced interpretability — a classic real-world tradeoff between model performance and explainability.

## 💼 Practical Use Cases

This notebook isn't just an academic exercise — it mirrors the exact workflow used on real data science and analytics teams:

- **📋 Problem framing first** — clearly defining the target variable and business question before touching code prevents wasted effort down the line, whether you're predicting income, churn, or fraud.
- **🔍 EDA drives every downstream decision** — spotting class imbalance, redundant features, and data quality issues *before* modeling is what separates a robust pipeline from one that silently produces biased or broken predictions.
- **🧹 Data prep is where most of the real work happens** — handling missing values, encoding categoricals, and treating outliers are the same steps used in production feature engineering pipelines at almost any company working with tabular data.
- **⚖️ Model selection is a tradeoff, not a competition** — knowing when a simpler, more interpretable model (like a Decision Tree) is the *better business choice* over a more complex one (like a neural net) is a judgment call every ML practitioner has to make and justify to stakeholders.
- **🧭 Ethical review is a required step, not an afterthought** — thinking through who could be harmed by a model's errors and where bias could creep in is now a standard expectation in industry ML workflows, not an optional add-on.
- **🤖 Using AI tools responsibly** — leveraging AI assistance for debugging, brainstorming prep steps, and validating output while still applying independent judgment reflects how AI is increasingly used as a collaborator in real data science work.

## 🛠️ Tech Stack

- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Traditional ML:** `scikit-learn` (`DecisionTreeClassifier`, `GridSearchCV`, `StandardScaler`)
- **Deep Learning:** `TensorFlow` / `Keras` (`Sequential`, `Dense`, `SGD`)

## 📁 Structure

```
├── AKM-BTT-Capstone.ipynb   # Main notebook (EDA → prep → modeling → comparison)
└── data_capstone/
    ├── censusData.csv        # 1994 U.S. Census dataset (used in this project)
    └── airbnbListingsData.csv # Alternative dataset option (not used here)
```
