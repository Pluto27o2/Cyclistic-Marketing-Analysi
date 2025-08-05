# Credit Risk Classification Using Machine Learning

## 1. Introduction

### 1.1. Background and Objective

In today’s financial industry, managing **credit risk** — the potential for financial loss when a borrower fails to repay a debt — is a critical function. To make lending decisions that are **fast, objective, and accurate**, financial institutions increasingly rely on **data-driven credit scoring models**, moving away from traditional manual assessment methods.

The central objective of this project is to **build a machine learning model** to classify credit customers into **five distinct risk groups**, based on their historical debt status and repayment behavior.

This multi-tiered classification is strategically important, as it enables financial institutions to:

- Tailor credit policies more precisely  
- Accelerate suitable loan approvals  
- Minimize exposure to default risk  
- Ensure fairness in lending decisions  

---

### 1.2. Approach

To achieve this objective, the project follows a three-stage scientific process:

#### 1. Data Preprocessing

- Data cleaning and feature engineering  
- **Numerical variables**: Log-transformed and imputed with medians where necessary  
- **Categorical variables**: One-hot encoded  
- **Class imbalance**: Addressed using **SMOTE** to improve representation of minority classes  

#### 2. Model Development and Training

Six classification models were implemented and compared:

- **Traditional model**:  
  - Logistic Regression — valued for interpretability  

- **Tree-based models**:  
  - Decision Tree  
  - Random Forest  
  - LightGBM — efficient and high-performing  

- **Advanced models**:  
  - CatBoost  
  - Neural Network — deep learning-based exploration  

All models were trained on **SMOTE-balanced data** using a **stratified train-test split**.

#### 3. Model Evaluation and Selection

Models were evaluated on a **hold-out test set** using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

Special attention was given to **class-wise performance** and the **distribution of predicted classes**.  
**LightGBM** achieved the best trade-off between **accuracy** and **minority class detection**, making it the most suitable for deployment.

---

## 2. Data Exploration & Preparation

### 2.1. Dataset Overview

The dataset includes **financial**, **demographic**, and **loan-specific** features to predict a borrower's credit risk. Key variable groups are summarized below.

---

### 2.1.1. Core Financial and Behavioral Variables

| Variable         | Description                                                     |
|------------------|-----------------------------------------------------------------|
| `BASE_BAL`       | Original principal amount of the loan                          |
| `CURR_BAL`       | Current outstanding balance (repayment progress)               |
| `DUNO_QD`        | Overdue balance — strong predictor of default                  |
| `LAISUAT`        | Annual interest rate (%)                                       |
| `DURATION_DAYS` / `DURATION_MONTHS` | Loan tenure in days or months                       |

---

### 2.1.2. Loan Characteristics

| Variable                  | Description                                                  |
|---------------------------|--------------------------------------------------------------|
| `MJACCTTYPCD`, `MJACCTTYPDESC` | Loan type (e.g., Commercial, Consumer, Mortgage)           |
| `PHUONGTHUCCHOVAY`        | Lending method (e.g., term loan)                             |
| `ID_TIME`, `DESC_TIME`    | Loan term classification (short, medium, long-term)          |
| `MUCDICHVAY`              | Purpose of the loan                                          |
| `CURRENCY`                | Currency (e.g., VND, USD) — USD loans carry forex risk       |

---

### 2.1.3. Customer Information

| Variable   | Description                                      |
|------------|--------------------------------------------------|
| `LOAIKH`   | Customer type (1 = Individual, 2 = Corporate)    |
| `SEX`      | Gender (Male, Female); blank for corporates      |

---

### 2.1.4. Transaction and Organizational Information

| Variable             | Description                                       |
|----------------------|---------------------------------------------------|
| `OPEN_DATE`          | Loan origination date                             |
| `NGAYDENHAN`         | Loan maturity date                                |
| `ORGNBR`, `ORGNAME`  | Lending branch ID and name                        |
| `PARENTORGNBR`, `PARENTORGNAME` | Parent organization ID and name                   |

---

### 2.1.5. Target Variable: `NHOMNOMOI`

The target variable defines the **borrower’s credit risk group**, categorized as follows:

| Group | Description                                                                 |
|--------|-----------------------------------------------------------------------------|
| **1. Excellent**     | Timely payments, minimal credit risk                          |
| **2. Low-Risk**      | Minor payment delays, generally reliable                      |
| **3. Moderate-Risk** | Occasional issues, early signs of financial distress          |
| **4. High-Risk**     | Consistently late payments, increased likelihood of default   |
| **5. Very High-Risk**| Significantly overdue or defaulted, major risk of loss        |

This classification allows the organization to:

- Improve segmentation  
- Tailor risk-adjusted credit policies  
- Accelerate appropriate approvals  
- Minimize financial loss  
- Ensure fair and risk-sensitive lending decisions  

---

> ⚠️ *Note:* All data has been either anonymized or simulated. Model development complies with responsible and ethical AI practices.

