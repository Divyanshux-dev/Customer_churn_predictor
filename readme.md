# Customer Churn Predictor

End-to-end machine learning project to predict telecom customer churn using the IBM Telco Customer Churn dataset.

## Objective
Build a production-style ML pipeline that:
- analyzes churn behavior through EDA
- preprocesses customer data
- trains churn prediction models
- exposes predictions through an API
- supports deployment and monitoring extensions

## Dataset
- **Dataset:** IBM Telco Customer Churn
- **Target column:** `Churn`

## Planned Project Structure
```text
Customer_Churn_Predictor/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── api/
│   └── utils/
├── tests/
├── artifacts/
├── requirements.txt
├── README.md
└── .gitignore