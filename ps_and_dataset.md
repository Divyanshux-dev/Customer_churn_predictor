# Customer Churn Prediction — Problem Framing

## Objective
Build a machine learning system that predicts whether a customer is likely to churn based on customer demographics, account details, subscribed services, and billing information.

## What is churn?
In this project, churn means that a customer has discontinued or canceled the company’s service. The exact definition will be confirmed from the dataset target column.

## Who will use this model?
This model can be used by the retention / CRM / customer success team to identify high-risk customers and take preventive action.

## Business value
Retaining an existing customer is generally cheaper than acquiring a new one. Predicting churn early can help the company reduce revenue loss and prioritize retention efforts.

## What action follows a high churn prediction?
Possible actions include retention offers, support outreach, personalized discounts, or service issue resolution.

## Cost of mistakes
- False Positive: customer is targeted for retention even though they would not churn.
- False Negative: customer is missed and leaves the company.
In many churn settings, false negatives are more costly because the business loses the customer and future revenue.

## ML framing
This is a supervised binary classification problem where:
- Target = churn / no churn
- Inputs = customer-level demographic, service, and billing features
- Output = churn probability and predicted class

# Customer Churn Prediction — Dataset Understanding

## Dataset Overview
- Dataset: **IBM Telco Customer Churn**
- Shape: **7043 rows × 21 columns**
- Each row represents **one customer**

## Target Column
- **Churn**
  - `Yes` → customer left the telecom service
  - `No` → customer stayed

## Important Columns
### Identifier
- `customerID` → unique customer ID, drop before modeling

### Numerical Features
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

### Categorical Features
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- `PhoneService`, `MultipleLines`, `InternetService`
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`
- `StreamingTV`, `StreamingMovies`
- `Contract`, `PaperlessBilling`, `PaymentMethod`

## Key Findings
- **No duplicate rows**
- **No duplicate customer IDs**
- `Churn` is **imbalanced**:
  - **No** → 5174
  - **Yes** → 1869
- `TotalCharges` has **11 blank values** and is stored as **object**
  - needs cleaning
  - convert to numeric

## Preprocessing Decisions
- Drop `customerID`
- Convert `TotalCharges` to numeric
- Handle missing values in `TotalCharges`
- Encode categorical columns
- Convert target `Churn` to 0/1 for modeling

## Likely Important Features
- `tenure`
- `Contract`
- `MonthlyCharges`
- `InternetService`
- `TechSupport`
- `PaymentMethod`

## EDA Focus
- Churn distribution
- Churn vs tenure
- Churn vs contract
- Churn vs monthly charges
- Churn vs internet service
- Churn vs payment method
- Churn vs tech support