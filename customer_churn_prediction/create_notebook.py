import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Customer Churn Prediction\n",
    "\n",
    "In this notebook, we load our customer data, perform EDA, and train several machine learning models to predict customer churn."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from xgboost import XGBClassifier\n",
    "from imblearn.over_sampling import SMOTE\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Data\n",
    "Let's load the generated synthetic dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('churn_data.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Data Preprocessing\n",
    "Handle missing values (e.g., spaces in TotalCharges)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')\n",
    "df.fillna(df['TotalCharges'].mean(), inplace=True)\n",
    "df.drop('customerID', axis=1, inplace=True)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We also visualize our target variable to understand the baseline churn rate before balancing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.countplot(data=df, x='Churn')\n",
    "plt.title('Distribution of Churn')\n",
    "plt.show()\n",
    "print(df['Churn'].value_counts(normalize=True))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Encoding and Train-Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "target = 'Churn'\n",
    "X = df.drop(target, axis=1)\n",
    "y = df[target].apply(lambda x: 1 if x == 'Yes' else 0)\n",
    "\n",
    "X = pd.get_dummies(X, drop_first=True)\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)\n",
    "X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Handling Imbalanced Data using SMOTE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "smote = SMOTE(random_state=42)\n",
    "X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)\n",
    "\n",
    "print('Original train shape:', X_train_scaled.shape)\n",
    "print('Original train churn counts:', y_train.value_counts())\n",
    "print('\\nSMOTE train shape:', X_train_sm.shape)\n",
    "print('SMOTE train churn counts:', y_train_sm.value_counts())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Model Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "models = {\n",
    "    'Logistic Regression': LogisticRegression(random_state=42),\n",
    "    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),\n",
    "    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')\n",
    "}\n",
    "\n",
    "results = []\n",
    "\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train_sm, y_train_sm)\n",
    "    y_pred = model.predict(X_test_scaled)\n",
    "    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]\n",
    "    \n",
    "    acc = accuracy_score(y_test, y_pred)\n",
    "    prec = precision_score(y_test, y_pred)\n",
    "    rec = recall_score(y_test, y_pred)\n",
    "    f1 = f1_score(y_test, y_pred)\n",
    "    auc = roc_auc_score(y_test, y_pred_proba)\n",
    "    \n",
    "    results.append({\n",
    "        'Model': name,\n",
    "        'Accuracy': acc,\n",
    "        'Precision': prec,\n",
    "        'Recall': rec,\n",
    "        'F1 Score': f1,\n",
    "        'ROC-AUC': auc\n",
    "    })\n",
    "\n",
    "results_df = pd.DataFrame(results).sort_values(by='Recall', ascending=False)\n",
    "display(results_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Evaluation\n",
    "With Churn, identifying customers who might leave is highly critical. Therefore, we prioritize **Recall**. Let's review the Confusion Matrix of the Top Model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_model_name = results_df.iloc[0]['Model']\n",
    "best_model = models[best_model_name]\n",
    "best_predictions = best_model.predict(X_test_scaled)\n",
    "\n",
    "cm = confusion_matrix(y_test, best_predictions)\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])\n",
    "plt.title(f'Confusion Matrix - {best_model_name}')\n",
    "plt.ylabel('True Label')\n",
    "plt.xlabel('Predicted Label')\n",
    "plt.show()\n",
    "\n",
    "print(f\"\\nClassification Report for {best_model_name}:\\n\")\n",
    "print(classification_report(y_test, best_predictions))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Feature Importance & Insights"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if hasattr(best_model, 'feature_importances_'):\n",
    "    importances = best_model.feature_importances_\n",
    "elif hasattr(best_model, 'coef_'):\n",
    "    importances = np.abs(best_model.coef_[0])\n",
    "else:\n",
    "    importances = np.zeros(len(X.columns))\n",
    "\n",
    "feat_importances = pd.Series(importances, index=X_train_sm.columns)\n",
    "feat_importances.nlargest(10).plot(kind='barh', title='Top 10 Important Features').invert_yaxis()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🚀 Conclusion & Business Impact\n",
    "\n",
    "Based on the analysis and feature importances, we can typically notice the following:\n",
    "- **Contracts**: Month-to-month contracts tend to be strong drivers of churn.\n",
    "- **Cost**: Higher charges can influence churn positively.\n",
    "- **Tenure**: Customers with shorter tenure are much more likely to abandon the service.\n",
    "\n",
    "### Recommended Actions:\n",
    "1. **Incentivize Long-term Contracts**: Offer discounts for users upgrading from month-to-month to one-year contracts.\n",
    "2. **Improve Onboarding**: Target new customers (low tenure) with exceptional support or check-ins.\n",
    "3. **Proactive Outreach**: Utilize our ML model to flag users at high risk of churn and provide targeted retention offers."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("Customer_Churn_Prediction.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
