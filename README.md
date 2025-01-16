ThreeMobile Customer Churn Analysis
===================================

Overview
--------

The ThreeMobile Customer Churn Analysis project aims to analyze customer churn in a telecommunications context using data science methods. The project prioritizes performance insights and churn prediction using statistical and machine learning methods, specifically emphasizing deep survival analysis techniques to understand customer lifetimes.


Description
-----------

This project employs a comprehensive analysis of customer churn using demographic data, usage patterns, and transaction history to create predictive models. By employing deep learning for survival analysis, we attempt to estimate the time until a customer might churn and the factors leading up to this event.

Key components include:

-   Data processing and imputation techniques.
-   Building a deep learning model (`DeepSurv`) to predict survival probabilities.
-   Visualizing survival curves to analyze risk and customer behavior over time.
-   Analyzing various customer features and their correlations with churn.

Prerequisites
-------------

Ensure you have the following Python packages installed:

-   `pandas`
-   `numpy`
-   `torch` (PyTorch)
-   `scikit-learn`
-   `lifelines`
-   `seaborn`
-   `matplotlib`
-   `imblearn`

Installation
------------

To set up the project locally, follow these commands in your terminal:

1.  Clone the Repository

    ```
    git clone https://github.com/yourusername/customer-churn-analysis.git
    cd customer-churn-analysis
    ```

2.  Create and Activate a Virtual Environment

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install Required Dependencies

    ```
    pip install -r requirements.txt
    ```

Data Preparation
----------------

The dataset consists of several features relevant to customer interactions and churn indicators. Key steps involve:

1.  Loading data: Loading relevant datasets, including customer demographics and churn history.

2.  Handling Missing Values: Using KNN Imputer for filling missing values:

    ```
    1
    2
    3
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=15, weights='uniform')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    ```

3.  Feature Selection: Selecting relevant features for survival analysis, such as:

    -   UserID
    -   TenureInMonths
    -   Churned
    -   Age
    -   ContractRiskScore
    -   Spend and Usage metrics.

Deep Survival Analysis
----------------------

The core attractive feature is the implementation of a deep survival analysis using a neural network:

1.  Survival Dataset Class: This Structure encapsulates the features along with duration and event status.

    ```
    1
    2
    3
    4
    5
    class SurvivalDataset(Dataset):
        def __init__(self, X, durations, events):
            self.X = torch.FloatTensor(X.values)
            self.durations = torch.FloatTensor(durations.values)
            self.events = torch.FloatTensor(events.values)
    ```

2.  DeepSurv Model: The model architecture consists of:

    -   Multiple linear layers with ReLU activations and dropout for regularization.

    ```
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    class DeepSurv(nn.Module):
        def __init__(self, input_dim):
            super(DeepSurv, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(16, 1)
            )
    ```

3.  Loss Function: We utilize negative log likelihood to appropriately handle censoring in survival analysis.

    ```
    1
    2
    3
    def negative_log_likelihood(risk_pred, durations, events):
        # Implementation...
        return neg_likelihood
    ```

4.  Model Training: The model is trained for 150 epochs with gradient descent optimization:

    ```
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ```

Visualization
-------------

The survival analysis components are visualized using:

1.  Individual Survival Curves: Plotting survival probabilities for sample users.
2.  Average Survival Curves: Illustrating survival probabilities segmented by risk groups.
3.  Probability Distributions: Visual analysis of churn likelihood across different risk categories using KDE plots:

    ```
    sns.kdeplot(data=predictions[predictions['risk_group'] == risk_group]['survival_prob_90d'], fill=True)
    ```

Usage Instructions
------------------

Once the model is trained, you can predict survival probabilities using the `get_survival_predictions` function:

```
predictions = get_survival_predictions(deep_surv_model, analysis_df)
```

Run visual analyses and evaluate the models against known churn values to validate prediction accuracy.

Model Evaluation
----------------

To gauge the model's performance, evaluate metrics like:

-   AUC-ROC
-   Confusion matrices
-   Classification reports
