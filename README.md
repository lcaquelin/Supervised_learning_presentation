# Supervised_learning_presentation
# Ad Conversion Optimization & Efficiency Prediction

## 📄 Description
Multi-target ML project predicting ad conversions and efficiency. Benchmarks several models (Linear, Lasso, RF, GB, etc.) using MLflow. Features rigorous hyperparameter tuning via Hyperopt/GridSearchCV to optimize metrics on conversion data.

## 🎯 Objectives
The primary goal of this project is to predict the performance of social media advertisements based on audience demographics and campaign details.

Unlike standard approaches, this project uses a **Multi-Target Regression** strategy to predict two key metrics simultaneously:
1.  **`approved_conversion`**: The absolute number of final product purchases (Volume).
2.  **`spent_on_conversion_ratio`**: A custom efficiency metric representing "Sales per Euro Spent" (ROI/Efficiency).

## 📊 Dataset
**Source:** [Kaggle - Clicks Conversion Tracking](https://www.kaggle.com/datasets/loveall/clicks-conversion-tracking/data) (File: `KAG_conversion_data.csv`)

The dataset contains data from an anonymous organization's social media ad campaigns.

**Key Features:**
* **Campaign Info:** `xyz_campaign_id` (General), `fb_campaign_id` (Ad Set), `ad_id` (Unique Creative).
* **Demographics:** `age`, `gender`.
* **User Interest:** `interest` (Categorical code from Facebook public profile).
* **Performance Metrics:** `Impressions`, `Clicks`, `Spent`, `Total_Conversion` (Marketing funnel and amount spent for the ad).

## 📈 Exploratory Data Analysis (EDA) & Preprocessing

The analysis pipeline follows a structured approach to clean data and uncover drivers of ad performance.

### 1. Data Cleaning & Preprocessing
* **Type Conversion:** Categorical features (`ad_id`, `xyz_campaign_id`, `fb_campaign_id`, `age`, `gender`, `interest`) were converted to optimization-friendly formats.
* **Feature Selection:**
    * Dropped `ad_id` (Unique identifier with no predictive value).
    * **Removed Multicollinear Features:** Dropped `Impressions`, `Clicks`, and `Total_Conversion`. These features are highly correlated with the final target and represent "intermediate" steps in the funnel. Removing them forces the model to predict conversions based on *user characteristics* (Age, Interest, Gender) rather than just "volume of traffic".

### 2. Demographic & Global Analysis
* **Univariate:** Analyzed distributions using histograms and boxplots to detect outliers.
* **Demographics:** Visualized Age and Gender distribution via pie charts and grouped bar charts.
* **Correlation:** Generated a heatmap to inspect relationships between numerical variables (leading to the removal of intermediate funnel metrics).
* **Bivariate Analysis:** Examined the direct relationship between demographics and performance using boxplots (Age vs. Approved Conversion, Gender vs. Approved Conversion)

### 3. Deep Dive: Interest Analysis
A significant portion of the EDA focuses on the `interest` feature to identify high-performing audience niches.
* **Pareto Analysis:** Analyzed the cumulative distribution of interest categories (checking the 80/20 rule).
* **Treemap Visualization:** Maps **Frequency** (Box Size) vs. **Efficiency** (Color Scale: Red to Green). This allows instant identification of interests that are frequently used but yield poor results (Red/Large) vs. niche interests that perform exceptionally well (Green/Small).
* **Bubble Chart:** Visualizes **Frequency** (X-axis) vs. **Efficiency/Mean Conversion** (Y-axis). The **Bubble Size** represents `Average_Spent`. This reveals "Money Pits" (Large bubbles, Low Efficiency) and "Stars" (High Efficiency).

### 4. 🛠 Feature Engineering
### Custom Target: `spent_on_conversion_ratio`
To measure ad cost-effectiveness, we engineered a custom ratio metric based on the actual spend per sale.
* **Formula:** `Spent / Approved_Conversion`
* **Handling Zeros:** If `Approved_Conversion` is 0, the ratio is set to **0**.
* **Logic:** This calculates the cost incurred to generate one approved conversion. While mathematically an undefined cost (division by zero) usually implies infinite cost, we impute these cases as 0 to maintain numerical stability for the regression models.

### Models Benchmarked
We tested models of increasing complexity, handling multi-output regression either natively or via wrappers (`MultiOutputRegressor`).

## 🤖 Modeling Strategy
The project uses **MLflow** to track all experiments and a **Multi-Target Regression** approach (via `sklearn.multioutput.MultiOutputRegressor`) to predict both targets with a single model instance.

### Experiments Tracked (MLflow)
All metrics (MAE, MSE, R²) are logged to MLflow and summarized in `out/score.txt`.

| Model Pipeline | Type | Key Characteristics |
| :--- | :--- | :--- |
| **1. Dummy Regressor** | Baseline | Predicts mean values. Serves as the performance floor. |
| **2. Linear Regression** | Linear | Standard OLS regression. |
| **3. Lasso Regression** | Linear (Regularized) | L1 regularization to penalize complex models (`alpha=1.0`). |
| **4. Polynomial Regression** | Non-Linear | Degree 2 polynomial features to capture curvature. |
| **5. Poly + Lasso** | Hybrid | Degree 2 features combined with Lasso to control overfitting. |
| **6. Decision Tree** | Tree-based | Captures non-linear patterns and interactions naturally. |
| **7. Random Forest & Others** | Ensemble | (Planned/Next Steps) for higher accuracy. |

### Evaluation Metrics
For each model, we calculate and visualize:
* **Metrics:** Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.
* **Visuals:** "Actual vs. Predicted" scatter plots with a diagonal reference line.

### Optimization Goal (Hyperopt)
The hyperparameter tuning aims to maximize the **MAE** with **Hyperopt**, with a specific focus on the `spent_on_conversion_ratio` target.

## 📦 Requirements
* Python 3.x
* os
* pandas
* matplotlib
* seaborn
* squarify
* scikit-learn
* numpy
* mlflow
* hyperopt
* lightgbm

## 🚀 Usage
1.  **Setup:** Ensure `KAG_conversion_data.csv` is in the project root.
2.  **Install dependencies:** `pip install -r requirements.txt`
3.  **Run the training script:**
    * This will train all models sequentially.
    * Metrics are appended to `out/score.txt`.
    * MLflow runs are created in the `mlruns` directory.
4.  **Analyze Results:**
    * Check `out/score.txt` for a quick text summary.
    * Launch the dashboard:
        ```bash
        mlflow ui
        ```


