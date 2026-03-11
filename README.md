# Project Highlight
- Research extension investigating optimistic performance estimation in standard K-fold cross-validation without nested cross-validation, implemented in `08_research_extension`
- Built an end-to-end regression pipeline with data leakage prevention using`ColumnTransformer()` and `Pipeline()`
- Distinguished structured vs random missingness and designed imputation strategies that preserve semantic meanng in data 
- Improved baseline model performance through feature selection and feature engineering.

Model RMSE improvement: 

Baseline RMSE: 0.147378

Final Ensemble RMSE: 0.1376601  

Improvement: 6.59% reduction in error 

Evaluation metric used : 

Root Mean Square Error: √(1/n ∑ᵢ₌₁ⁿ (yᵢ − ŷᵢ)²)

  - in `03_feature_engineering.ipynb`: 

  Benchmark RMSE from baseline: 0.147378

  | versions | changes | RMSE | remarks |
  |----------|---------|------|---------|
  | v1| added engineered features | 0.14255 | raw features remained in the dataset|
  | v2| removed features that are relevant to the already engineered features| 0.14114| Performance improved after feature removal|

- Compared performances of multiple models using K-fold validation
  - in `04_model_comparison.ipynb`: 

  | Models | mean RMSE | STD |
  |----------|---------|------|
  | Ridge | 0.1411 | 0.0224 |
  | Lasso | 0.1371 | 0.0271 |
  | Elastic Net | 0.1501 | 0.0254 |
  | Random Forest Regressor | 0.1429 | 0.0083 |

  Observations: 
  - Lasso achieved the lowest mean RMSE
  - Random Forest Regressor has the lowest variance 
  - Ridge remained competitive 

- Analysed model residuals using Out-of-Fold (OOF) predictions, in `06_error_analysis.ipynb`, residual by price quantile plot - shows residuals increase at the tails of the target distribution, indicating difficulty modelling extreme house prices 

- Quantified model diversity with residual correlation to make informed ensemble construction
 - in `06_error_analysis.ipynb`: 

  | | ridge_residual| lgb_residual | rfr_residual |
  |----------|---------|------|------|
  | ridge_residual | 1.00000 | 0.7728 | 0.7229 |
  | lgb_residual | 0.7728 | 1.0000 | 0.9319|
  | rfr_residual | 0.7229 | 0.9319 | 1.0000  |

- Enforced convex combination of the weights of ensemble models to prevent overfitting 
  y = w1 yridge + w2 ylgb, subjected to convex constraint: 
  w1 + w2 = 1, wi >= 0 
- identified limitations and proposed improvements in later part *Limitations and Possible Improvements* 
- Treated the project as an experimental study rather than leaderboard optimisation

# Project Structure 
- `01_eda.ipynb` : Exploratory data analysis and dataset understanding
- `02_baseline.ipynb`: Baseline Ridge Regression model 
- `03_feature_engineering.ipynb`: Feature creation and feature selection
- `04_model_comparison.ipynb`: Comparison of classical regression models
- `05_hyperparameter_tuning.ipynb`: GridSearchCV and RandomizedSearchCV 
- `06_error_analysis.ipynb`: Residual analysis and model error investigation
- `07_ensemble.ipynb`: Weighted ensemble construction 
- `08_research_extension.ipynb`: NestedCV experiment

# Problem and Context
This project aims to build an end-to-end machine learning pipeline, exploring feature engineering, classical regression models, model comparison, hyperparameter tuning, and ensemble learning

# Dataset
Source: *Kaggle competition: House Prices - Advanced Regression Techniques*

Loaded from: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

The dataset used is The Ames Housing dataset compiled by Dean De Cock.

This dataset contains mixed types of attributes, including numerical (eg. LotArea - Lot size in sqft), nominal (Neighborhood - name of the neighborhood the house is in) and ordinal (eg. OverallCond - the overall condition rating of the house)

Target variable : SalePrice 

# Exploratory Data Analysis (EDA)
- Explored basic data properties: columns, shape...
- Identified some numerical attributes to have categorical meanings
- plotted target variable distribution and identified target variable is skewed
- separated attributes into groups of: numerical, nominal, and ordinal
- Identified structured missingness in data
- Identified random missing data 
- Conducted correlation analysis: 
  - Used Pearson correlation to analyse the correlation between features and target
  - Used Heatmap to check for correlated features

# Baseline Model
- Used Ridge Regrssion as baseline
- Evaluated with RSME
- Created benchmark RMSE for comparison


# Feature Engineering
- Converted rare categorical values into binary features
- Added more linear combination of features 
- Removed Original columns that were already covered by the newly engineered binary features
- Observed RMSE improvements after removal on baseline model



# Model Comparison 
- Compared Linear models including Ridge, Lasso, ElasticNet and Tree model RandomForestRegressor
- analysed RMSE mean and standard deviation to evaluate both performance and stability


# Hyperparameter Tuning 
- Performed GridSearchCV on Ridge and RandomForestRegressor 
- Introduced LightGBM, and did RandomizedSearchCV on it, LightGBM achieved the best predictive performance 

# Error Analysis
- Used Out Of Fold (OOF) predictions from Ridge, RandomForest, and LightGBM to perform residual analysis
- Residual vs prediction plotted to identify systematic error patterns 
- Checked correlation between predictions and residuals to detect bias
- Checked residual S.D to compare error dispersion between models
- Analysed Prices ranges that each model makes the least error at 
- Checked the worst 5% errors and suggested the causes of these errors
- Extracted feature_importance_ for LightGBM and identified the most important features
- Examined the residual correlation matrix produced by the three models to make informed decisions about building an ensemble 


# Ensemble
- Decided to build a weighted ensemble with LightGBM and Ridge because they can each capture different relationships in data (nonlinear and linear)
- Blended the two models with weights subjected to convex constraint, which ensures that the final prediction lies between the individual model predictions, preventing the ensemble prediction from extrapolating outside the range of individual model predictiions
- Compared RMSE with a ensemble of three blended models
- Decided on final model and produced submission predictions

# Research extension (08_research_extension.ipynb)
To investigate hyperparameter selection bias, a 3x3 Nested Cross - Validation framework was implemented. 
- inner loop: hyperparameter tuning
- outer loop: unbiased performance estimation

Results showed that Nested Cross-Validation produced higher RMSE and higher RMSE variance, suggesting that the original K-fold estimate was optimistically biased 

This experiment highlights the importance of proper evaluation methodology in machine learning experiments 


# Key Learnings
- Not all missing values are random, some are structured, which means they are expected and they represent certain meaning 
- skewed distritbution is identified by the plotted distribution has a long tail on one side, and this will cause the extreme values to dominate loss function
- A linear model is a good baseline model because of its interpretability
- When comparing models, attention should be on both the mean and the variance 
- Certain models may perform better on certain value ranges, better mean rmse does not mean the model is good for the entire range 
- When building an ensemble, adding similar models would likely bring modest improvements since the models make similar mistakes 

# Limitations and Possible Improvements
- The column lists during feature selection were hardcoded. This caused the pipeline to be less reproducible, prone to errors, and risks leakage. A better way will be to include feature selection inside pipeline

- Baseline model imputed most_frequent values to structured missing catgorical columns, it could have distorted the signal for the baseline model. A better way will be to impute with a 'NIL'. This has been improved after 02_baselines.ipynb

- During Feature Engineering, binary features were added and relevant non-binary features were removed, this could have benefitted tree models later unfairly compared to other linear models. The feature engineered did not benefitted Ridge much since most features are still linear combinations. Interaction terms could be added such as OverallQual * GrLivArea to capture interactions

- During Feature Engineering, encoding for categorical features were done outside of pipeline, it could potentially cause data leakage. A better way will be to include them inside the pipeline

- Baseline model started with Ridge, and in model comparison different regularisation were compared, the ordinary least square LinearRegression could be included to compare the effect of regularisation as well

- During hyperparameter tuning, CV results were used to tune the hyperparameters and the same CV results were used to estimate performance, it could have introduced optimism in estimated performance. A better way will be to do Nested CV instead. `08_research_extension` was conducted based on this hypothesis

- For Error Analaysis, I analysed Errors by price quantile, more of errors on key features could be done especially using the key features analysis later on to see if model fails on certain particular feature range. 