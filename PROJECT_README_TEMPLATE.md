# House Price Prediction - Linear Regression Project

## 1. Project Overview
**Objective:** Predict house listing prices using linear regression based on property features.

**Dataset:** NY Real Estate Sold Properties (2026)
- Total records: 10,254
- Final dataset (after cleaning): ~3,500 records
- Target variable: `listPrice`

---

## 2. Problem Statement
Given property characteristics (sqft, bathrooms, stories, bedrooms, garage), predict the listing price accurately using linear regression.

---

## 3. Exploratory Data Analysis (EDA)

### Data Exploration
- Checked data shape, types, and missing values
- Identified 23 columns with varying data quality
- Found significant missing values in `garage` (51%) and `stories` (30%)

### Feature Correlation Analysis
Analyzed correlation of numerical features with `listPrice`:

| Feature | Correlation | Status |
|---------|------------|--------|
| baths | 0.57 | Strong |
| sqft | 0.42 | Moderate |
| stories | 0.37 | Moderate |
| beds | 0.19 | Weak |
| garage | 0.11 | Very Weak |
| zip_code | -0.36 | Weak (Negative) |

**Decision:** Selected features with correlation > 0.3 + beds for modeling.

### Data Visualization
- Histograms: Checked distribution of numerical features
- Heatmap: Analyzed feature correlations
- Box plots: Identified outliers

---

## 4. Data Cleaning & Preprocessing

### Handling Missing Values
- **Approach:** Removed rows with missing values (not filled)
- **Reason:** Preserves data integrity; enough records remaining
- Rows removed: 4,177 → 3,916 (after outlier removal)

### Outlier Removal
- **Method:** IQR (Interquartile Range)
- **Formula:** 
  - Q1 = 25th percentile
  - Q3 = 75th percentile
  - IQR = Q3 - Q1
  - Bounds: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- **Result:** Removed extreme luxury properties (>$5M equivalent range)

### Feature Engineering
- **One-Hot Encoding:** Converted categorical features (`city`, `type`, `size_tier`) to binary columns
- **Reason:** Linear regression requires numerical input
- **Result:** Expanded features from 5 to 50+ (due to categorical encoding)

### Final Features Used
1. sqft (continuous)
2. stories (continuous)
3. baths (continuous)
4. beds (continuous)
5. garage (continuous)
6. city_* (one-hot encoded, 10+ cities)
7. type_* (one-hot encoded, property types)
8. size_tier_* (one-hot encoded, size categories)

---

## 5. Model Development

### Algorithm: Linear Regression
- **Why Linear Regression?** Good starting point for regression; interpretable coefficients
- **Library:** scikit-learn `LinearRegression()`

### Train-Test Split
- **Split Ratio:** 80% training, 20% testing
- **Random State:** 42 (for reproducibility)
- Training samples: ~3,130
- Testing samples: ~786

### Model Training
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

---

## 6. Model Evaluation

### Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **R² Score** | 0.6600 | Model explains 66% of price variance |
| **RMSE** | $358,000 | Average prediction error |
| **MAE** | $147,000 | Mean absolute deviation |

### Performance Progression
- Initial model (3 features): R² = 0.26
- After adding beds + garage: R² = 0.44
- After categorical encoding: R² = 0.62
- After removing weak features (zip_code): R² = 0.66

### Model Strengths
✅ Reasonable R² score (0.66)
✅ Interpretable coefficients
✅ Fast training and prediction
✅ Good generalization (test R² close to training R²)

### Model Limitations
❌ Still explains only 66% of variance
❌ May underperform on luxury properties
❌ Assumes linear relationships
❌ RMSE of $358k is significant for price prediction

---

## 7. Key Learnings

### Technical Skills Learned
1. **EDA & Visualization:** histograms, heatmaps, scatter plots
2. **Data Cleaning:** Handling missing values, outlier detection (IQR method)
3. **Feature Engineering:** One-hot encoding for categorical data
4. **Feature Selection:** Using correlation analysis to choose relevant features
5. **Model Evaluation:** R², RMSE, MAE metrics and interpretation
6. **Train-Test Split:** Avoiding data leakage

### Data Science Insights
- Feature selection is crucial — added features improved R² by 1.54x (0.26 → 0.66)
- Categorical data requires encoding; one-hot encoding works well for linear models
- Linear relationships exist but are weak — other patterns unexplained
- Domain knowledge matters — removed zip_code despite thinking it would help initially

### Process Insights
1. Always start with EDA, not modeling
2. Don't add features blindly — validate with metrics
3. Removing outliers improves model quality
4. Document everything for reproducibility

---

## 8. Next Steps / Improvements

### Possible Enhancements
1. **Try polynomial regression** — Capture non-linear relationships
2. **Use advanced models:** Random Forest, Gradient Boosting
3. **Feature scaling:** Normalize features for better performance
4. **Hyperparameter tuning:** Optimize model parameters
5. **Cross-validation:** Better estimate of model performance
6. **Add more features:** Location details, property age, etc.

### Why NOT done in this project
- Kept it simple for **first project** (learning linear regression)
- Focusing on understanding over optimization
- Moving to **logistic regression next** for new learning

---

## 9. Files & Code
- **Notebook:** `houser_price_pred.ipynb`
- **Dataset:** `ny_real_estate_sold_properties_2026.csv`
- **Language:** Python 3
- **Libraries:** pandas, scikit-learn, matplotlib, seaborn, numpy

---

## 10. Conclusion
Successfully built a linear regression model achieving **R² = 0.66** for house price prediction. The project demonstrates understanding of the complete ML pipeline: exploration → cleaning → feature engineering → modeling → evaluation. While there's room for improvement, the model provides reasonable price estimates and serves as a solid foundation for learning regression concepts.

**Status:** ✅ Complete (First Project)
**Next:** Start logistic regression project
