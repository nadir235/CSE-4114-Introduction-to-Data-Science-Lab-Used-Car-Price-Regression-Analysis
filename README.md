# Used Car Price Prediction --- Regression Analysis

**Course:** CSE4114 --- Introduction to Data Science\
**Project type:** Supervised Machine Learning --- Regression\
**Dataset:** `car_dataset.csv`\
**Target variable:** `price` (Bangladeshi Taka, BDT)

## Overview

This project explores whether the asking price of a used car in
Bangladesh can be estimated from observable vehicle details such as
brand, model year, engine capacity, mileage, transmission, body type,
and fuel system.

The analysis is implemented in a Google Colab/Jupyter notebook and
covers the full machine-learning workflow: data inspection, cleaning,
exploratory data analysis, feature engineering, feature selection, model
training, evaluation, and export of results.

## Objectives

-   Inspect and understand the used-car listing dataset.
-   Handle missing values, duplicate listings, impossible values, and
    extreme outliers.
-   Explore patterns in price and vehicle specifications using
    statistics and visualizations.
-   Create useful features, including vehicle age and usage intensity.
-   Encode categorical and multi-label fuel information for machine
    learning.
-   Select informative features using multiple methods.
-   Train and compare regression models using consistent evaluation
    metrics.
-   Review model errors and identify important price drivers.

## Dataset

The notebook expects a CSV file named `car_dataset.csv`, containing
approximately 1,209 listings and 10 columns.

  Column              Description
  ------------------- ---------------------------------------------------
  `car_name`          Full listing title
  `brand`             Vehicle manufacturer
  `car_model`         Model line
  `model_year`        Model/registration year
  `transmission`      Transmission type
  `body_type`         Vehicle body style
  `fuel_type`         Fuel system; may contain multiple values
  `engine_capacity`   Engine displacement in cc
  `kilometers_run`    Odometer reading in kilometres
  `price`             Advertised asking price in BDT; prediction target

The notebook describes the source as public used-car classified listings
from the Bangladeshi resale market. The dataset file itself is required
to run the analysis.

## Notebook Workflow

1.  **Problem statement and objectives** --- defines the regression
    problem and project goals.
2.  **Dataset description** --- explains the source, columns, and data
    characteristics.
3.  **Data collection/loading** --- uploads `car_dataset.csv` in Google
    Colab or loads a local copy.
4.  **Data cleaning and preprocessing** --- handles missing values,
    duplicates, impossible records, and outliers.
5.  **Exploratory data analysis (EDA)** --- examines price
    distributions, numerical variables, categories, and correlations.
6.  **Data visualization** --- uses plots to inspect relationships
    between price and vehicle characteristics.
7.  **Feature engineering** --- creates features such as `car_age` and
    `km_per_year_log`, transforms skewed variables, and encodes
    categorical data.
8.  **Feature selection** --- applies an F-test, tree-based feature
    importance, and Recursive Feature Elimination (RFE), and checks
    multicollinearity.
9.  **Model development** --- compares nine regression approaches:
    -   Linear Regression
    -   Ridge
    -   Lasso
    -   ElasticNet
    -   Decision Tree Regressor
    -   Random Forest Regressor
    -   Gradient Boosting Regressor
    -   K-Nearest Neighbours (KNN) Regressor
    -   Support Vector Regression (SVR)
10. **Model evaluation** --- compares models using R², RMSE, MAE, and
    MAPE, with cross-validation and hyperparameter tuning.
11. **Results and export** --- reviews predictions and residuals,
    examines feature importance, provides a reusable prediction
    function, and exports results and the trained model.

## Tools and Libraries

The notebook uses Python and common data-science / machine-learning
libraries, including:

-   pandas and NumPy for data handling
-   Matplotlib and Seaborn for visualization
-   scikit-learn for preprocessing, feature selection, regression
    models, and evaluation
-   Google Colab upload utilities when running in Colab
-   Python utilities for saving outputs and the trained model

## How to Run

### Option 1: Google Colab

1.  Upload `Used_Cars_Regression_Analysis_Colab.ipynb` to Google Colab.
2.  Click **Runtime → Run all**.
3.  When prompted, upload the dataset file `car_dataset.csv`.
4.  Allow the notebook cells to run in order and inspect the generated
    tables, plots, and evaluation results.

### Option 2: Jupyter Notebook

1.  Install Python and Jupyter Notebook.
2.  Place the notebook and `car_dataset.csv` in an accessible working
    directory.
3.  Open the notebook and run the cells from top to bottom.
4.  If a cell uses a Google Colab-only upload widget, use the notebook's
    local-file fallback or adapt that cell to load the CSV from the
    working directory.

### Dependencies

Install the main libraries if they are not already available:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

The notebook may use additional standard-library or environment-specific
modules. Run cells in order and install any missing dependency reported
by the environment.

## Main Findings Described in the Notebook

-   The data-cleaning workflow retains most listings while addressing
    missing body types, duplicate advertisements, logically impossible
    records, and extreme values.
-   Feature engineering adds vehicle age and usage-intensity information
    and represents multi-label fuel systems using multi-hot encoding.
-   The notebook reports that ensemble tree methods outperform the
    linear models on this dataset, suggesting that used-car prices
    depend on non-linear relationships and interactions.
-   Vehicle age, engine capacity, mileage, and brand/model market
    frequency are described as important price drivers.
-   The notebook applies a log transformation to price to reduce the
    effect of its right-skewed distribution.

Exact evaluation scores should be taken from the results produced when
the notebook is run with the dataset; this README does not invent metric
values.

## Limitations

-   The dataset contains only around 1,200 listings, so rare vehicle
    categories are represented by relatively few examples.
-   Listings are imbalanced by brand, with Toyota accounting for a large
    share of the data.
-   The target is the advertised asking price, not the final negotiated
    transaction price.
-   The dataset does not include important condition-related details
    such as accident history, service records, paint/interior condition,
    or number of previous owners.
-   The listings represent a snapshot in time; market changes may reduce
    prediction accuracy.
-   Outlier capping compresses extreme values, including some genuine
    high-end listings.

## Intended Use

This project is an academic valuation aid for exploring advertised
used-car prices. Predictions should not be treated as professional
appraisals or guaranteed sale prices.

## Project Files

-   `Used_Cars_Regression_Analysis_Colab.ipynb` --- analysis notebook.
-   `car_dataset.csv` --- input dataset required to run the notebook.
-   `README.md` --- project overview and setup instructions.

## Academic Note

This README summarizes the workflow and findings described in the
notebook. Dataset details and model performance depend on the input CSV
and the actual execution results.
