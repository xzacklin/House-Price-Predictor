# Housing Market Price Predictor

This notebook explores the relationship between macroeconomic factors (like interest rates, CPI, and rental vacancy rates) and inflation-adjusted housing prices. The goal is to predict whether prices will go up or down in the following quarter using a Random Forest classifier.

---

## Data Sources

- **Zillow**: Weekly median sale prices & home value index across U.S. metros
- **FRED (Federal Reserve)**:
  - 30-year mortgage rates (`MORTGAGE30US`)
  - Rental vacancy rates (`RRVRUSQ156N`)
  - Consumer Price Index (`CPIAUCSL`)

---

## What It Does

- Merges housing and economic time series into one clean DataFrame
- Adjusts housing prices for inflation (CPI-adjusted values)
- Engineers a binary target: 1 if inflation-adjusted price is higher next quarter, 0 if not
- Trains a Random Forest model to predict that change
- Backtests the model with rolling validation
- Visualizes prediction accuracy and feature importances

---

##  Key Features

- CPI-adjusted prices and values
- Lagged target (`next_quarter`) with `.shift()`
- Rolling mean ratios for feature normalization
- `RandomForestClassifier` for binary classification
- Accuracy tracking and prediction color-coding
- Feature importance via permutation analysis

---

## File Structure

- `*.csv`: Zillow and FRED CSVs (expect local files like `MORTGAGE30US.csv`, `CPIAUCSL.csv`, etc.)
- `housing_predictor.ipynb`: Jupyter Notebook with full workflow

---

## Accuracy (as tested)

- Accuracy using raw predictors: ~59%
- Accuracy with rolling normalized features: ~64.7%

---

## Sample Visuals

- Scatter plot of predictions (`green` for correct, `red` for incorrect)
- Bar chart of feature importances based on permutation score

---

## Getting Started

To run this project locally:

1. Clone the repo and ensure required CSV files are in the same folder
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open the notebook and run all cells

---

## Requirements

- pandas
- numpy
- matplotlib
- scikit-learn

Install with:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## 📄 License

MIT – use, modify, or build on it.

---
