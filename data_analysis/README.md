# Trader Performance vs Market Sentiment Analysis  
Primetrade.ai – Data Science Intern Assignment  

## Objective  
Analyze how Bitcoin market sentiment (Fear/Greed Index) relates to trader behavior and performance on Hyperliquid, and derive actionable regime-based trading insights.

---

## Data Preparation  

Two datasets were used:  

1. Bitcoin Fear & Greed Index (Date, Classification)  
2. Hyperliquid Historical Trader Data (Account, Coin, Execution Price, Size, Side, Timestamp, Closed PnL, Fee, etc.)

Steps performed:

- Converted trader timestamps to datetime
- Converted sentiment UNIX timestamps to datetime
- Extracted daily-level date
- Checked missing values and duplicates
- Aggregated trader metrics at daily level
- Merged trader data with sentiment data using date alignment

---

## Feature Engineering  

Daily trader-level metrics created:

- Daily PnL per account  
- Win rate  
- Average trade size (USD)  
- Trade frequency  
- Long/Short ratio  

Market-level metrics:

- Average PnL by sentiment regime  
- Win rate by regime  
- PnL volatility by regime  

---

## Key Findings  

1. Performance is regime-dependent:  
   Traders show higher average PnL during Greed regimes, while Extreme Fear regimes exhibit higher PnL dispersion and volatility.

2. Behavioral shifts occur across regimes:  
   Trade frequency increases during Fear periods, while position sizes and long bias increase during Greed periods.

3. Risk concentration:  
   High leverage traders experience significantly larger drawdowns during Fear regimes compared to low leverage traders.

---

## Segmentation Analysis  

- High vs Low Leverage Traders:  
  High leverage traders are more vulnerable during Fear regimes.

- Frequent vs Infrequent Traders:  
  Frequent traders perform better during Greed regimes but are more exposed during Fear.

- Consistent vs Inconsistent Traders:  
  Consistent traders maintain stable expectancy across regimes.

---

## Strategy Recommendations  

1. Regime-Based Leverage Adjustment  
   Reduce leverage during Extreme Fear regimes (30–50%).  
   Allow controlled leverage expansion during Greed regimes.

2. Segment-Specific Risk Control  
   During Fear regimes, frequent and high-leverage traders should reduce exposure.  
   During Greed regimes, consistent traders can increase trade frequency.

---

## Reproducibility  
git clone https://github.com/igxd23/projects.git
cd projects/data_analysis
Install dependencies:  pip install -r requirements.txt
Run notebook:Open `main.ipynb` and run all cells.

---

Author: Suneet Datta 
Submission for Primetrade.ai – Junior Data Scientist
