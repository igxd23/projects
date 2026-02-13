Methodology

1. Data Cleaning and Standardization
	•	Converted Timestamp IST to datetime format.
	•	Converted Fear/Greed UNIX timestamps to datetime.
	•	Extracted daily-level date from both datasets.
	•	Checked and handled missing values and duplicates.
	•	Standardized column names for consistency.

2. Data Alignment
	•	Aggregated trader data at daily frequency.
	•	Merged trader dataset with sentiment dataset using date as the join key.
	•	Ensured each trade record was mapped to its corresponding daily sentiment regime.

3. Feature Engineering

Daily trader-level metrics:
	•	Total daily PnL per account
	•	Win rate (profitable trades / total trades)
	•	Average trade size (USD)
	•	Trade count per day
	•	Long/Short ratio

Market-level aggregations:
	•	Average PnL by sentiment regime
	•	Win rate by regime
	•	PnL volatility (standard deviation) by regime

4. Regime Comparison
	•	Compared distributions of PnL across Fear vs Greed regimes.
	•	Evaluated behavioral shifts (frequency, size, bias) across sentiment classes.
	•	Analyzed drawdown proxy using negative PnL dispersion.

5. Segmentation

Created behavioral segments:
	•	High vs Low leverage traders
	•	Frequent vs Infrequent traders
	•	Consistent vs Inconsistent traders

Segment-level performance was analyzed across sentiment regimes to identify interaction effects.
