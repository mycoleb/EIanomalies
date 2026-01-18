# Canada EI Claims Anomaly Detection

A Python tool for detecting statistical anomalies in Canadian Employment Insurance (EI) claims data using machine learning. This project downloads official Statistics Canada data, processes it by province/territory, and identifies unusual patterns that may warrant further investigation.

## Overview

This tool uses **Isolation Forest**, an unsupervised machine learning algorithm, to flag statistically unusual months in EI claims across Canadian provinces and territories. It's designed to help identify potential data quality issues, policy impacts, or unusual economic events.

### Key Features

- **Automated data retrieval** from Statistics Canada's official data tables
- **Province-by-province analysis** with separate anomaly detection models
- **Feature engineering** including rolling statistics, month-over-month changes, and z-scores
- **Visual outputs** showing time series with highlighted anomalies
- **CSV export** for further analysis and auditing

## What the Analysis Shows

The visualizations reveal several notable anomalies in Canada's EI claims history:

- **2020-2021 COVID-19 Pandemic**: Massive spike in claims during October-November 2020, followed by elevated claims through early 2021
- **1958-1959 Recession**: Early anomalies during economic downturn
- **Historical patterns**: The model successfully identifies known economic shocks and policy changes

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
- pandas (data manipulation)
- numpy (numerical computing)
- requests (HTTP downloads)
- scikit-learn (machine learning)
- matplotlib (visualization)
- python-dateutil (date parsing)

## Usage

### Basic Usage

Run the script with default settings (analyzes Canada-wide data):

```bash
python p.py
```

This will:
1. Download the latest Statistics Canada EI claims data (Table 14-10-0005-01)
2. Process and clean the data
3. Run anomaly detection for all regions
4. Generate `ei_anomaly.png` (visualization for Canada)
5. Export `ei_anomaly_flags.csv` (full dataset with anomaly flags)

### Analyze Specific Provinces

To analyze a specific province or territory:

```bash
# Ontario
python p.py --region Ontario --out ontario_ei_anomaly.png

# Quebec
python p.py --region Quebec --out quebec_ei_anomaly.png

# British Columbia
python p.py --region "British Columbia" --out bc_ei_anomaly.png
```

### Advanced Options

Adjust the sensitivity of anomaly detection:

```bash
# More sensitive (flag more anomalies)
python p.py --contamination 0.05 --region Ontario

# Less sensitive (flag fewer anomalies)
python p.py --contamination 0.02 --region Ontario
```

The `--contamination` parameter represents the expected proportion of anomalies (default: 0.03 or 3%).

### Command-Line Arguments

```
--zip-id         Statistics Canada table ZIP ID (default: 14100005)
--lang           Language: 'eng' or 'fra' (default: eng)
--region         Province/territory to plot (default: Canada)
--contamination  Expected anomaly fraction, e.g., 0.02-0.08 (default: 0.03)
--out            Output image path (default: ei_anomaly.png)
--export-csv     CSV export path (default: ei_anomaly_flags.csv)
```

## How It Works

### 1. Data Collection

The script downloads official Statistics Canada data using their full-table CSV ZIP download pattern:
- **Data Source**: Table 14-10-0005-01 (Employment Insurance claims received by province/territory, monthly)
- **Format**: Seasonally adjusted monthly data
- **Coverage**: Historical data from 1940s to present

### 2. Feature Engineering

For each province/territory time series, the script calculates:
- **Raw claim values**: Actual monthly claims
- **First differences**: Month-over-month change
- **Percent changes**: Relative change from previous month
- **Rolling statistics**: 6-month rolling mean and standard deviation
- **Z-scores**: Standardized deviation from rolling mean
- **Month indicator**: Seasonal component (1-12)

### 3. Anomaly Detection

**Isolation Forest Algorithm**:
- Unsupervised learning method ideal for outlier detection
- Works by isolating observations through random partitioning
- Anomalies are points that are easier to isolate (require fewer splits)
- No labeled training data required

**Model Configuration**:
- 300 decision trees for robust detection
- Separate model per region (adapts to local patterns)
- Minimum 36 data points required per region
- Configurable contamination parameter

### 4. Visualization

Generated plots show:
- Complete time series of EI claims
- Anomalous months marked with blue dots
- Most extreme anomalies labeled with dates
- Clear identification of major economic events

## Output Files

### 1. Visualization (PNG)

Time series plot showing:
- Monthly EI claims over time
- Highlighted anomalies
- Labeled extreme outliers
- 200 DPI resolution for publication quality

### 2. Anomaly Flags (CSV)

Structured dataset with columns:
- `ref_date`: Month of observation
- `geo`: Province/territory name
- `value`: EI claims (seasonally adjusted)
- `anomaly`: Boolean flag (True = anomaly detected)
- `anomaly_score`: Continuous score (lower = more anomalous)

Use this CSV for:
- Custom analysis and visualization
- Integration with other datasets
- Statistical auditing
- Policy research

## Interpreting Results

### Understanding Anomalies

An anomaly is **not necessarily fraud or error**. Anomalies can indicate:

✅ **Legitimate unusual events**:
- Economic recessions or booms
- Natural disasters
- Policy changes
- Pandemic impacts

⚠️ **Data quality issues**:
- Reporting errors
- Collection method changes
- Administrative adjustments

🔍 **Worth investigating**:
- Unexpected regional patterns
- Outliers without obvious explanation
- Persistent anomalies over multiple months

### Anomaly Score Interpretation

- **Lower scores** (more negative): More anomalous
- **Higher scores**: More normal/typical
- The model uses these scores to classify points as anomalies or normal

### Known Limitations

1. **Context-free detection**: The algorithm doesn't know about COVID-19, recessions, or policy changes
2. **Equal treatment**: All anomalies flagged equally, regardless of magnitude or context
3. **Seasonal adjustment**: Data is pre-adjusted by Statistics Canada
4. **Historical bias**: Very old data may have different collection standards

## Use Cases

### For Researchers
- Identify periods requiring deeper economic analysis
- Compare provincial responses to national events
- Study labor market shocks and recovery patterns

### For Data Analysts
- Data quality validation for EI datasets
- Automated monitoring of new monthly releases
- Historical pattern analysis

### For Policy Makers
- Understand impact of policy changes on claims
- Identify regional disparities in economic shocks
- Monitor program utilization patterns

### For Students
- Learn practical anomaly detection techniques
- Understand real-world time series analysis
- Explore Canadian labor market data

## Technical Details

### Algorithm Choice: Why Isolation Forest?

- **No labeled data required**: We don't have pre-labeled "fraud" or "error" examples
- **Efficient**: Fast training and prediction even with decades of data
- **Robust**: Handles multivariate features without assuming data distribution
- **Interpretable**: Anomaly scores provide continuous measure of unusualness

### Feature Selection Rationale

The features capture different aspects of anomalies:
- **Absolute level** (value): Catches magnitude outliers
- **Changes** (diff, pct_change): Catches sudden shifts
- **Context** (z-scores): Catches deviations from recent trends
- **Seasonality** (month): Accounts for regular patterns

## Data Source

**Statistics Canada Table 14-10-0005-01**
- Official title: "Employment insurance beneficiaries receiving regular benefits by province and territory, monthly, unadjusted for seasonality"
- Publisher: Statistics Canada
- License: Open Government License - Canada
- Update frequency: Monthly
- URL: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1410000501

## Contributing

Suggestions for improvement:
- Additional anomaly detection algorithms (LOF, DBSCAN)
- Interactive visualizations (Plotly, Bokeh)
- Time series forecasting
- Integration with other economic indicators
- Web dashboard interface

## License

This code is provided as-is for educational and research purposes. The Statistics Canada data is covered under the Open Government License - Canada.

## Disclaimer

This tool provides statistical analysis only and does not constitute:
- Proof of fraud or misconduct
- Official government audit results
- Legal or policy recommendations

Always verify anomalies against official sources and domain expertise before drawing conclusions.

## Contact & Support

For questions about:
- **The data**: Contact Statistics Canada
- **The methodology**: Review scikit-learn Isolation Forest documentation
- **This implementation**: Review the source code comments in `p.py`

---

*Generated visualizations show historical EI claims patterns with COVID-19 pandemic surge clearly visible in 2020-2021 across all provinces.*
