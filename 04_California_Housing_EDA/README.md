# California Housing Exploratory Data Analysis (EDA)

This project provides a comprehensive exploratory data analysis of the California housing dataset, focusing on understanding housing price patterns and relationships between various features.

## Project Overview

This analysis explores the California housing dataset to uncover insights about housing prices, geographical distributions, and feature relationships that influence property values.

## Dataset

The California housing dataset contains information about housing districts in California, including:

- Longitude: A measure of how far west a house is
- Latitude: A measure of how far north a house is
- Housing Median Age: Median age of a house within a block
- Total Rooms: Total number of rooms within a block
- Total Bedrooms: Total number of bedrooms within a block
- Population: Total number of people residing within a block
- Households: Total number of households within a block
- Median Income: Median income for households within a block
- Median House Value: Median house value for households within a block (target variable)
- Ocean Proximity: Location of the house w.r.t ocean/sea

## Files Structure

- `California_Housing_EDA.ipynb`: Complete exploratory data analysis notebook

## Analysis Features

- **Data Overview**: Dataset shape, info, and basic statistics
- **Missing Data Analysis**: Identification and handling of missing values
- **Univariate Analysis**: Distribution of individual features
- **Bivariate Analysis**: Relationships between features
- **Geographical Analysis**: Spatial distribution of housing prices
- **Correlation Analysis**: Feature correlation matrix and heatmaps
- **Data Visualization**: Various plots including histograms, scatter plots, and maps
- **Feature Engineering**: Creation of new meaningful features
- **Outlier Detection**: Identification of unusual data points

## Key Insights

The analysis reveals:
- Geographical patterns in housing prices
- Correlation between median income and house values
- Impact of proximity to ocean on property values
- Age distribution of housing stock
- Population density effects on housing markets

## Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required packages: pandas, numpy, matplotlib, seaborn, plotly (optional)

### Installation

1. Install dependencies:
   ```
   pip install pandas numpy matplotlib seaborn plotly jupyter
   ```

2. Launch Jupyter Notebook:
   ```
   jupyter notebook California_Housing_EDA.ipynb
   ```

## Usage

1. Open the Jupyter notebook
2. Run cells sequentially to see the complete EDA
3. Explore interactive visualizations
4. Modify analysis parameters to investigate specific aspects
5. Use insights for feature selection in predictive modeling
