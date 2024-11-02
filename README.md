# Electric Vehicle Population Analysis in the US

## Table of Contents

- [Project Description](#project-description)
- [Repository Contents](#repository-contents)
- [Running the Code](#running-the-code)
- [Data Cleaning and Normalization](#data-cleaning-and-normalization)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Visualization](#data-visualization)


## Project Description

This project analyzes the electric vehicle population in Washington State using data from the Washington State Department of Licensing (DOL). The dataset includes information about Battery Electric Vehicles (BEVs) and Plug-in Hybrid Electric Vehicles (PHEVs) that are currently registered.

**Dataset Description:**

This dataset shows the Battery Electric Vehicles (BEVs) and Plug-in Hybrid Electric Vehicles (PHEVs) that are currently registered through Washington State Department of Licensing (DOL).

**Project Goals:**

The main goals of this project are:

- To clean, process, and normalize the electric vehicle registration data.
- To conduct exploratory data analysis (EDA) to uncover insights and trends in the data, and create relevant visualizations.
- To identify factors that influence electric vehicle adoption in Washington state.
- To investigate the distribution of electric vehicles across different regions and demographics.
- To predict and forecast future electric vehicle adoption trends.

## Repository Contents:

- `data/Electric_Vehicle_Population_Data.csv` - The raw electric vehicle dataset.
- `Electric-Vehicle-Population-Data.ipynb.ipynb` - Jupyter Notebook containing the cleaning, data analysis, and visualization code.
- `report/1210190_1210145_ML Assignment_1.pdf` - A technical report that describes the conclusions in more detail.
- `Electric-Vehicle-Population-Data.py` - A Python script version of the notebook for broader compatibility.


## Running the Code:

To run the analysis, open the `Electric-Vehicle-Population-Data.ipynb` file in Google Colab or a Jupyter Notebook environment. Ensure you have the following libraries installed:

| Library       | Version     | Installation Command                             |
| ------------- | ---------- | ------------------------------------------------ |
| pandas        | 2.0.3      | `!pip install pandas`                     |
| matplotlib    | 3.7.1      | `!pip install matplotlib`                 |
| seaborn       | 0.12.2     | `!pip install seaborn`                    |
| geopandas     | 0.13.2     | `!pip install geopandas`                  |
| contextily    | 1.3.1     | `!pip install contextily`                  |

You can also run the provided code cells that use pip to install the dependencies, they might not be compatabile with your chosen environment, however.


## Data Cleaning and Normalization

The data cleaning and normalization process includes the following steps:

- Handling missing values: Imputing with mode or median depending on the column.
- Encoding categorical features: Using one-hot or label encoding based on cardinality.
- Normalization: standardizing numeric data with z-score.

## Exploratory Data Analysis (EDA)

EDA involves various descriptive statistics, visualizations, and correlational analysis to find patterns and insights in the data, including:

- Electric Range and distribution analysis.
- Top models, cities, and counties.
- Visualization of electric vehicle locations on a map (using geopandas).
- Trend analysis of electric vehicle adoption over the years.

## Data Visualization
- The project includes visualizations using Matplotlib, Seaborn and Geopandas to help understand the data.

## Contributors 

- [Yazan AbuAloun](https://github.com/yazan6546)
- [Ahamd Qaimari](https://github.com/ahmadq44)

