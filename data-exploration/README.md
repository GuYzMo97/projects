Exploring the Data.

OBJECTIVE:<br/>
The purpose of this task is to perform a comprehensive data quality<br/>
assessment and exploration of a given dataset. You will generate key<br/>
visualizations and reports to understand the data distribution, detect<br/>
patterns, and address any data quality issues, such as missing or incorrect<br/>
data.<br/><br/>


STEPS<br/>
1. Load the Dataset:<br/>
- Load the dataset from the file listings_new_york_2024.csv1.<br/>
- Do preliminary data exploration:<br/>
    1. Identify number of rows and columns.<br/>
    2. Display data types of each column and convert data types if<br/>
    necessary (e.g., dates, categorical variables).<br/>
    3. Generate summary statistics (mean, median, mode, etc.) for<br/>
    numerical columns.<br/>
    4. Count unique values, find the first and the second mode, the<br/>
    frequency of the first and the second mode for categorical<br/>
    columns.<br/><br/>
    
2. Data Quality Report<br/>
- Missing Data Analysis:<br/>
    – Identify columns with missing values and the percentage of<br/>
  missing data in each column.<br/>
- Incorrect Data Detection:<br/>
    – Detect potential outliers or incorrect data entries.<br/>
    
3. Handling Incorrect Data<br/>
- Identify and handle outliers or incorrect data entries.<br/>
- Use domain knowledge to filter or replace incorrect values where<br/>
  necessary.<br/>
- Explain any assumptions made and the process for correcting these<br/>
  errors.<br/>
  
4. Dealing with Missing Data<br/>
- Apply strategies to handle missing data:<br/>
    – Remove rows or columns with a high percentage of missing data.<br/>
    – Impute missing values using mean/median (for numerical data)<br/>
    or mode (for categorical data).<br/>
- Document your approach and reasoning behind handling missing<br/>
  data.<br/><br/>
  
5. Data Exploration<br/>
- Histograms:<br/>
    – Create histograms for numerical columns to understand the data<br/>
    distribution (normal, skewed, etc.)<br/>
- Bar Plots:<br/>
    – Generate bar plots for categorical columns to examine the<br/>
    distribution of categories.<br/>
- Scatter Plots:<br/>
    – Create scatter plots to explore relationships between pairs of<br/>
    numerical columns.<br/>
- Correlation Matrix:<br/>
    – Calculate the correlation matrix for numerical variables.<br/>
    – Visualize the correlation matrix using a heatmap to identify<br/>
    highly correlated variables.<br/><br/>
    
6. Final Data Summary<br/>
- Provide a summary of the cleaned dataset, including the final<br/>
  number of rows and columns, and a comparison with the original<br/>
  dataset.<br/>
- Comment on any transformations or imputations applied.<br/><br/>

  
DELIVERABLES<br/>
IPython notebook containing:<br/>
- A detailed data quality report for the original dataset.<br/>
- Categorization of all variables (features) in the dataset.<br/>
- Visualizations including histograms, bar plots, scatter plots, and a<br/>
  correlation matrix heatmap.<br/>
- A data quality plan, outlining strategies for handling missing and<br/>
  incorrect data.<br/><br/>

Answers to the following questions:<br/>
  1. What is the distribution of property prices across different<br/>
  neighborhoods, and are there significant differences between them?<br/>
  2. How does the room type (Entire home/apt, Private room, etc.)<br/>
  affect the price? Are certain room types consistently more expensive?<br/>
  3. What is the correlation between the number of reviews and the<br/>
  availability of listings (availability_365)? Do listings with more<br/>
  reviews tend to be less available?<br/>
  4. Are there any outliers in the price or minimum night stays? How<br/>
  do they compare to typical listings?<br/>
  5. How do hosts with multiple listings compare to those with a<br/>
  single listing in terms of reviews, pricing, and availability?<br/>


  ### This file is taken from the : Inside Airbnb dataset, New York City, United States, file date: July 5, 2024.<br/>
