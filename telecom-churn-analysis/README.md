Customer Churn Prediction.

OBJECTIVE:<br/>
The goal of this project is to predict customer churn in the Indian telecom sector using demographic and usage data.<br/>
We will explore data science techniques to build predictive models and determine which factors are most influential in customer churn.<br/> 
Additionally, we will compare the performance of two encoding methods – OneHotEncoder and OrdinalEncoder – and evaluate their effect on model performance.<br/><br/>


Scenario

The telecommunications sector in India is rapidly evolving, with many businesses being created and customers frequently switching between providers.<br/> 
Churn refers to the process where customers stop using a company’s services or products. It is a key challenge for telecom companies to predict<br/>
and minimize customer churn, as retaining customers is critical to business growth. As a data scientist for a telecom company, our task is to predict<br/>
customer churn using demographic and usage data from four major telecom partners: Airtel, Reliance Jio, Vodafone, and BSNL.<br/>
We will explore the key factors contributing to customer churn and build predictive models to help the company reduce churn rates.<br/>


Datasets
- telecom_demographics.csv: Contains customer demographic infor-
mation.
- telecom_usage.csv: Contains customer usage patterns and churn
information.


STEPS<br/>

Step 1: Data Preprocessing<br/>

1. Data Merging:<br/>
- Merge the telecom_demographics.csv and telecom_usage.csv datasets using the customer_id column.<br/>

2. Feature Engineering:<br/>
- Drop unnecessary columns such as customer_id, pincode, and registration_event since they are unlikely to be useful for prediction.<br/>
- Handle missing values if any.<br/>
    
3. Encoding Categorical Variables:<br/>
- Compare the use of OneHotEncoder and OrdinalEncoder for encoding categorical variables like telecom_partner, city, state, gender.<br/>
- Train the models using both methods and evaluate the impact on model performance.<br/>
  
4. Scaling Numerical Features:<br/>
- Use StandardScaler to scale numerical features like calls_made, age, estimated_salary, sms_sent, etc.<br/>

Step 2: Model Building<br/>

1. Decision Tree Model:<br/>
- Train a Decision Tree model to predict customer churn using the preprocessed data.<br/>
- Compare the performance of the Decision Tree model for different values of the max_depth parameter (e.g., 3, 5, 10).<br/>
Evaluate model performance in terms of accuracy and analyze how the depth of the tree affects overfitting or underfitting.<br/>
- Visualize the first 4 levels of the trained decision tree using plot_tree.<br/>
    
2. Random Forest Model:<br/>
- Train a Random Forest model for churn prediction.<br/>
- Evaluate the performance of the model using accuracy.<br/>

Step 3: Feature Importance Analysis<br/>

1. Feature Importance in Decision Tree:<br/>
- Use the feature_importances_ attribute of the Decision Tree (when OrdinalEncoder is used) to identify the most important<br/>
features contributing to customer churn.<br/>
- Visualize the feature importance using a bar chart.<br/>
