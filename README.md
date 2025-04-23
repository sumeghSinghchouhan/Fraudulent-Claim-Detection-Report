Fraudulent Claim Detection Report
Submitted by:
●	Sumegh Singh Chouhan
●	Sudarshan  Chavan
Executive Summary
This report presents a comprehensive analysis of Global Insure's insurance claims data to develop predictive models for identifying fraudulent claims. The company processes thousands of claims annually, with a significant percentage turning out to be fraudulent, resulting in considerable financial losses. The current manual inspection process is time-consuming and inefficient, often detecting fraud too late. To address this challenge, we developed and compared Logistic Regression and Random Forest models based on historical claim data to enable early fraud detection in the approval process.
Our models achieved validation accuracies of 75.33% (Logistic Regression) and 79.00% (Random Forest), demonstrating the feasibility of automating fraud detection. The Random Forest model performed better in identifying legitimate claims with 87.17% specificity, while the Logistic Regression model was better at detecting fraudulent claims with 62.16% sensitivity.
Key predictive features included incident severity, vehicle characteristics, and claimant profile factors. Based on our findings, we recommend implementing a two-tiered screening system that leverages both models' strengths to enhance fraud detection capabilities while optimizing claims processing.
Problem Statement
Global Insure, a leading insurance company, processes thousands of claims annually. However, a significant percentage of these claims turn out to be fraudulent, resulting in considerable financial losses. The company's current process for identifying fraudulent claims involves manual inspections, which is time-consuming and inefficient. Fraudulent claims are often detected too late in the process, after the company has already paid out significant amounts.
Business Objective
Global Insure wants to build a model to classify insurance claims as either fraudulent or legitimate based on historical claim details and customer profiles. By using features like claim amounts, customer profiles, and claim types, the company aims to predict which claims are likely to be fraudulent before they are approved.


Key questions to address:
●	How can we analyze historical claim data to detect patterns that indicate fraudulent claims?
●	Which features are most predictive of fraudulent behavior?
●	Can we predict the likelihood of fraud for an incoming claim based on past data?
●	What insights can be drawn from the model that can help in improving the fraud detection process?
Methodology
Our approach followed a structured data science methodology:
1.	Data Preparation: Loading and initial examination of the dataset
2.	Data Cleaning: Handling null values, removing redundant columns, fixing data types
3.	Train-Validation Split: 70-30 split with stratification on the target variable
4.	Exploratory Data Analysis: Univariate and bivariate analysis of training data
5.	Feature Engineering: Resampling, feature creation, handling categorical variables
6.	Model Building:
○	Logistic Regression with feature selection and cutoff optimization
○	Random Forest with hyperparameter tuning
7.	Model Evaluation: Comprehensive assessment of model performance on validation data
Data Description
The dataset contained 40 columns and 1000 rows of insurance claims information, including:
●	Customer details (age, gender, education, occupation)
●	Policy information (premium, state, coverage limits)
●	Incident details (type, severity, date/time, location)
●	Claim information (total amount, injury claim, property claim, vehicle claim)
●	Target variable: fraud_reported (Y/N)
Data Cleaning
Handling Null Values
We identified missing values in the 'authorities_contacted' column, which were replaced with the mode value. We also identified and dropped a completely empty column ('_c39').
Handling Redundant Values
We removed several columns with limited predictive value:
●	Columns with unique identifiers ('policy_number', 'incident_location')
●	Columns with too many unique values ('insured_zip', 'policy_bind_date')
●	Columns containing invalid values were cleaned ('property_damage', 'collision_type', 'police_report_available')
●	Rows with illogical values (negative 'umbrella_limit') were removed
Data Type Conversion
●	'incident_date' was converted to datetime format
●	Created new date-based features: 'incident_year', 'incident_month', 'incident_weekday'
●	Converted 'fraud_reported' to binary (1 for 'Y', 0 for 'N')
Exploratory Data Analysis
Class Balance
The dataset showed significant class imbalance with approximately 75% legitimate claims and 25% fraudulent claims.  

Univariate Analysis
Numerical Features:
●	Age distribution showed most customers were between 30-50 years
●	Total claim amount distribution was right-skewed, with most claims being smaller amounts
●	Strong correlation between total claim amount and its components (vehicle, property, and injury claims)
●	Unusual distribution in number of vehicles involved (high for 1 and 3, low for 2)
[INSERT VISUALIZATION: Histograms of key numerical features (age, total_claim_amount, etc.)]
Categorical Features:
●	Most common incident types were Single Vehicle Collision and Vehicle Theft
●	Incident severity was primarily Minor Damage and Major Damage
●	Most common authorities contacted were Police and None
Correlation Analysis
The correlation heatmap revealed:
●	High correlation between 'age' and 'months_as_customer' (0.92)
●	Strong correlation between 'total_claim_amount' and its components:
○	'vehicle_claim' (0.98)
○	'injury_claim' (0.81)
○	'property_claim' (0.81)
●	These correlations indicated potential multicollinearity issues for modeling
 
Bivariate Analysis
Categorical Features vs. Fraud:
●	Incident Severity: Major Damage showed dramatically higher fraud likelihood compared to other categories
[INSERT VISUALIZATION: Bar chart of fraud rate by incident severity]
●	Vehicle Make: Strong relationship with fraud, with Mercedes, Ford and Audi showing highest fraud rates 
●	Incident Type: Single Vehicle Collision and Multi-vehicle Collision had significantly higher fraud rates than Parked Car or Vehicle Theft
 




●	Insured Hobbies: Certain hobbies showed remarkably high fraud rates (chess, cross-fit)
 
●	Insured Occupation: Executive-managerial occupations showed the highest fraud rates
 
Numerical Features vs. Fraud:
●	Higher total claim amounts were associated with higher fraud likelihood
●	Age showed some relationship (younger customers more likely to commit fraud)
●	Policy deductible showed relationship (higher deductibles associated with higher fraud)
    
         
Feature Engineering
Resampling
Due to class imbalance, we applied RandomOverSampler to balance the training data, ensuring the model would learn effectively from both classes.
Feature Creation
●	Created age group bins ('18-30', '31-40', etc.)
●	Generated incident weekday numeric feature
●	Extracted policy_csl components (upper and lower limits)
Categorical Feature Processing
●	Combined infrequent categories in 'incident_type' and 'incident_state'
●	Encoded education levels and incident severity ordinally
●	Created dummy variables for all remaining categorical features
Feature Scaling
Applied StandardScaler to normalize numerical features, preventing features with larger magnitudes from dominating the models.
Model Building
Logistic Regression Model
Feature Selection: We used Recursive Feature Elimination with Cross-Validation (RFECV) to identify the most relevant features, selecting features that collectively provided the best predictive power.
Model Building and Multicollinearity Assessment:
●	Built logistic regression model with selected features
●	Analyzed VIFs to detect multicollinearity
●	Features with high VIFs were noted but retained for initial modeling
Cutoff Optimization:
●	Plotted ROC curve (AUC: 0.9114)
●	Analyzed sensitivity-specificity tradeoff at different cutoffs
●	Determined optimal cutoff: 0.5
●	Re-evaluated model performance with optimal cutoff 
Training Performance with Optimal Cutoff:
●	Accuracy: 82.41%
●	Sensitivity: 82.89%
●	Specificity: 81.94%
●	Precision: 82.11%
●	F1 Score: 0.8250
Random Forest Model
Feature Importance Analysis:
●	Built initial model and extracted feature importances
●	Selected features contributing to 95% of total importance
 
Cross-Validation Assessment:
●	Mean CV Score: 92.49%
●	Training Accuracy: 100.00%
●	Identified potential overfitting
Hyperparameter Tuning:
●	Used Grid Search to optimize hyperparameters
●	Best parameters: n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt'
Training Performance with Tuned Model:
●	Accuracy: 100.00%
●	Sensitivity: 100.00%
●	Specificity: 100.00%
●	Precision: 100.00%
●	F1 Score: 1.0000
Model Evaluation on Validation Data
Logistic Regression
Performance Metrics:
●	Accuracy: 75.33%
●	Sensitivity: 62.16%
●	Specificity: 79.65%
●	Precision: 50.00%
●	Recall: 62.16%
●	F1 Score: 0.5542
Confusion Matrix:
●	True Positives: 46
●	True Negatives: 180
●	False Positives: 46
●	False Negatives: 28
 

Random Forest
Performance Metrics:
●	Accuracy: 79.00%
●	Sensitivity: 54.05%
●	Specificity: 87.17%
●	Precision: 57.97%
●	Recall: 54.05%
●	F1 Score: 0.5594
Confusion Matrix:
●	True Positives: 40
●	True Negatives: 197
●	False Positives: 29
●	False Negatives: 34
 
Model Comparison
Metric	Logistic Regression	Random Forest
Validation Accuracy	75.33%	79.00%
Sensitivity	62.16%	54.05%
Specificity	79.65%	87.17%
Precision	50.00%	57.97%
F1 Score	0.5542	0.5594
AUC	0.9114	N/A

The Random Forest model demonstrated superior overall validation accuracy (79.00% vs 75.33%) but showed clear signs of overfitting with perfect (100%) training accuracy compared to its cross-validation score of 92.49%.
In terms of fraud detection capability, the Logistic Regression model provided better sensitivity (62.16% vs 54.05%), meaning it detected more fraudulent claims, while the Random Forest model demonstrated superior specificity (87.17% vs 79.65%), correctly identifying more legitimate claims.
Key Findings
Most Predictive Features
Based on our modeling, the most influential features for fraud detection were:
1.	Claim Characteristics:
○	Incident severity: Major Damage claims showed much higher fraud likelihood
○	Total claim amount and its components
○	Number of vehicles involved
2.	Vehicle Information:
○	Vehicle make: Luxury brands showed significantly higher fraud rates
○	Vehicle model: Certain models had very high fraud rates
3.	Claimant Profile:
○	Insured's hobbies: Chess and cross-fit showed extremely high fraud association
○	Insured's occupation: Executive-managerial positions had highest fraud rates
○	Insured's relationship: Other-relative and wife had higher fraud rates
4.	Context Factors:
○	Weekday: Monday and Saturday had highest fraud rates
○	Authorities contacted: "Other" authorities and Ambulance vs. Police
○	Location: Ohio and Arlington city showed highest fraud rates
○	Collision type: Rear collisions showed higher fraud rates
Fraud Patterns
Our analysis revealed several patterns indicative of potentially fraudulent claims:
1.	High-Risk Combinations:
○	Major damage claims for luxury vehicles
○	Claims filed on Mondays with certain characteristics
○	High-risk hobbies combined with certain occupations
2.	Documentation Red Flags:
○	Claims where property damage or police report status was "unknown"
○	Cases where police were not involved but other authorities were
○	Claims where documentation is incomplete
3.	Geographic Hotspots:
○	Ohio claims showed much higher fraud rates than other states
○	Arlington and Columbus showed significantly higher fraud rates than other cities
Recommendations
Based on our findings, we recommend the following actions for Global Insure:
1. Implement a Two-Tiered Screening System
●	First Tier: Use the logistic regression model as an initial screening tool for all incoming claims
●	Second Tier: Claims identified as high-risk should undergo additional review using the Random Forest model
2. Focus on High-Risk Indicators
Claims with the following characteristics deserve increased scrutiny:
●	Major Damage claims - especially for luxury vehicles
●	Claims involving Mercedes, Ford or Audi vehicles
●	Claims filed by individuals with chess or cross-fit listed as hobbies
●	Claims from executive-managerial occupations
●	Claims filed on Mondays and Saturdays
●	Claims from Ohio and Arlington
●	Claims involving rear-end collisions
●	Claims where police weren't contacted but other authorities were
3. Process Improvements
●	Implement automated flagging for claims with high-risk characteristics
●	Create standardized documentation requirements for high-risk scenarios
●	Develop specialized training for adjusters handling potentially fraudulent claims
●	Set different approval thresholds for claims based on risk score
4. Model Deployment Strategy
●	Use combined model approach for prediction (leveraging strengths of both models)
●	Set appropriate threshold (around 0.5) to balance false positives and negatives
●	Regularly retrain the model (at least quarterly) with new data
●	Track model performance over time to ensure continued effectiveness
Assumptions
Throughout this analysis, several assumptions were made that could influence the interpretation of our results:
1.	Data Completeness and Accuracy: We assumed the historical claims data provided is representative of the general claims population and contain sufficiently accurate information. Missing values in 'authorities_contacted' were replaced with the mode, assuming this was the most likely value.
2.	Fraud Identification: We assumed that the 'fraud reported' flag in the dataset accurately represents actual fraud cases without significant false positives or negatives in the historical labeling process.
3.	Feature Independence: While we identified and noted multicollinearity between certain features (age and months_as_customer; total_claim_amount and its components), we retained these features in modeling, assuming their collective predictive power outweighed the statistical concerns of correlation.
4.	Class Imbalance Handling: We assumed that random oversampling of the minority class would create a balanced dataset without introducing significant bias, effectively addressing the 75/25 imbalance between legitimate and fraudulent claims.
5.	Equalized Misclassification Costs: Our model optimization assumed that false positives (legitimate claims classified as fraudulent) and false negatives (fraudulent claims classified as legitimate) carry similar business costs, which may not always be the case in practice.
6.	Temporal Stability: We assumed that the patterns of fraudulent behavior identified in historical data will remain relatively stable over time, though fraudsters may adapt their methods in response to detection systems.
7.	Generalizability: When selecting our optimal cutoff threshold (0.5), we assumed that the performance on our validation set would generalize to future unseen claims with similar characteristics.
Understanding these assumptions is important for interpreting the model's predictions and recommendations in real-world deployment scenarios. Should any of these assumptions prove invalid over time, the model and strategy may need to be adjusted accordingly.
Limitations and Future Work
1.	Model Refinement:
○	Address overfitting in the Random Forest model
○	Explore additional classification algorithms (XGBoost, Neural Networks)
○	Implement cost-sensitive learning to account for different misclassification costs
2.	Feature Engineering:
○	Develop more interaction terms between key features
○	Incorporate temporal patterns and seasonal factors
○	Consider external data sources (weather, crime statistics, etc.)
3.	Business Process Integration:
○	Create a real-time scoring system for incoming claims
○	Develop a feedback loop for continuously improving model performance
○	Design user-friendly dashboards for claims adjusters
Conclusion
Our analysis demonstrates that machine learning models can effectively identify patterns indicative of fraudulent insurance claims, providing Global Insure with a powerful tool to enhance their fraud detection capabilities. The complementary strengths of the Logistic Regression and Random Forest models suggest that a combined approach would yield the best results, with the former excelling at detecting fraudulent claims while the latter more accurately identifies legitimate claims.
By implementing our recommended two-tiered screening system and process improvements, Global Insure can significantly reduce financial losses associated with fraudulent claims while streamlining the claims handling process for legitimate customers. Continuous model refinement and integration with business processes will further enhance the effectiveness of the fraud detection system over time.

