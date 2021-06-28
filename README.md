# Credit Risk Analysis


## Overview Of Analysis
The purpose of this analysis is to test multiple machine learning models to determine which one is best suited to identify the risk levels for individual credit applicants. These models were fed more than 80 data points per applicant including demographics, credit history, home ownership, and more, in order to create reliable models to predict whether applicants were likely to default on their credit. Due to the lower prevalence of High-Risk applicants as compared to Low-Risk applicants, these models seek to balance the reprensentation of High-Risk applicants in the training data pool. This is done by increasing sampling from the High-Risk class, reducing sampling from the Low-Risk class or a combination of the two. 

## Results
- Model 1: Naive Random Oversampling
![ros_accuracy](https://github.com/ghynox/credit_risk_analysis/blob/main/ros_accuracy_correct.png)
![ros_classreport](https://github.com/ghynox/credit_risk_analysis/blob/main/ros_classreport_correct.png)
The Naive Random Sampling model shows lackluster accuracy (0.6), Precision scores (0.01/1.0), and F1 Scores (0.02/0.77) giving the overall expression that this model is unable to correctly identify the high risk applicants from the low risk applicants. 

- Model 2: SMOTE Oversampling
![smote_accuracy](https://github.com/ghynox/credit_risk_analysis/blob/main/smote_accuracy_correct.png)
![smote_classreport](https://github.com/ghynox/credit_risk_analysis/blob/main/smote_classreport_correct.png)
Using SMOTE oversampling, we still see very poor findings for Precision (0.01 and 1.0), Recall (0.67), and F1 Score (0.02) for high risk applicants. This model shows difficulty identifying high risk applicants much the same as the Random Oversampling model. 

- Model 3: Clustered Centroid Undersampling
![cc_accuracy](https://github.com/ghynox/credit_risk_analysis/blob/main/cc_accuracy_correct.png)
![cc_classreport](https://github.com/ghynox/credit_risk_analysis/blob/main/cc_classreport_correct.png)
Looking at the undersampling model, we lowered our accuracy to 0.54 and F1 Score of high risk applicants to 0.01. Across the board, the undersampling model shows a decrease in the ability to identify high risk applicants and with only 246 applicants used for training the model, it seems likely that it will be the worst model of this set. 

- Model 4: SMOTEEN Over/Under Sampling
![smoteen_accuracy](https://github.com/ghynox/credit_risk_analysis/blob/main/smoteenn_accuracy.png)
![smoteen_classreport](https://github.com/ghynox/credit_risk_analysis/blob/main/smoteenn_classreport.png)
Given the Accuracy (0.66), Precision (0.01/1.0), Recall (0.73/0.62), and F1 Scores (0.02/0.77) for the SMOTEENN model, we can see that we have a poor ability to differentiate the high and low risk classes.

- Model 5: Balanced Random Forest Classifier
![brfc_accuracy](https://github.com/ghynox/credit_risk_analysis/blob/main/brfc_accuracy.png)
![brfc_classreport](https://github.com/ghynox/credit_risk_analysis/blob/main/brfc_clasreport.png)
With an accuracy of 1.0, Recalls of 0.8/1.0 and Preceisions of 0.8/1.0, on top of F1 scores of 0.89/1.0, the balanced random forest classifier model is the most reliable model seen so far. This model is capable of idenifying and predicting high and low risk applicants with a high confidence. 

- Model 6: Easy Ensemble Classifier
![eec_accuracy](https://github.com/ghynox/credit_risk_analysis/blob/main/eec_accuracy_correct.png)
![eec_classreport](https://github.com/ghynox/credit_risk_analysis/blob/main/eec_classreport_correct.png)
Easy Ensemble Classification easily takes the cake as the most accurate and reliable model given this specific set of data with 1.0 for all measurement values. This 

## Summary
Given the analysis of each of the six models, it is easy to see that the Balanced Random Forest and Easy Ensemble Classifiers had the greatest effect in forming models that would identify high and low risk credit applicants. The model that had the overall best performance was the Easy Ensemble model, however, due to the perfect score that it achieved in all areas, further testing of this model is in order to ensure that we are not overfitting the model so that it performs great on this data set and poorly on others. As a back up, the Balanced Random Forest Classifier model also showed very high accuracy and recall as well as notably high precision. If the Easy Ensemble model is later proven to be overfit, the Balanced Random Forest model should be able to step in. 
