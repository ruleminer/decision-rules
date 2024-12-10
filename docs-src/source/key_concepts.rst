Introduction
====================

This section describes the key concepts behind the decision-rules package and how they are implemented.

Supported types of problems
---------------------------

This package support three types of problems: classification, regression and survival.

**Classification** is the process of assigning objects to specific categories based on the characteristics of examples. For example, given a dataset describing patient test results and a decision class indicating whether the patient is sick or not, a rule model can learn which test results indicate the patient's illness. In the case of classification, the conclusion of the decision rule will indicate the assignment of the example to the selected category (class).


**Regression** is a process used to understand how selected variables (characteristics) affect another variable. It allows building a model to predict the value of one numerical variable based on other variables. An example of regression analysis might be checking how the amount of time spent on physical activity affects blood cholesterol levels. Another example of a regression-type analysis may be examining how the characteristics of an apartment being sold, such as price, location, year of construction, affect the final selling price of that apartment. In the case of regression, the rule's conclusion will point to a specific numerical value.

**Survival analysis** is used to study the amount of time until a given event occurs. This method examines how different factors can affect the probability of survival over a specific time frame. An example of such an analysis may be understanding how long patients survive after heart surgery and how various factors, such as age, gender or pre-operative health status, affect that survival. Survival analysis can be applied not only to medical data, but also to predictive maintenance analysis, time-to-event economic studies and others. In the case of survival analysis, in the conclusion of the rule there is an estimator of the survival function, most often the Kaplan-Meier estimator. Such a function tells us how the probability of occurrence of the event under study changes over time.

Dataset
-------

Rule sets operate on tabular datasets in the form of pandas DataFrames.

* **Classification:** The label column can be a number or string (but will be treated as a discrete variable).
* **Regression:** The target column must be a number.
* **Survival Analysis:** Requires two columns:
    * **Survival time:** Numerical.
    * **Survival status:** String with values "1" (event occurred) and "0" (event did not occur). 

Rule Sets
----------

Rule sets in decision rules are predictive models consisting of multiple rules. However, rule sets are more than just a set of rules. They also define how prediction is performed using these rules. 

Each rule set must contain at least one rule, in addition, it also contains one special rule called the default rule. The default rule is used to make predictions for examples discovered by other rules. Such a rule behaves in a unique way. Its premise is a condition that is always true. For this reason, in this package we talk about default conclusions instead of default rules.

Rules
-----

Rules are the key components of the this package. Each rule consist of the main two parts: premise and conclusions. Rules have a following form:
    IF premise THEN conclusion


The conclusion is a part of rule specifying the decision that should be made for the example if it satisfyes the rule's premise part. Conclusions can have different form depending on the type of problem we're facing. 
    * **Classification:** Assigns the example to a specific class.
    * **Regression:** Predicts a value and provides a confidence interval (mean ± standard deviation). By default the value (lets denote is as α) is set to the mean value of the label attribute of all examples covered by the training set however it can be set to any other value. The confidence interval is by default defined as (α ± *std*), where *std* is standard deviation of the label attribute values of the examples covered by rule and can be modified.
    * **Survival Analysis:** Provides a Kaplan-Meier estimator of the survival function. Such a function tells us how the probability of occurrence of the event under study changes over time.

Conditions
----------

This package provides the following predefined condition types:

* *a* ∈ *A* (where *a* is numerical and *A* is a value interval) e.g., `age ∈ [18, 33)`
* *a* = *d* (where *a* is a discrete attribute and *d* is its specific value) e.g., `gender = male`
* *a* </>/≤/ ≥ *b* or *a* =/≠ *b* (where *b* is another attribute) e.g. `height > width`
* Conjunctions and disjunctions of these conditions (nested conditions are supported). e.g `gender = male ∧ age > 18`

New condition types can be easily implemented.

Prediction
----------

Rule sets perform prediction for given examples using specified prediction strategy. 
This package provides two predefined prediction strategies:

* **Voting:** All applicable rules "vote" for their predicted value. The final prediction is the average of these votes, weighted by rule voting weights.
* **Best Rule:** The best-matching rule (based on a chosen metric) is used for prediction.

Custom prediction strategies can be implemented.