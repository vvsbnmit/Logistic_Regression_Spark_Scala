# Logistic_Regression_Spark_Scala
This is the code for Logistic Regression using SparkML using Scala

I am going to use the Global Terrorism Database (GTD) for this project. My goal is to predict a binary response variable corresponding to the lethality of a terrorist attack. I am using SparkML to fit a logistic regression classifier.

The GTD contains several variables associated with each terrorist attack. Many if not most of these variables are categorical in nature. Noting that logistic regression expects continuous predictors and a binary response, categorical variables must be recast into dummy variables (one-hot representation) before they can be used in algorithms such as logistic regression.

The response (dependent) variable is “nkill.” It is a continuous variable that corresponds to the total number of human lives lost as a result of the terrorist attack. However, we are predicting a binary response. Pig is used for data wrangling purpose. 
