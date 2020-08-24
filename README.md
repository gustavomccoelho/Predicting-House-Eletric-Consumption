##  Predicting House Eletric Consumption

This is a IoT project focused on predicting home appliance eletric consumption. Data is provided by colecting the output of temperature and humidity sensors in a wireless networks, in addition to the weather forecast from a airport weather station. The data is stored every 10 minutes during 5 months. The train and test sets are randomly splitted from this dataset.   

##  Scrip overview

The train and test data are loaded and the first action is to analyse the target variable histogram:

![target_hist](/pictures/target_hist.png)

As we can see, the distribution is close to a normal shape, but with a long tale, which indicated existance of outliers. This is confirmed by looking at the boxplot:

![target_boxplot](/pictures/target_boxplot.png)

The outliers are removed by keeping the dataset 97% quantile. The result is as follows:

![target_hist_after](/pictures/target_hist_after.png)
![target_boxplot_after](/pictures/target_boxplot_after.png)

The nexty step is to verify the presence of NA values, which is none. 

Next, we analyse the WeekStatus and Day_of_week variables. It's decided to not trust it's validity (i.e if they match with the date variable), and they are remade by using the date column as reference. In addition, we decide to add the hour variable, since it is reasonable to assume that the consumption has a high correlation to the hour of the day. 

Here we can see the correlation matrix:

![correlation](/pictures/correlation.png)

By using Ranger (Random Forest Model), we are able to collect the feature importance (Gini index):

![importance](/pictures/importance.png)

As we can see, the following variables are probably not helpfull to the model, and thus should be removed:

-WeekStatus
-Day_of_week
-rv1
-rv2

The data is scaled with zero mean and unitary standard deviaton, and is ready for the prediction models.

The following models were tested:

-Ranger (Random Forest) - R2: 0.57, RMSE: 42.06
-SVM - R2: 0.27, RMSE: 54.36
-Linear Regression - R2: 0.26, RMSE: 54.95

As we can see, Random Forest seem to be the best out of this options. 

At least, cross validation is used in the Random Forest model, which didn't add any accuracy. 


