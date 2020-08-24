# Setting up work directory

setwd("~/Projects/Predicting-House-Eletric-Consumption/code")

library(ranger)
library(ggplot2)
library(e1071)
library(caret)
library(xgboost)
library(spm)

# Loading train and test data

train <- read.csv("../dataset/train_data.csv")
test <- read.csv("../dataset/test_data.csv")

# Checking basic shape

str(train)
str(test)

View(train)
View(test)

# Checking target feature

table(train$Appliances)
table(test$Appliances)

quantile(train$Appliances, c(0.25, 0.50, 0.75, 0.90, 0.97))

ggplot(train, aes(x = Appliances)) + geom_histogram(binwidth = 10) + ggtitle("Distribution of Appliances")
ggplot(train, aes(x = Appliances)) + geom_boxplot() + ggtitle("Distribution of Appliances")

# Removing outliers from the target

threshold <- quantile(train$Appliances, 0.97)

train_2 <- train[train$Appliances < threshold, ]
test_2 <- test[test$Appliances < threshold, ]

length(table(train_2$Appliances))
length(table(test_2$Appliances))

ggplot(train_2, aes(x = Appliances)) + geom_histogram(binwidth = 1) + ggtitle("Distribution of Appliances after removing outliers")
ggplot(train_2, aes(x = Appliances)) + geom_boxplot() + ggtitle("Distribution of Appliances after removing outliers")

# Checking the numbers of NA values

sum(is.na.data.frame(train_2))
sum(is.na.data.frame(test_2))

# We will not trust on the variables "Weekstatus" and "Day_of_week". Instead, we will delete them and remake it form the variable data

train_3 <- train_2[,!(names(train_2)) %in% c("WeekStatus", "Day_of_week")]
test_3 <- test_2[,!(names(test_2)) %in% c("WeekStatus", "Day_of_week")]

train_3$date <- as.POSIXct(train_3$date, format = "%Y-%m-%d %H:%M:%S")
test_3$date <- as.POSIXct(test_3$date, format = "%Y-%m-%d %H:%M:%S")

train_3$Day_of_week <- weekdays(train_3$date)
test_3$Day_of_week <- weekdays(test_3$date)

set_weekday_language <- function(X){
  weekday = ifelse(X == "segunda", "monday",
                   ifelse(X == "terça", "tuesday",
                          ifelse(X == "quarta", "wednesday",
                                 ifelse(X == "quinta", "thursday",
                                        ifelse(X == "sexta", "friday",
                                               ifelse(X == "sábado", "saturday", "sunday"))))))
  return(weekday)
}

train_3$Day_of_week <- as.factor(apply(train_3["Day_of_week"], 2, set_weekday_language))
test_3$Day_of_week <- as.factor(apply(test_3["Day_of_week"], 2, set_weekday_language))

set_weekstatus <- function(X){
  weekstatus = ifelse(X == "saturday" | X == "sunday", "weekend", "weekday")
  return(weekstatus)
}

train_3$WeekStatus <- as.factor(apply(train_3["Day_of_week"], 2, set_weekstatus))
test_3$WeekStatus <- as.factor(apply(test_3["Day_of_week"], 2, set_weekstatus))

# Creating another variable related to the hour of each row

train_3$hour <- as.factor(format(strptime(train_3$date,"%Y-%m-%d %H:%M:%S"),'%H'))
test_3$hour <- as.factor(format(strptime(test_3$date,"%Y-%m-%d %H:%M:%S"),'%H'))

train_3$date <- NULL
test_3$date <- NULL

# Analyzing the Correlation Matrix

train_cor = train_3
train_cor$WeekStatus <- as.integer(train_3$WeekStatus)
train_cor$Day_of_week <- as.integer(train_3$Day_of_week)
train_cor$hour <- as.integer(train_3$hour)

heatmap(cor(train_cor))

# Scaling the numeric columns

factors = c("WeekStatus", "Day_of_week", "hour")
normParam <- preProcess(train_3[,!(names(train_3)) %in% factors])
train_4 = predict(normParam, train_3[,!(names(train_3)) %in% factors])
test_4 = predict(normParam, test_3[,!(names(test_3)) %in% factors])

for(col in factors){
  train_4[col] = train_3[col]
  test_4[col] = test_3[col]
}

# Feature importance

model <- ranger(Appliances ~ ., data = train_4, importance = "impurity")
importance <- as.data.frame(model$variable.importance)
names(importance) <- c("values")

ggplot(importance, aes(x = rownames(importance), y = values)) + 
          geom_bar(stat="identity", fill="#f68060", alpha=.6, width=.4) +
          coord_flip() +
          xlab("") +
          theme_bw()

# As expected, rv1 and rv2 have very low importance. So, we will remove them
# We will also remove the date variables, except for the hours

train_5 <- train_4[,!(names(train_4)) %in% c("rv1", "rv2", "Day_of_week", "WeekStatus")]
test_5 <- test_4[,!(names(test_4)) %in% c("rv1", "rv2", "Day_of_week", "WeekStatus")]

# Trying first model - Random Forest

model_1 <- ranger(Appliances ~ ., data = train_5)
predictions_1 = predict(model_1, test_5)
predictions_1 <- predictions_1$predictions
predictions_1 <- predictions_1*normParam$std["Appliances"] + normParam$mean["Appliances"]

# Evaluating model: R2 - 0.57, RMSE - 42.06

actual = test_5$Appliances*normParam$std["Appliances"] + normParam$mean["Appliances"]
rss_1 <- sum((predictions_1 - actual) ^ 2)  ## residual sum of squares
tss <- sum((actual - mean(actual)) ^ 2)  ## total sum of squares
rsq_1 <- 1 - rss_1/tss
RMSE_1 <- RMSE(pred = predictions_1, obs = actual)
rsq_1
RMSE_1

# Trying secound model (SVM)

model_2 <- svm(Appliances ~ ., data = train_5)
predictions_2 = predict(model_2, test_5)
predictions_2 <- predictions_2*normParam$std["Appliances"] + normParam$mean["Appliances"]

# Evaluating model: R2 - 0.27, RMSE - 54.36 

rss_2 <- sum((predictions_2 - actual) ^ 2)  ## residual sum of squares
rsq_2 <- 1 - rss_2/tss
RMSE_2 <- RMSE(pred = predictions_2, obs = actual)
rsq_2
RMSE_2

# Trying third model (Linear Regression)

model_3 <- lm(Appliances ~ ., data = train_5)
predictions_3 <- predict(model_3, test_5)
predictions_3 <- predictions_3*normParam$std["Appliances"] + normParam$mean["Appliances"]

# Evaluating model: R2 - 0.26, RMSE - 54.95

rss_3 <- sum((predictions_3 - actual) ^ 2)  ## residual sum of squares
rsq_3 <- 1 - rss_3/tss
RMSE_3 <- RMSE(pred = predictions_3, obs = actual)
rsq_3
RMSE_3

# Random Forest seem to be the best model. Let's try cross validating

train.control <- trainControl(method = "cv", number = 10)
model = train(Appliances ~ ., train_5, method = "ranger", trControl = train.control)

predictions = predict(model, test_5)
predictions <- predictions*normParam$std["Appliances"] + normParam$mean["Appliances"]

rss <- sum((predictions - actual) ^ 2)  ## residual sum of squares
rsq <- 1 - rss/tss
RMSE <- RMSE(pred = predictions, obs = actual)
rsq
RMSE
