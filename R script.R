library(readr)
library(dlookr)
Train <- read_csv("Train.csv")
colnames(Train)
str(Train)


plot_na_pareto(Train[1:5])
plot_na_pareto(Train[6:10])
plot_na_pareto(Train[11:15])
plot_na_pareto(Train[16:20])
plot_na_pareto(Train[21:25])
plot_na_pareto(Train[26:30])
plot_na_pareto(Train[31:35])
plot_na_pareto(Train[36:40])
plot_na_pareto(Train[41:45])
plot_na_pareto(Train[46:49])

library('dplyr')
na_count1 <-sapply(Train, function(y) sum(length(which(is.na(y)))))
na_count1 <- data.frame(na_count1)
na_count1


Train<-select(Train, -youtube,-medium,-asset_id)


data_features=colnames(Train)
data_features


for (var in data_features){
  if (class(Train[[var]])=="numeric"){
    Train[is.na(Train[[var]]),var] <- mean(Train[[var]], na.rm = TRUE)
  }else if (class(Train[[var]]) %in% c("character", "factor")) {
    Train[is.na(Train[[var]]),var] <-Mode(Train[[var]], na.rm = TRUE)
  }
}




data_features=colnames(Train)
data_features



for (i in data_features){
    print(paste("The unique values for feature",i, "are :",length(unique(Train[[i]]))))
  }





library(caret)
# The createDataPartition function is used to create training and test sets
trainIndex <- createDataPartition(Train$close, 
                                  p = .6, 
                                  list = FALSE, 
                                  times = 1)

TRAIN <-Train[trainIndex, ]
TEST <- Train[-trainIndex, ]



library(randomForest)
rf_model <- randomForest(close ~open+high+low+volume+market_cap+volume_24h_rank+
                           social_volume_24h_rank+market_cap_global+news+tweet_sentiment_impact5,
                         data = TRAIN,
                         ntree= 550,
                         mtry=3,
                         keep.forest=TRUE,
                         importance=TRUE)

rf_model


TF_model <- randomForest(close ~open+high+low+volume+market_cap+volume_24h_rank+
                           social_volume_24h_rank+market_cap_global+news+tweet_sentiment_impact5,
                         data = Train,
                         ntree= 550,
                         mtry=3,
                         keep.forest=TRUE,
                         importance=TRUE)

TF_model



varImp(rf_model)


y_hat <- predict(rf_model, TEST)

#predict(diab_pop.no_na_vals.train.rf_model, diab_pop.no_na_vals.test, type ="prob")

test.rf_scored <- as_tibble(cbind(TEST, y_hat))

glimpse(test.rf_scored)



RMSE_rf_TEST <- yardstick::rmse(test.rf_scored, truth=close, estimate=y_hat)

RMSE_rf_TEST









library(readr)
Test <- read_csv("Test.csv")
na_count1 <-sapply(Test, function(y) sum(length(which(is.na(y)))))
na_count1 <- data.frame(na_count1)
na_count1

Test<-select(Test, -youtube,-medium, -asset_id)







Mode <- function (x, na.rm) {
  xtab <- table(x)
  xmode <- names(which(xtab == max(xtab)))
  if (length(xmode) > 1) xmode <- ">1 mode"
  return(xmode)
}
data_features=colnames(Test)
data_features


for (var in data_features){
  if (class(Test[[var]])=="numeric"){
    Test[is.na(Test[[var]]),var] <- mean(Test[[var]], na.rm = TRUE)
  }else if (class(Test[[var]]) %in% c("character", "factor")) {
   Test[is.na(Test[[var]]),var] <-Mode(Test[[var]], na.rm = TRUE)
  }
}



Target <- predict(TF_model, Test)
test.rf_scored <- as_tibble(cbind(Test, Target))

glimpse(test.rf_scored)

Submission<-dplyr::select(test.rf_scored,id,Target)


write.csv(Submission,"Submission.csv",row.names = FALSE)






# Normalize the data
Train<-select(Train,-id)
maxs <- apply(Train, 2, max) 
mins <- apply(Train, 2, min)
scaled <- as.data.frame(scale(Train, center = mins, 
                              scale = maxs - mins))

# Split the data into training and testing set
index <- sample(1:nrow(Train), round(0.75 * nrow(Train)))
train_ <- scaled[index,]
test_ <- scaled[-index,]

library(neuralnet)
library(MASS)
nn <- neuralnet(close ~open+high+low+volume+market_cap+volume_24h_rank+
                  social_volume_24h_rank+market_cap_global+news+tweet_sentiment_impact5, 
                data = train_, hidden = c(5, 3), 
                linear.output = TRUE)

# Predict on test data
pr.nn <- compute(nn, test_)


pr.nn_ <- pr.nn$net.result * (max(Train$close) - min(Train$close)) 
+ min(Train$close)
test.r <- (test_$close) * (max(Train$close) - min(Train$close)) + 
  min(Train$close)
MSE.nn <- sum((test.r - pr.nn_)^2) / nrow(test_)
MSE.nn
