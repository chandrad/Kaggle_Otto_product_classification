gc(reset=TRUE)
rm(list=ls(all=TRUE))
list.files("~/Otto")
setwd('~/Otto/')
install.packages("h2o")
library(h2o)
localH20 <- h2o.init(nthread=16,Xmx="60g")


train <- read.csv("train.csv")
head(train)
for(i in 2:94){
  train[,i] <- as.numeric(train[,i])
  train[,i] <- sqrt(train[,i]+(3/8))
}


test <- read.csv("test.csv")
head(test)
for(i in 2:94){
  test[,i] <- as.numeric(test[,i])
  test[,i] <- sqrt(test[,i]+(3/8))
}



train.hex <- as.h2o(localH20,train)
test.hex <- as.h2o(localH20,test[,2:94])

predictors <- 2:(ncol(train.hex)-1)
response <- ncol(train.hex)

submission <- read.csv("submission.csv")
head(submission)
submission[,2:10] <- 0

for(i in 1:20){
  print(i)
  model <- h2o.deeplearning(x=predictors,
                            y=response,
                            data=train.hex,
                            classification=T,
                            activation="RectifierWithDropout",
                            hidden=c(1024,512,256),
                            hidden_dropout_ratio=c(0.5,0.5,0.5),
                            input_dropout_ratio=0.05,
                            epochs=50,
                            l1=1e-5,
                            l2=1e-5,
                            rho=0.99,
                            epsilon=1e-8,
                            train_samples_per_iteration=2000,
                            max_w2=10,
                            seed=16801)
  submission[,2:10] <- submission[,2:10] + as.data.frame(h2o.predict(model,test.hex))[,2:10]
  print(i)
  write.csv(submission,file="submission.csv",row.names=FALSE) 
}      


