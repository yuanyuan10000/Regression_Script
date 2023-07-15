#norm.test<-data.frame(scale(testdata))
#data1<-as.matrix(norm.data[,1:117])
#data2<-as.matrix(norm.test[,1:117])

setwd("I:/caco2")

traindata <- read.csv('./caco2_result/XGB_caco2_RF117_train_error.csv',header=TRUE)    
testdata <- read.csv('./caco2_result/XGB_caco2_RF117_test_error.csv',header=TRUE) 

train_error = traindata[,3]
x = testdata[,3]

n <- length(x)  # 50 
plot(sort(x),(1:n)/n,type="s",ylim=c(0,1))
qqnorm(x)
