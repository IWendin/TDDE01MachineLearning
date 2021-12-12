# Create project and store the .csv file in it before working
mydata = read.csv("optdigits.csv", header= FALSE)

#1. Partition data
n=dim(mydata)[1]
set.seed(12345) 
id=sample(1:n, floor(n*0.5)) 
dig.train=mydata[id,] 

id1=setdiff(1:n, id)
set.seed(12345) 
id2=sample(id1, floor(n*0.25)) 
dig.valid=mydata[id2,]

id3=setdiff(id1,id2)
dig.test=mydata[id3,] 

#2. Fit 30-nearest neighbor - why continuous data?
library(kknn)

# Testing with training data
dig.model_train = kknn(as.factor(V65) ~ ., train=dig.train, test=dig.train, k=30, kernel="rectangular")
fit_train = fitted(dig.model_train)

# Testing with test data
dig.model = kknn(as.factor(V65) ~ ., train=dig.train, test=dig.test, k=30, kernel="rectangular")
fit = fitted(dig.model)

# Confusion matrix train data
table(dig.train$V65, fit_train)

# Confusion matrix test data - several 8 was thought to be 1, 9 had several misclass, it predicted many to be 9
table(dig.test$V65, fit)

misclass = function(X,X1) {
  n = length(X)
  return (1-sum(diag(table(X,X1)))/n)
}

# Misclassification train - ~4% misclass, slightly better than test but no overfitting
misclass(dig.train$V65, fit_train)

# Misclassification test - ~5% misclass
misclass(dig.test$V65, fit)

#3. Find 8:s in training data

  # add column with probability to classify the digit as an 8 
dig.train_prob8 = cbind(dig.train,prob8=dig.model_train$prob[,9])

  # subset with only 8 and order ascending
dig.train_prob8 = subset(dig.train_prob8,V65==8)
dig.train_prob8 = dig.train_prob8[order(dig.train_prob8$prob8),]
dig.train_prob8

  # Lowest: We can see that rowname=1793 had 10% prob to be classified as an 8, rowname=869 had 13,3%, rowname=2068 had 16,7% chance

dig.train_prob8 = dig.train_prob8[order(dig.train_prob8$prob8, decreasing=TRUE),]
dig.train_prob8

  # Highest: We can see that rowname=1890 had 100% prob to be classified as an 8, rowname=1982 had 100% chance

  # create heatmap - good values - Note that the output in the heatmap has been flipped two times compared to the matrix
  # Can see the loops. It is easy to see that it is an 8 when I know that it is an 8 
dig.test_num = as.matrix(mydata[1890,1:64])
dig.test_num = matrix(dig.test_num, nrow=8, byrow=TRUE)
heatmap(dig.test_num, Colv=NA, Rowv=NA)

dig.test_num = as.matrix(mydata[1982,1:64])
dig.test_num = matrix(dig.test_num, nrow=8, byrow=TRUE)
heatmap(dig.test_num, Colv=NA, Rowv=NA)

  # create heatmap - bad values
  # Cannot see the loops, only a blob/vertical line -> cannot see that it is an 8
dig.test_num = as.matrix(mydata[1793,1:64])
dig.test_num = matrix(dig.test_num, nrow=8, byrow=TRUE)
heatmap(dig.test_num, Colv=NA, Rowv=NA)

dig.test_num = as.matrix(mydata[869,1:64])
dig.test_num = matrix(dig.test_num, nrow=8, byrow=TRUE)
heatmap(dig.test_num, Colv=NA, Rowv=NA)

dig.test_num = as.matrix(mydata[2068,1:64])
dig.test_num = matrix(dig.test_num, nrow=8, byrow=TRUE)
heatmap(dig.test_num, Colv=NA, Rowv=NA)

#4. Kknn - plot - Seems like k=4 is minimal - More neighbors with larger k makes the model more complex, but more stable with smaller variance

k.set = data.frame(K=integer(), Test=integer(), Valid=integer() )

for (k in 1:30) {
    #training
  dig.kknn = kknn(as.factor(V65)~., dig.train, dig.train, k=k, kernel="rectangular")
  m1 = misclass(dig.train$V65, dig.kknn$fitted.values)
    #validation
  dig.kknn.valid = kknn(as.factor(V65)~., dig.train, dig.valid, k=k, kernel="rectangular")
  m2 = misclass(dig.valid$V65, dig.kknn.valid$fitted.values)
  
  k.set = rbind(k.set, c(k,m1,m2)) 
}
colnames(k.set) = c("K", "Test", "Valid")

plot(k.set$K, k.set$Test, col="blue", main="Misclasification of training and validation data", ylab="Misclasification", xlab="K")
points(k.set$K, k.set$Valid, col="green")


  #Test error for K = 4 - test error is 2,5% which is about the same as validation error and worse than training error
dig.kknn.test = kknn(as.factor(V65)~., dig.train, dig.test, k=4, kernel="rectangular")
misclass(dig.test$V65, dig.kknn.test$fitted.values)

#5. KKnn fit to training data - K = 6 is the k for maximized likelihood


k.set = data.frame(K=integer(), Log_likelihood=integer())

for (k in 1:30) {
  
  dig.kknn.valid = kknn(as.factor(V65)~., dig.train, dig.valid, k=k, kernel="rectangular")

  n = dim(dig.valid)[1]
  # column indexes = target values + 1
  index = matrix(c(1:n,(dig.valid$V65+1)), ncol = 2)
  prob = dig.kknn.valid$prob[index]
  prob = matrix(prob, ncol=1)
  prob = prob + 1e-15
  m1 = apply(prob, 2, log)
  m2 = apply(m1, 2, sum)
  
  k.set = rbind(k.set, c(k, m2))

}

colnames(k.set) = c("K", "Log_likelihood")
plot(k.set$K, k.set$Log_likelihood, xlab="K", ylab="Log likelihood")

# Find optimal
index = which(k.set$Log_likelihood== max(k.set$Log_likelihood))
K.opt = k.set$K[index]
K.opt
