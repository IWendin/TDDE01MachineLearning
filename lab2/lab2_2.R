data = read.csv2("bank-full.csv", stringsAsFactors = TRUE)

# 1 remove duration - column 
data = data[-12]
n = dim(data)[1]

set.seed(12345)
id = sample(1:n, floor(n*0.4))
train = data[id, ]

id1 = setdiff(1:n, id)
set.seed(12345)
id2 = sample(id1, floor(n*0.3))
valid = data[id2, ]

id3 = setdiff(id1, id2)
test = data[id3, ]

# 2 Fit decision tree to training data (y = yes/no)

library(tree)
# convert chr to factor for tree

#train.factor <- as.data.frame(unclass(train), stringsAsFactors=TRUE)
#valid.factor <- as.data.frame(unclass(valid), stringsAsFactors = TRUE)
#test.factor <- as.data.frame(unclass(test), stringsAsFactors = TRUE)

# 2a
tree.a <- tree(as.factor(y)~., data=train)
summary(tree.a) #Misclassification error rate: 0.1048 = 1896 / 18084
plot(tree.a)
text(tree.a, pretty=0)
#print(tree.a)

# 2b smallest node size = 7000 (up from 10)
tree.b <- tree(as.factor(y)~., data=train, control = tree.control(nobs=18084, minsize=7000))
summary(tree.b) #Misclassification error rate: 0.1048 = 1896 / 18084  - reduced to 5 terminal
plot(tree.b)
text(tree.b, pretty=0)
#print(tree.b)

# 2c Decision trees minimum deviance to 0.0005. (down from 0.01)
tree.c <- tree(as.factor(y)~., data=train, control = tree.control(nobs = 18084, mindev=0.0005))
summary(tree.c) #Misclassification error rate: 0.09362 = 1693 / 18084  BEST! - more divisions
plot(tree.c)
text(tree.c, pretty=0)
#print(tree.c)

# validation
y.a <- predict(tree.a, newdata = valid, type="class") #class -> get the output as yes/no
mis.a = 1-sum(diag(table(valid$y, y.a)))/length(y.a)
mis.a # 0.1092679 misclass

y.b <- predict(tree.b, newdata = valid, type="class")
mis.b = 1-sum(diag(table(valid$y, y.b)))/length(y.b)
mis.b # 0.1092679 misclass - same

y.c <- predict(tree.c, newdata = valid, type="class")
mis.c = 1-sum(diag(table(valid$y, y.c)))/length(y.c)
mis.c # 0.1118484 misclass - larger!

# If the required deviance is reduced from 0.01 to 0.005,
# then division of nodes can continue longer and more nodes will be generated
# this is because smaller differences are accepted

# minsize is the smallest amount of samples accepted within a node for it to split,
# thus if minsize is increased from 10 to 7000, then some divisions might
# no longer be allowed since the node children might contain too few observations

# 3 - optimal depth of 2c

trainScore = rep(0,50)
testScore = rep(0,50)

# 2:50 because object of class "singlenode" when only 1 leaf = only root
for(i in 2:50) {
  prunedTree = prune.tree(tree.c, best=i)
  pred.valid = predict(prunedTree, newdata = valid, type="tree") #tree -> get output as a tree
  trainScore[i] = deviance(prunedTree)
  testScore[i] = deviance(pred.valid)
}

plot(2:50, trainScore[2:50], col="green", ylim = c(8000,12000), main = "Deviance to no of leaves", xlab = "Number of leaves 2:50", ylab = "Deviance")
points(2:50, testScore[2:50], col="purple")
legend(x=45, y=12000, inset = 0.02, legend = c("Test", "Valid"), col = c("green", "purple"), lty = 1:2, cex = 0.8 )
abline(8200, 0, col = "red")
# It looks like the smallest deviance for validation data is achieved at around 20 leaves

# Control - zoom in
plot(19:23, testScore[19:23], col = "purple",  main = "Deviance to no of leaves [focus plot]", xlab = "Number of leaves 19:23", ylab = "Deviance")
# We see that 22 leavs has the smallest deviance

# Most important variables
tree.opt = prune.tree(tree.c, best = 22)
summary(tree.opt)
# For decision making, valiables: "poutcome","month","contact","pdays","age","day","balance","housing","job"  
# were used and are thus deemed to be the most important for decision making in the tree
# tree structure
# The structure of the tree
print(tree.opt)
plot(tree.opt)

# Misclass rate and confusion matrix for the test data
test.opt <- predict(tree.opt, newdata = test, type = "class")

table("Target"=test$y, "Predicted"=test.opt)[c(2,1),c(2,1)]
mis.opt = 1-sum(diag(table(test$y, test.opt)))/length(test.opt)
mis.opt # 0.1089649 - about the same as for training data (0.1039), and better than 2a,b,c
# but 10% misclassification is a classification tool I would trust in medicine

# 4 Classification of test data with loss matrix -> need rpart


library(rpart)
# Create loss matrix - in this case "no" is positive, "yes" is negative
loss <- matrix(c(0,1,5,0), ncol = 2, byrow = TRUE)
tree.loss <- rpart(as.factor(y)~., data = train, parms = list(loss = loss))
test.loss <- predict(tree.loss, newdata = test, type = "class")

# Do not use rpart? use instead the optimal model and change the probabilities
test.loss <- predict(tree.opt, newdata = test, type = "vector")

#change probability to make it harder to get a no
test.loss.class <- ifelse(test.loss[,1]>0.8,"no","yes")


table(test$y, test.loss.class)[c(2,1),c(2,1)] # change displaying order

# optimal tree from 3
table(test$y, test.opt)[c(2,1),c(2,1)]
# Looks like an improvements since we are clearly punishing the predicting no, and favouring predicted yes 

mis.loss <- 1-sum(diag(table(test$y, test.loss)))/length(test.loss)
mis.loss # 0.140519 - worse than opt 0.1089649 in misclassifications, but better result if we think that predicting no is more severe

# 5 optimal tree + Naive bayes 
# Since the result y="yes" is interpreted as good for the company,
# Y=1=yes if the probability of y="yes" > pi 
# -> find probabilities of opt to classify as yes

test.opt.prob <- predict(tree.opt, newdata = test, type = "vector")
# translate yes and no in test data to 1 and 0 for comparison
test.num <- matrix(ifelse(test$y == "yes", 1, 0))


tp.rate <- data.frame()
fp.rate <- data.frame()

N.plus = sum(test.num)
N.minus = length(test.num) - N.plus

for (i in seq(from = 0.05, to = 0.95, by = 0.05)) {
  # compare probability of yes with pi, if greater than pi then y.hat = 1 
  y.hat <- matrix(ifelse(test.opt.prob[,2]>i, 1, 0))
 
  # CONTROL confusion matrix with order
  #tn, fn, fp, tp 
  #t = table(test.num, y.hat)
  
  tp = sum(y.hat*test.num)
  tp.rate = rbind(tp.rate, tp/N.plus)
  
  
  fp = sum(y.hat[which(y.hat == 1)])-tp
  fp.rate = rbind(fp.rate, fp/N.minus)

}

names(tp.rate)[1] <- c("tp.rate")
names(fp.rate)[1] <- c("fp.rate")

# Naive Bayes
library(e1071)
bayes.fit = naiveBayes(as.factor(y)~., data = train)
# yes and no class probabilities
bayes.pred = predict(bayes.fit, newdata = test, type="raw") # raw -> yes/no answers

bayes.tp.rate <- data.frame()
bayes.fp.rate <- data.frame()

for (i in seq(from = 0.05, to = 0.95, by = 0.05)) {
  # compare probability of yes with pi, if greater than pi then y.hat = 1 
  bayes.y.hat <- matrix(ifelse(bayes.pred[,2]>i, 1, 0))
  
  # CONTROL confusion matrix with order
  #tn, fn, fp, tp 
  #t = table(test.num, bays.y.hat)
  
  bayes.tp = sum(bayes.y.hat*test.num)
  bayes.tp.rate = rbind(bayes.tp.rate, bayes.tp/N.plus)
  
  
  bayes.fp = sum(bayes.y.hat[which(bayes.y.hat == 1)])-tp
  bayes.fp.rate = rbind(bayes.fp.rate, bayes.fp/N.minus)
  
}

names(bayes.tp.rate)[1] <- c("bayes.tp.rate")
names(bayes.fp.rate)[1] <- c("bayes.fp.rate")


# ROC curve
#fp rate  vs tp rate ROC?
plot(fp.rate$fp.rate, tp.rate$tp.rate, ylab = "TP rate", xlab = "FP rate", ylim = c(-0.1,1), col = "magenta", main ="ROC curves")
points(bayes.fp.rate$bayes.fp.rate, bayes.tp.rate$bayes.tp.rate, ylab = "TP rate", xlab = "FP rate", ylim = c(-0.1,1), col = "blue")
legend(0,1, legend = c("Optimal Tree", "Naïve Bayes"), col = c("magenta", "blue"), lty=1:2)

# We can see that the curve for the optimal tree is higher than the naïve bayes <-> has higher ratio of TP for the same FP rate -> Optimal tree is better (larger integral)


# Olegs way not using rpart
#11030 något
#771, 814
#???


