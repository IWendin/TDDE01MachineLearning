mydata = read.csv("tecator.csv")

# prepare data
set.seed(12345)
n = dim(mydata)[1]
id = sample(1:n, floor(n*0.5))
dat.train = mydata[id,]
dat.test = mydata[-id,]

# No pre-processing 
# 1
  # remove all exept channels 1-100, collect fat
dat.train.fat = dat.train[,2:102]
dat.train.fat.channels = dat.train[,2:101]
dat.train.fat.target = dat.train[,102]
dat.test.fat = dat.test[,2:102]
dat.test.fat.target = dat.test[,102]
dat.test.fat.channels = dat.test[,2:101]

  # Fitting to training data
reg.mod <- lm(Fat ~., data = dat.train.fat)
  # Probabilistic model
reg.mod$coefficients
  # Note that many variables have unsatisfactory significance levels
summary(reg.mod)

  # Predict training data and test data 
reg.fit.train <- predict(reg.mod, dat.train.fat)
reg.fit.test <- predict(reg.mod, dat.test.fat)

  # Error: E((train- predict of train)^2) = 0.005709117 good
err.train.fat = mean((dat.train.fat.target-reg.fit.train)^2)
err.train.fat

  # Error: E((test - predict of test)^2) = 722.4294  #Really bad(!)
err.test.fat = mean((dat.test.fat.target-reg.fit.test)^2)
err.test.fat

  # Checking residual - control
#plot(x=dat.test$ï..Sample, y=((dat.test.fat.target-reg.fit.test)^2)) 


# 2 LASSO on fat - objective function
# Objective: minimize the cost function = minimize -loglikelihood = min sum((y-y_hat)^2) = min sum((y - predicted y)^2)
# Subject to lambda*sum(abs(w))<= s, where lambda = 1
# So objective function is minimize residuals, just like the regression, but now with an additional boundary -> glmnnet can solve this problem 

# 3- glmnet() with alpha = 1
library(glmnet)
     
 lasso.mod <- glmnet(as.matrix(dat.train.fat.channels),dat.train.fat.target, alpha=1, family="gaussian" )
 # Plot function displays relation between coefficients and log lambda
 plot(lasso.mod, xvar="lambda")
  
 # When penalty is greater (more to the right(e0)) then few coefficients are != 0, and when the penalty is smaller, then we can have more coefficients (lambda = e-4)
 # Most coefficients are variating between 0 and -50, but three are positive
 
 #df =  The number of nonzero coefficients for each value of lambda. Visually looks like around lambda = 0 has 3

plot(x=lasso.mod$lambda, y=lasso.mod$df, xlab="lambda", ylab="df = non-zero coefficiants") # 3 coeff around around lambda = 0,9 is confirmed


lambda = which(lasso.mod$lambda <= 1)
lasso.mod$df[22] # has 3 coefficients
lasso.mod$lambda[22] # 3 coefficients when lambda = 0.8530452

# 4 Df - penalty parameter plot -> Df vs lambda

plot(x=lasso.mod$lambda, y=lasso.mod$df, xlab="Lambda", ylab="df") # The trend is expected, since the closer to 0 (the greater more negative log lambda), the more coefficients are not 0 (same result as the log plot above)

# 5 ridge regression - alpha = 0
ridge.mod <- glmnet(as.matrix(dat.train.fat.channels), dat.train.fat.target, alpha=0, family = "gaussian")

plot(ridge.mod, xvar="lambda") # THe ridge plot for lambda also has fewer non-zero coefficients with greater lambda, but only a few values of lambda in this graph compared to the lasso
plot(x=ridge.mod$lambda, y=ridge.mod$df) # no non-zero coefficients for any of the lambda values -> No value of lambda will give 3 coefficients

# 6 Cross-validation lasso
lasso.cv.mod = cv.glmnet(as.matrix(dat.train.fat.channels),dat.train.fat.target, alpha=1, family="gaussian")

plot(lasso.cv.mod) # The CV score = MSE, is low for small lambdas (small penalties), MSE takes off around log lambda = -2,5, increases to 0 and then rises more slow, lambda.min is the dotted line to the left
# The plot shows that the log lambda_min has an overlapping MSE interval with log lambda = -2, and it is thus impossilbe to say that log lambda min is significantly better than -2 
# It looks like 9 coefficients are in the model for lambda min, but check coef() to verify
lasso.cv.mod$lambda.min # -2.856921

coef.opt = matrix(coef(lasso.cv.mod, s='lambda.min'))
table(coef.opt[,1]!=0)# there are 9 non-zero coefficients when optimal lambda (including intersection), as thought!


test.glmnet = predict(lasso.cv.mod, newx=as.matrix(dat.test.fat.channels), s="lambda.min") # specifying predict for lambda min

scatter.smooth(dat.test.fat.target, test.glmnet)
abline(0,1, col="red") # line which the data should follow

#plot(x=dat.test[,1], y=test.glmnet, xlab="Index", ylab="", col="pink")
#points(x=dat.test[,1], y=dat.test.fat.target, xlab="Index", ylab="Predicted and Test", col="blue")
#legend("top", inset=.02, legend=c("Predicted", "Test"), col=c("pink", "blue"),  lty=1:2, cex=0.8)

# The model is not very good, but not horrible.
# Notice the that values are off at the outer points
# Checking residuals

err.test.fat = mean((dat.test.fat.target - test.glmnet)^2)
err.test.fat # squared residuals = 13.67339 much lower that ~700 before

# Plotting residuals show that some predictions are quite off - often predicted bigger value than test 
residual = dat.test.fat.target - test.glmnet
plot(x=dat.test.fat[,1], y=residual, col="purple")
abline(0,0, col="red")

# 7 Generate new target values - y ~ N(wTx, sigma^2)
  # w_opt = coef.opt from Lasso result (+ intercept), dat.test.fat.channels = x (-intercepts, must add x0)

  # Add w0 to first position
x0 = matrix(rep(c(1), 108), ncol=1)
x0
test.channels = dat.test.fat.channels
test.channels = cbind(test.channels, x0)
test.channels = subset(test.channels, select=c(101,1:100))

  # change to matrix and transpose
test.channels = as.matrix(test.channels)
test.channels = t(test.channels)

  # Create the y by multiplying w with x and add error from N()
s2.residual = mean(residual^2)
e = matrix(rnorm(108, 0, s2.residual), nrow = 1)
w = t(coef.opt)
new.test.target = w%*%test.channels
new.test.target = new.test.target + e
new.test.target

#plot(x= dat.test[,1], y=dat.test.fat.target, col="blue")
#points(x=dat.test[,1], y=new.test.target, col="orange")
#legend(x=50, y=50, inset = 0.02, legend = c("Generated target", "Original Fat"), col=c("orange", "blue"), lty=1:2, cex=0.8)

scatter.smooth(dat.test.fat.target, new.test.target)
abline(0,1, col="red")

# The predictions are still near the target but sometimes a bit off. One can clearly see the displayed pattern
err.test.fat = mean((dat.test.fat.target - new.test.target)^2)
err.test.fat # squared residuals =  181.5103 worse than 13.67339 much lower that ~700 before
