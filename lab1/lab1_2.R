mydata = read.csv("parkinsons.csv")

# 1 y ~ N, p17


# 2 prepare data
set.seed(12345)
n = dim(mydata)[1]
id = sample(1:n, floor(n*0.6))

#only scaling result and explaining variables
mydata.scaled = scale(mydata[,5:22])

dat.train = mydata.scaled[id,]
dat.test = mydata.scaled[-id,]

# 3 implement functions



#loglikelihood of training data
Loglikelihood <- function(w, sigma) {# w as a col vector
  n = dim(dat.train)[1]
  y <- matrix(dat.train[,1], ncol=1) # col vector y
  x <- matrix(dat.train[,3:18], ncol=16)
  
  wTx = x%*%w
  res = y-wTx
  res = res*res
  res <- res/(2*sigma^2)
  res <- sum(res)
  
  loglikelihood <- (-n/2*log(2*pi*sigma^2)- res)
  return(loglikelihood)
}

Ridge <- function(vector, lambda) {# vector=c(w,sigma)
  w = vector[1:(length(vector)-1)]
  sigma = vector[length(vector)]
  w.ridge <- (-Loglikelihood(w, sigma)+lambda*norm(w,"2")^2)
  return(w.ridge)
}

RidgeOpt <- function(lambda) {
  vector=rep(1,17)
  opt <- optim(vector, Ridge, NULL, lambda, method="BFGS") # provide guess for w and sigma
  return(opt$par) 
}

DF <- function(lambda) { # df = trace of ridge hat matrix = trace((X(XTX+lambda*I)^-1XT)
  x <- matrix(dat.train[,3:18], ncol=16)
  hat <- x%*%solve(t(x)%*%x+lambda*diag(16))%*%t(x)
  df <- sum(diag(hat))
  return(df)
}


opt1 <- RidgeOpt(1)
w1 = opt1[1:16]
sigma1 = opt1[17]
opt100 <- RidgeOpt(100)
w100 = opt100[1:16]
sigma100 = opt100[17]
opt1000 <- RidgeOpt(1000)
w1000 = opt1000[1:16]
sigma1000 = opt1000[17]


#train data
y <- matrix(dat.train[,1], ncol=1)
x <- matrix(dat.train[,3:18], ncol=16)

y1 <- x%*%w1
y100 <- x%*%w100
y1000 <- x%*%w1000

# train MSE
MSE.train.1 <- mean((y-y1)^2) # 0.873277
MSE.train.100 <- mean((y-y100)^2) #0.8790595
MSE.train.1000 <- mean((y-y1000)^2) #0.9156267

#test data
y <- matrix(dat.test[,1], ncol=1)
x <- matrix(dat.test[,3:18], ncol=16)

y1.test <- x%*%w1
y100.test <- x%*%w100
y1000.test <- x%*%w1000 

#test MSE
MSE.test.1 <- mean((y-y1.test)^2) # 0.929036
MSE.test.100 <- mean((y-y100.test)^2) #0.9263161
MSE.test.1000 <- mean((y-y1000.test)^2) #0.9479166

# Smallest MSE for test is when lambda = 100

# 5
AIC <- function(lambda, w, sigma) {
  aic <- 2*(DF(lambda)-max(Loglikelihood(w, sigma)))
  return(aic)
}

aic1 <- AIC(1, w1, sigma1) #9553.596
aic100 <- AIC(100, w100, sigma100) #9569.013
aic1000 <- AIC(1000, w1000, sigma1000) #9704.087

# lambda = 1is smallest -> best