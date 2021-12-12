data <- iris

# 1 scatterplot
plot(x = data$Sepal.Length, y = data$Sepal.Width, col = data$Species)
# Yes, there are is a distinct group of data in the right bottom corner,
# and the group of two data sets in upper left corner seems to be possible to split
# into two halves, but with less distinct result

# 2 Linear discriminant analysis

# 2a
# my (Length, width)
my.sepal.length <- tapply(data$Sepal.Length, data$Species, mean)
my.sepal.width <- tapply(data$Sepal.Width, data$Species, mean)

my.setosa = my.sepal.length[1]
my.setosa[2] = my.sepal.width[1]
my.setosa #  5.006  3.428 

my.versicolor = my.sepal.length[2]
my.versicolor[2] = my.sepal.width[2]
my.versicolor   # 5.936      2.770

my.virginica = my.sepal.length[3]
my.virginica[2] = my.sepal.width[3]
my.virginica   #  6.588     2.974 

# cov
data.setosa = data[which(data$Species=="setosa"),]
data.versicolor = data[which(data$Species=="versicolor"),]
data.virginica = data[which(data$Species == "virginica"),]

# Cov in order x=length, y=width
cov.setosa = cov(data.setosa[,1:2])
cov.setosa

cov.versicolor = cov(data.versicolor[,1:2])
cov.versicolor

cov.virginica = cov(data.virginica[,1:2])
cov.virginica 

# Prior probabilities
prior = table(data$Species)/length(data$Species)
prior[1] # setosa 0.3333333
prior[2] # versicolor 0.3333333
prior[3] # virginica 0.3333333

# 2b
cov.pooled = (cov.setosa*nrow(data.setosa)+cov.versicolor*nrow(data.versicolor)+cov.virginica*nrow(data.virginica))/nrow(data)
cov.pooled


# 2c Probabilistic model for LDA
# x|y = Ci, myi, Cov Ñ(myi, Cov)
# y|pi ~Multinomial(pi1,...,pik)


# 2d - discriminant function for each class
d.setosa <- function(x) {
  return (x%*%(solve(cov.pooled)%*%my.setosa)-0.5*my.setosa%*%solve(cov.pooled)%*%my.setosa+log(prior[1]))
}

d.versicolor <- function(x) {
  return (x%*%(solve(cov.pooled)%*%my.versicolor)-0.5*my.versicolor%*%solve(cov.pooled)%*%my.versicolor+log(prior[2]))
}

d.virginica <- function(x) {
  return(x%*%(solve(cov.pooled)%*%my.virginica)-0.5*my.virginica%*%solve(cov.pooled)%*%my.virginica+log(prior[3]))
}

# 2c Decision boundaries
# Exist in the points x which cause d1 = d2, d1 = d3 and d2 = d3
# Between setosa (d1) and versicolor (d2), we have that (w1 - w2)p + (w01 - w02) = 0, where p=(x, y)
w12 = solve(cov.pooled)%*%my.setosa - solve(cov.pooled)%*%my.versicolor
w012 = -0.5*my.setosa%*%solve(cov.pooled)%*%my.setosa+log(prior[1]) - (-0.5*my.versicolor%*%solve(cov.pooled)%*%my.versicolor+log(prior[2]))
# on the formula y = kx + m
k12 = -w12[1]/w12[2]  # 0.6458834
m12 = -w012/w12[2]    #  -0.4346283

# Between setosa and virginica
w13 = solve(cov.pooled)%*%my.setosa - solve(cov.pooled)%*%my.virginica
w013 = -0.5*my.setosa%*%solve(cov.pooled)%*%my.setosa+log(prior[1]) - (-0.5*my.virginica%*%solve(cov.pooled)%*%my.virginica+log(prior[3]))
# on the formula y = kx + m
k13 = -w13[1]/w13[2]  # 0.8413487
m13 = -w013/w13[2]    # -1.676298


# Between versicolor and virginica
w23 = solve(cov.pooled)%*%my.versicolor - solve(cov.pooled)%*%my.virginica
w023 = -0.5*my.versicolor%*%solve(cov.pooled)%*%my.versicolor+log(prior[2]) - (-0.5*my.virginica%*%solve(cov.pooled)%*%my.virginica+log(prior[3]))
# on the formula y = kx + m
k23 = -w23[1]/w23[2]  # 8.809989
m23 = -w023/w23[2]    # -52.29615


# 3 predict class
# creating a matrix with the discriminant values
comp <-as.data.frame(apply(as.matrix(data[,1:2]), 1, FUN=d.setosa), ncol = 1)
comp <- cbind(comp, apply(as.matrix(data[,1:2]), 1, d.versicolor))
comp <- cbind(comp, apply(as.matrix(data[,1:2]), 1, d.virginica))
names(comp) <- c("setosa", "versicolor", "virginica")

pred.iris <- matrix(colnames(comp)[apply(as.matrix(comp), 1, which.max)])
data.res <- cbind(data, pred.iris)

plot(x=data.res$Sepal.Length, y=data.res$Sepal.Width, col = as.factor(data.res$pred.iris))
# And with the decision bundaries from 2e
abline(m12, k12, col ="red")
abline(m13, k13, col ="green")
abline(m23, k23, col ="orange")


table("True"=data.res$Species, "Predict"=data.res$pred.iris)
# misclassification
1-sum(diag(table(data.res$Species, data.res$pred.iris)))/length(data.res$Species) # misclassification = 0.2

# Quality assessment
# misclassification setosa
1- table(data.res$Species, data.res$pred.iris)[1,1]/nrow(data.setosa) # 0.02 low misclassification of setosa
# misclassification of versicolor
1- table(data.res$Species, data.res$pred.iris)[2,2]/nrow(data.versicolor) # 0.28 high degree of misclassification
# misclassification of virginica
1- table(data.res$Species, data.res$pred.iris)[3,3]/nrow(data.virginica) # 0.3 worse degree of misclassification

library(MASS)
model.lda <- lda(as.factor(Species)~Sepal.Width + Sepal.Length, data = data)
pred.lda <- predict(model.lda)

table("Real"=data$Species, "Manual"=data.res$pred.iris)
table("Real"=data$Species, "LDA function"=matrix(pred.lda$class))

#The results are exactly the same results -> same error as the manual method
# The function uses the same assumptions as the manual method, which means that
# the results should be the same

# 4 Generate new data
# Determine amounts of each class - replace to keep the original probabilities
n <- sample(data$Species, 150, replace = TRUE)
t <- table(n)
n1 <- t[1]
n2 <- t[2]
n3 <- t[3]

library(mvtnorm)
set.seed(12345)
data.new.s <- rmvnorm(n=n1, mean = my.setosa, sigma = cov.pooled)
plot(x=data.new.s[,1], y=data.new.s[,2], xlim = c(4,8), ylim = c(1.5,5), main = "New values and original values", xlab = "Sepal Length", ylab = "Sepla Width")

set.seed(12345)
data.new.ve <- rmvnorm(n=n2, mean = my.versicolor, sigma = cov.pooled)
points(x=data.new.ve[,1], y=data.new.ve[,2], col="red")

set.seed(12345)
data.new.vi <- rmvnorm(n=n3, mean = my.virginica, sigma = cov.pooled)
points(x=data.new.vi[,1], y=data.new.vi[,2], col="green")

# original values are diamonds
points(x = data$Sepal.Length, y = data$Sepal.Width, col = data$Species, pch = 18)


# All groups are spread out more compared to the original data, but they seemed to be centered about the same way as the original data.
# Thus setos is a distinc group, while virginica and versicolor are blending into eachother from two directions.
# might be because pooled variance is used instead of each class's own variance (since they have different variance)

