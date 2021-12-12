set.seed(1234567890)
library(geosphere)
stations <- read.csv("stations.csv")
temps <- read.csv("temps50k.csv")
st <- merge(stations,temps,by="station_number")

h_distance <- 150000 # These three values are up to the students
h_date <- 10
h_time <- 2

a <- 58.3836  # The point to predict (up to the students) 58.3836  14.826
b <- 14.826
date <- "2013-02-04" # The date to predict (up to the students) 
times <- c("04:00:00", "06:00:00", "08:00:00", "10:00:00", "12:00:00", "14:00:00", "16:00:00", "18:00:00", "20:00:00", "22:00:00", "24:00:00")
temp <- vector(length=length(times))

# Help functions
get.date.dist <- function(old.date, new.date){
  d <- NULL
  # Old month
  o.month <- as.matrix(as.numeric(strftime(old.date,"%m")))
  o.day <- as.matrix(as.numeric(strftime(old.date, "%d")))
  # New month
  n.month <- as.numeric(strftime(new.date,"%m"))
  n.day <- as.numeric(strftime(date, "%d"))
  
  #Calculate distance
  for(i in 1:length(old.date)) {
    if(o.month[i]==n.month) {
      d = c(d,abs(o.day[i]-n.day))
    } else {
      #distance months and days - assumes 31 days each month
      # Direction 1
      d1.month <- ifelse(n.month-o.month[i]<0, (n.month-o.month[i])%%12, n.month-o.month[i])
      d1.day <- (31-o.day[i]+n.day) 
      d1 <- (d1.month-1)*31+d1.day
      
      # Direction 2
      d2.month <- ifelse(o.month[i]-n.month<0, (o.month[i]-n.month)%%12, o.month[i]-n.month)
      d2.day <- (31-n.day+o.day[i]) 
      d2 <- (d2.month-1)*31+d2.day
      
      d=c(d,min(d1,d2))
    }
  }
  
  return (d)
}

gaus.kernel.p <- function(x) {
  return(exp(-(norm(x/h_distance, "2"))^2))
}

gaus.kernel.d <- function(x) {
  return(exp(-(norm(x/h_date, "2"))^2))
}

gaus.kernel.t <- function(x) {
  return(exp(-(norm(x/h_time, "2"))^2))
}


#filter out values after the chosen date and convert to date 
st_filtered <- st[as.Date(st$date) < as.Date(date),]
st_filtered[9] <- as.Date(st_filtered[,9])

#physical distance in meter
target.loc <- c(a,b)
p.dist <- distHaversine(st_filtered[4:5], target.loc)
k.p <- as.matrix(sapply(p.dist, gaus.kernel.p)) #exp(-norm(x/h_distance, "2")^2)

              
# date distance in days - year is irrelevant for distance
date <- as.Date(date)
d.dist <- get.date.dist(st_filtered[,9], date)
k.d <- as.matrix(sapply(d.dist, gaus.kernel.d)) #exp(-norm(x/h_date, "2")^2)



# Control for kernel values physical distance and distance in days 
plot(p.dist, k.p, xlab = "Distance in meter")
plot(d.dist, k.d, xlab = "Distance in days")


# The temperature = (sum((k1+k2+k3)*t))/sum(k1+k2+k3) - sapply(time, function(x) difftime(n.time,x))

k.tot <- k.p + k.d
time <- strptime(st_filtered[,10], "%H:%M:%S")

#04:00:00
t <- NULL
n.time <- strptime(times[1], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j])))
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot + k.t
temp[1] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[1]


#06:00:00
t <- NULL
n.time <- strptime(times[2], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot + k.t
temp[2] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[2]



#08:00:00
t <- NULL
n.time <- strptime(times[3], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot + k.t
temp[3] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[3]




#10:00:00
t <- NULL
n.time <- strptime(times[4], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot + k.t
temp[4] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[4]




#12:00:00
t <- NULL
n.time <- strptime(times[5], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot + k.t
temp[5] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[5]


#14:00:00
t <- NULL
n.time <- strptime(times[6], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot + k.t
temp[6] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[6]



#16:00:00
t <- NULL
n.time <- strptime(times[7], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot + k.t
temp[7] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[7]


#18:00:00
t <- NULL
n.time <- strptime(times[8], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot + k.t
temp[8] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[8]


#20:00:00
t <- NULL
n.time <- strptime(times[9], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot + k.t
temp[9] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[9]


#22:00:00
t <- NULL
n.time <- strptime(times[10], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot + k.t
temp[10] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[10]


#00:00:00
t <- NULL
n.time <- strptime(times[11], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot + k.t
temp[11] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[11]


temp

# Students? code here
plot(seq(4,24, 2), temp, type="o", main = "Day temperature using sum kernel", xlab ="Time in hours")






# The temperature = (sum((k1*k2*k3)*t))/sum(k1*k2*k3)
# Multiplication makes the weight of perfect targets disappear
# Thus the overrepresentation of physical place is reduced/removed

k.tot <- k.p * k.d
time <- strptime(st_filtered[,10], "%H:%M:%S")



#04:00:00
t <- NULL
n.time <- strptime(times[1], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j])))
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot * k.t
temp[1] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[1]


#06:00:00
t <- NULL
n.time <- strptime(times[2], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot * k.t
temp[2] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[2]



#08:00:00
t <- NULL
n.time <- strptime(times[3], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot * k.t
temp[3] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[3]




#10:00:00
t <- NULL
n.time <- strptime(times[4], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot * k.t
temp[4] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[4]




#12:00:00
t <- NULL
n.time <- strptime(times[5], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot * k.t
temp[5] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[5]


#14:00:00
t <- NULL
n.time <- strptime(times[6], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot * k.t
temp[6] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[6]



#16:00:00
t <- NULL
n.time <- strptime(times[7], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot * k.t
temp[7] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[7]


#18:00:00
t <- NULL
n.time <- strptime(times[8], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot * k.t
temp[8] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[8]


#20:00:00
t <- NULL
n.time <- strptime(times[9], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot * k.t
temp[9] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[9]


#22:00:00
t <- NULL
n.time <- strptime(times[10], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot * k.t
temp[10] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[10]


#00:00:00
t <- NULL
n.time <- strptime(times[11], "%H:%M:%S")
for(j in 1:length(time)) {
  t <- c(t, abs(difftime(n.time,time[j]))) 
}
t <- ifelse(t>12, 12-t%%12, t)
k.t <- as.matrix(sapply(t, gaus.kernel.t))

# Predicted temperature for time
k.all <- k.tot * k.t
temp[11] <- sum(k.all*st_filtered[,11])/sum(k.all)
temp[11]


temp


plot(seq(4,24, 2),temp, type="o", main="Day temperature using multiplication kernel", xlab = "Time in hours")


#Control of h_t: for 24:00
plot(t, k.t, main="Kernel hour value to distance in hour for 24:00", xlab ="Distance in hours")





#-----------------------------------------------------
# Control Linköping

Linköping <- which(st_filtered[2]=="Linköping" )
Linköping <- st_filtered[Linköping,]

plot(row.names(Linköping), Linköping$air_temperature, xlab ="Data point id", ylab="Temperature", main ="Temperature in Linköping")

#-----------------------------------------------------




