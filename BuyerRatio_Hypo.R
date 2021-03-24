#importing data sert
library(readr)
Byr_Ratio<-read_csv(file.choose())
View(Byr_Ratio)
attach(Byr_Ratio)

###Normality test###
shapiro.test(Males)
# p-value = 0.3419 >0.05 so p high accept null hypothesis => It follows normal distribution
shapiro.test(Females)
# p-value = 0.2452 >0.05 so p high accept null hypothesis => It follows normal distribution

###Variance test###
var.test(Males,Females)#variance test
# p-value = 0.2285 > 0.05 so p high accept null hypothesis

###t.test###
t.test(Males,Females,alternative = "greater",var.equal = T)
# p-value = 0.9996 > 0.05 =>  accept null hypothesis

