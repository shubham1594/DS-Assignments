#importing data sert
library(readr)
cutlet<-read_csv(file.choose())
View(cutlet)
attach(cutlet)

###Normality test###
shapiro.test(cutlet$`Unit A`)
# p-value = 0.32 >0.05 so p high accept null Hypothesis => It follows normal distribution
shapiro.test(cutlet$`Unit B`)
# p-value = 0.5225 >0.05 so p high accept null Hypothesis => It follows normal distribution

###Variance test###
var.test(`Unit A`,`Unit B`)#variance test
# p-value = 0.3136 > 0.05 so p high accept null Hypothesis

# as the result depends on two variabes hence choosing 2 sample T Test
###2 sample T Test ### 
t.test(`Unit A`,`Unit B`,alternative = "two.sided",conf.level = 0.95,correct = TRUE)#two sample T.Test
# p-value = 0.4723 > 0.05 accept null Hypothesis 
