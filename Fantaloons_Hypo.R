#importing data set
library(readr)
Fantaloons<-read_csv(file.choose())
View(Fantaloons)
attach(Fantaloons)

### Proportional T Test ### (As we have to compare two populations with  each other)
table1 <- table(Weekdays,Weekend)
View(table1)
prop.test(x=c(66,47),n=c(167,120),conf.level = 0.95,correct = FALSE,alternative = "less")
# p-value = p-value = 0.5242 > 0.05 so p high accept null Hypothesis
prop.test(x=c(66,47),n=c(167,120),conf.level = 0.95,correct = FALSE,alternative = "greater")
# p-value = p-value = 0.4758 > 0.05 so p high accept null Hypothesis
prop.test(x=c(66,47),n=c(167,120),conf.level = 0.95,correct = FALSE,alternative = "two.sided")
# p-value = p-value = 0.9517 > 0.05 so p high accept null Hypothesis