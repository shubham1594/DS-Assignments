#importing data sert
library(readr)
LabTAT<-read_csv(file.choose())
View(LabTAT)
attach(LabTAT)

###Normality test###
shapiro.test(`Laboratory 1`)
# p-value = 0.5508 >0.05 so p high accept null Hypothesis => It follows normal distribution
shapiro.test(`Laboratory 2`)
# p-value = 0.8637 >0.05 so p high accept null Hypothesis => It follows normal distribution
shapiro.test(`Laboratory 3`)
# p-value = 0.4205 >0.05 so p high accept null Hypothesis => It follows normal distribution
shapiro.test(`Laboratory 4`)
# p-value = 0.6619 >0.05 so p high accept null Hypothesis => It follows normal distribution

###Variance test###
var.test(`Laboratory 1`,`Laboratory 2`)
# p-value = 0.1675 > 0.05 so p high accept null Hypothesis
# According to further variance test carried on other laboratory pairs, p-value > 0.05 for each pair

### ANOVA ### (as the result depends on four variables hence choosing ANOVA)
Stacked_Data <- stack(LabTAT)
View(Stacked_Data)
attach(Stacked_Data)
library(car)
leveneTest(values,ind,data= Stacked_Data)
Anova_results <- aov(values~ind,data = Stacked_Data)
summary(Anova_results)
# p-value = 0.05161 > 0.05 accept null hypothesis 