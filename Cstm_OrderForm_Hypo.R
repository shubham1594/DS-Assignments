#importing data set
library(readr)
cof<-read_csv(file.choose())
View(cof)
attach(cof)

### Chi-square Test### (As we have to compare more than two populations with  each other)

stacked_cof<-stack(cof)# countries are in their own columns; so we need to stack the data 
attach(stacked_cof)
View(stacked_cof)
table(stacked_cof$ind,stacked_cof$values)
chisq.test(table(stacked_cof$ind,stacked_cof$values))
# p-value = 0.2771 > 0.05 so p high accept null hypothesis