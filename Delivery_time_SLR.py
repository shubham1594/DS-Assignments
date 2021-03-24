# importing libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# importing the dataset
Dl_tm=pd.read_csv("delivery_time.csv")
## Delivery Time= DT, Sorting Time=ST

#Dl_tm.corr()
Dl_tm.DT.corr(Dl_tm.ST) # correlation value between X and Y cor(y,x)
np.corrcoef(Dl_tm.DT,Dl_tm.ST)

## Building SLR model ##
import statsmodels.formula.api as smf

model=smf.ols("DT~ST",data=Dl_tm).fit()
model.params # line equation y = 6.582734+1.649020(ST)
model.summary()
print(model.conf_int(0.05)) # 99% confidence interval
pred = model.predict(Dl_tm) # Predicted values of DT using the model
pred.corr(Dl_tm.DT) # 82.59

# Visualization of regresion line over the scatter plot of ST and DT
plt.scatter(x=Dl_tm['ST'],y=Dl_tm['DT'],color='red');plt.plot(Dl_tm['ST'],pred,color='black');plt.xlabel('ST');plt.ylabel('DT')


## Transforming variables for accuracy ##
model2 = smf.ols('DT~np.log(ST)',data=Dl_tm).fit()
model2.params ## line equation y = 1.159684+9.043413(ST)
model2.summary()
print(model2.conf_int(0.05)) # 99% confidence level
pred2 = model2.predict(Dl_tm)
pred2.corr(Dl_tm.DT) #83.39
plt.scatter(x=Dl_tm['ST'],y=Dl_tm['DT'],color='green');plt.plot(Dl_tm['ST'],pred2,color='blue');plt.xlabel('ST');plt.ylabel('DT')


## Exponential transformation ##
model3 = smf.ols('np.log(DT)~ST',data=Dl_tm).fit()
model3.params  # line equation y = 2.121372 +0.105552(ST)
model3.summary()
print(model3.conf_int(0.05)) # 99% confidence level
pred_log = model3.predict(Dl_tm)
pred3=np.exp(pred_log)  # as we have used log(DT) in preparing model so we need to convert it back
pred3.corr(Dl_tm.DT) # 80.85
plt.scatter(x=Dl_tm['ST'],y=Dl_tm['DT'],color='green');plt.plot(Dl_tm.ST,np.exp(pred_log),color='blue');plt.xlabel('ST');plt.ylabel('DT')


# Considering the model having highest R-Squared value which is the log transformation - "model3"
# getting residuals of the entire data set
student_resid = model3.resid_pearson 
plt.plot(pred3,model3.resid_pearson,"o");plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred3,y=Dl_tm.DT);plt.xlabel("Predicted");plt.ylabel("Actual")

## Quadratic model ##
Dl_tm["ST_Sq"] = Dl_tm.ST*Dl_tm.ST
model_quad = smf.ols("DT~ST+ST_Sq",data=Dl_tm).fit()
model_quad.params  #line equation y =  3.522234 +2.813002(ST)-0.093198(ST_sq)
model_quad.summary()
pred_quad = model_quad.predict(Dl_tm)

model_quad.conf_int(0.05) # 99% confidence level
plt.scatter(Dl_tm.ST,Dl_tm.DT,c="b");plt.plot(Dl_tm.ST,pred_quad,"r")

plt.scatter(np.arange(21),model_quad.resid_pearson);plt.axhline(y=0,color='red');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

plt.hist(model_quad.resid_pearson) # histogram for residual values 

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(Dl_tm.DT,pred_quad))

