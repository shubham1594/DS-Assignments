# importing libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# importing the dataset
Sl_d=pd.read_csv("Salary_Data.csv")
## Years Experience= YE, Salary= Sal

#Sl_d.corr()
Sl_d.YE.corr(Sl_d.Sal) # correlation value between X and Y cor(y,x)
np.corrcoef(Sl_d.YE,Sl_d.Sal)

## Building SLR model ##
import statsmodels.formula.api as smf

model1=smf.ols("YE~Sal",data=Sl_d).fit()
model1.params # line equation y = -2.383161+0.000101(Sal)
model1.summary()
print(model1.conf_int(0.05)) # 99% confidence interval
pred1 = model1.predict(Sl_d) # Predicted values of DT using the model
pred1.corr(Sl_d.YE) # 97.82

# Visualization of regresion line over the scatter plot of Sal and YE
plt.scatter(x=Sl_d['Sal'],y=Sl_d['YE'],color='red');plt.plot(Sl_d['Sal'],pred1,color='black');plt.xlabel('Sal');plt.ylabel('YE')


## Transforming variables for accuracy ##
model2 = smf.ols('YE~np.log(Sal)',data=Sl_d).fit()
model2.params ## line equation y = -77.696132+7.428821(Sal)
model2.summary()
print(model2.conf_int(0.05)) # 99% confidence level
pred2 = model2.predict(Sl_d)
pred2.corr(Sl_d.YE) #96.53
plt.scatter(x=Sl_d['Sal'],y=Sl_d['YE'],color='green');plt.plot(Sl_d['Sal'],pred2,color='blue');plt.xlabel('Sal');plt.ylabel('YE')


## Exponential transformation ##
model3 = smf.ols('np.log(YE)~Sal',data=Sl_d).fit()
model3.params  # line equation y = -0.094207 +0.000021(Sal)
model3.summary()
print(model3.conf_int(0.05)) # 99% confidence level
pred_log = model3.predict(Sl_d)
pred3=np.exp(pred_log)  # as we have used log(YE) in preparing model so we need to convert it back
pred3.corr(Sl_d.YE) # 96.38
plt.scatter(x=Sl_d['Sal'],y=Sl_d['YE'],color='green');plt.plot(Sl_d.Sal,np.exp(pred_log),color='blue');plt.xlabel('Sal');plt.ylabel('YE')


# Considering the model having highest R-Squared value which is the transformation - model1
# getting residuals of the entire data set
student_resid = model1.resid_pearson 
plt.plot(pred1,model1.resid_pearson,"o");plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred1,y=Sl_d.YE);plt.xlabel("Predicted");plt.ylabel("Actual")

## Quadratic model ##
Sl_d["Sal_Sq"] = Sl_d.Sal*Sl_d.Sal
model_quad = smf.ols("YE~Sal+Sal_Sq",data=Sl_d).fit()
model_quad.params  #line equation y =  -1.715993e+00 + 8.254010e-05(Sal)-1.162596e-10(Sal_sq)
model_quad.summary()
pred_quad = model_quad.predict(Sl_d)

model_quad.conf_int(0.05) # 99% confidence level
plt.scatter(Sl_d.Sal,Sl_d.YE,c="b");plt.plot(Sl_d.Sal,pred_quad,"r")

plt.scatter(np.arange(30),model_quad.resid_pearson);plt.axhline(y=0,color='red');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

plt.hist(model_quad.resid_pearson) # histogram for residual values 

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(Sl_d.YE,pred_quad))

