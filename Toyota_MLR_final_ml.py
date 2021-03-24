#importing libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# importing the dataset
tyc = pd.read_csv("ToyotaCorolla.csv")
#Age_08_04:-Age, Doors:-Dr, Gears:-Gr, Quarterly_Tax:-QT, Weight:-Wt
type(tyc)
tyc.head()

# Correlation matrix 
tycr=tyc.corr()
# There is no collinearity between input variables
 
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(tyc.iloc[:,:])

                  
# preparing MLR model considering all the variables 
import statsmodels.formula.api as smf # for regression model
## Preparing model                  
ml = smf.ols('Price~Age+KM+HP+cc+Dr+Gr+QT+Wt',data=tyc).fit() # regression model
ml.params # Getting coefficients of variables               
ml.summary() #Adj. R-squared:0.863
pred = ml.predict(tyc) # Predicted values using the model

# preparing model based only on Age
ml_a=smf.ols('Price~Age',data = tyc).fit()  
ml_a.summary() #Adj. R-squared:0.768
# p-value <0.05 

# Preparing model based only on KM
ml_k=smf.ols('Price~KM',data = tyc).fit()  
ml_k.summary() #Adj. R-squared:0.020
# p-value >0.05 

# Preparing model based only on HP
ml_h=smf.ols('Price~HP',data = tyc).fit()  
ml_h.summary() #Adj. R-squared:0.550
# p-value <0.05  

# Preparing model based only on cc
ml_c=smf.ols('Price~cc',data = tyc).fit()  
ml_c.summary() #Adj. R-squared:0.550
# p-value <0.05  

# Preparing model based only on Dr
ml_d=smf.ols('Price~Dr',data = tyc).fit()  
ml_d.summary() #Adj. R-squared:0.550
# p-value <0.05  

# Preparing model based only on Gr
ml_g=smf.ols('Price~Gr',data = tyc).fit()  
ml_g.summary() #Adj. R-squared:0.550
# p-value <0.05  

# Preparing model based only on QT
ml_q=smf.ols('Price~QT',data = tyc).fit()  
ml_q.summary() #Adj. R-squared:0.550
# p-value <0.05  

# Preparing model based only on Wt
ml_w=smf.ols('Price~Wt',data = tyc).fit()  
ml_w.summary() #Adj. R-squared:0.550
# p-value <0.05  

# p-values for KM is more than 0.05
# Checking whether data has any influential values using influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(ml)

# index 80 is showing high influence so we can exclude that entire row
# Studentized Residuals = Residual/standard deviation of residuals
tyc.drop
tyc_new = tyc.drop(tyc.index[[80]],axis=0) 


## Preparing new model                  
ml_new = smf.ols('Price~Age+KM+HP+cc+Dr+Gr+QT+Wt',data = tyc_new).fit()    
ml_new.params# Getting coefficients of variables        
ml_new.summary() #Adj. R-squared:0.869
# Predicted values of Profit 
profit_pred = ml_new.predict(tyc_new)

# Confidence values 99%
print(ml_new.conf_int(0.05)) 

# calculating VIF's values of independent variables
rsq_a = smf.ols('Age~KM+HP+cc+Dr+Gr+QT+Wt',data=tyc_new).fit().rsquared  
vif_a = 1/(1-rsq_a) 

rsq_k = smf.ols('KM~Age+HP+cc+Dr+Gr+QT+Wt',data=tyc_new).fit().rsquared  
vif_k = 1/(1-rsq_k) 

rsq_h = smf.ols('HP~KM+Age+cc+Dr+Gr+QT+Wt',data=tyc_new).fit().rsquared  
vif_h = 1/(1-rsq_h) 

rsq_c = smf.ols('cc~KM+HP+Age+Dr+Gr+QT+Wt',data=tyc_new).fit().rsquared  
vif_c = 1/(1-rsq_c) 

rsq_d = smf.ols('Dr~KM+HP+cc+Age+Gr+QT+Wt',data=tyc_new).fit().rsquared  
vif_d = 1/(1-rsq_d) 

rsq_g = smf.ols('Gr~KM+HP+cc+Dr+Age+QT+Wt',data=tyc_new).fit().rsquared  
vif_g = 1/(1-rsq_g)

rsq_q = smf.ols('QT~KM+HP+cc+Dr+Gr+Age+Wt',data=tyc_new).fit().rsquared  
vif_q = 1/(1-rsq_q) 

rsq_w = smf.ols('Wt~KM+HP+cc+Dr+Gr+QT+Age',data=tyc_new).fit().rsquared  
vif_w = 1/(1-rsq_w) 

# Storing vif values in a data frame
d1 = {'Variables':['Age','KM','HP','cc','Dr','Gr','QT','Wt'],'VIF':[vif_a,vif_k,vif_h,vif_c,vif_d,vif_g,vif_q,vif_w]}
Vif_frame = pd.DataFrame(d1)  

# Added varible plot for new model
sm.graphics.plot_partregress_grid(ml_new)
# added varible plot for Gr is not showing any significance 

## Prepring final model
final_ml= smf.ols('Price~Age+KM+HP+cc+Dr+Gr+QT',data = tyc_new).fit()
final_ml.params
final_ml.summary() #  Adj. R-squared:0.961
profit_pred = final_ml.predict(tyc_new)
# As we can see that r-squared value has increased from 0.948 to 0.961.


#Evaluation of final model
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(tyc_new.Profit,profit_pred))


# added variable plot for the final model
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(final_ml)
