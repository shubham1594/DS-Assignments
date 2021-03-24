#importing libraries
import pandas as pd 
import numpy as np
import seaborn as sns

# importing the data
stup = pd.read_csv("50_Startups.csv")
#R&D Spend:-RD, Administration:-Adm, Marketing Spend:-Mkt

type(stup)
stup.head()

# Correlation matrix 
stupr=stup.corr()
# There exists High collinearity between input variables, so there exists collinearity problem
 
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(stup.iloc[:,:])

# preparing MLR model considering all the variables 
import statsmodels.formula.api as smf # for regression model
## Preparing model                  
ml = smf.ols('Profit~RD+Adm+Mkt',data=stup).fit() # regression model
ml.params # Getting coefficients of variables               
ml.summary() #Adj. R-squared:0.948
pred = ml.predict(stup) # Predicted values using the model

# preparing model based only on RD
ml_r=smf.ols('Profit~RD',data = stup).fit()  
ml_r.summary()
# p-value <0.05 

# Preparing model based only on Adm
ml_a=smf.ols('Profit~Adm',data = stup).fit()  
ml_a.summary() 
# p-value >0.05 

# Preparing model based only on Mkt
ml_m=smf.ols('Profit~Mkt',data = stup).fit()  
ml_m.summary() 
# p-value <0.05  

# p-values for Adm is more than 0.05
# Checking whether data has any influential values using influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(ml)

# index 49,48 and 45 is showing high influence so we can exclude that entire row
# Studentized Residuals = Residual/standard deviation of residuals
stup.drop
stup_new = stup.drop(stup.index[[49,48,45]],axis=0) # ,inplace=False)
stup_new


## Preparing new model                  
ml_new = smf.ols('Profit~RD+Adm+Mkt',data = stup_new).fit()    
ml_new.params# Getting coefficients of variables        
ml_new.summary() #Adj. R-squared:0.962

profit_pred = ml_new.predict(stup_new) # Predicted values of Profit 

print(ml_new.conf_int(0.05)) # 99% confidence level

# calculating VIF's values of independent variables
rsq_r = smf.ols('RD~Adm+Mkt',data=stup_new).fit().rsquared  
vif_r = 1/(1-rsq_r) #2.1238

rsq_a = smf.ols('Adm~RD+Mkt',data=stup_new).fit().rsquared  
vif_a = 1/(1-rsq_a) # 1.1959

rsq_m = smf.ols('Mkt~RD+Adm',data=stup_new).fit().rsquared  
vif_m = 1/(1-rsq_m) #  2.1003

# Storing vif values in a data frame
d1 = {'Variables':['RD','Adm','Mkt'],'VIF':[vif_r,vif_a,vif_m]}
Vif_frame = pd.DataFrame(d1)  

# Added varible plot for new model
sm.graphics.plot_partregress_grid(ml_new)
# added varible plot for Adm is not showing any significance 

## Prepring final model
final_ml= smf.ols('Profit~RD+Mkt',data = stup_new).fit()
final_ml.params
final_ml.summary() #  Adj. R-squared:0.961
profit_pred = final_ml.predict(stup_new)
# As we can see that r-squared value has increased from 0.948 to 0.961.


#Evaluation of final model
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(stup_new.Profit,profit_pred))


# added variable plot for the final model
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(final_ml)
