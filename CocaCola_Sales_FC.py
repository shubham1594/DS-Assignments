#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importiing dataset 
coc_s = pd.read_excel("CocaCola_Sales.xlsx")
quarter =['Q1','Q2','Q3','Q4'] 
p = coc_s["Quarter"][0]

coc_s['quarters']= 0

for i in range(42):
    p = coc_s["Quarter"][i]
    coc_s['quarters'][i]= p[0:3]

quarter_dummies = pd.DataFrame(pd.get_dummies(coc_s['quarters']))
coc_s= pd.concat([coc_s,quarter_dummies],axis = 1)

coc_s["t"] = np.arange(1,43)
coc_s["t_squared"] = coc_s["t"]*coc_s["t"]
coc_s["log_sales"] = np.log(coc_s["Sales"])
coc_s.columns

# dividing the dataset into train and test data
Train = coc_s.head(38)
Test = coc_s.tail(4)

plt.plot(coc_s.iloc[:,1])

# to change the index value in pandas data frame 
Test.set_index(np.arange(1,5),inplace=True)

### BUILDING VARIOUS FORECASTING MODELS AND IMPLEMENTIING ###

import statsmodels.formula.api as smf 

## L I N E A R ##
linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
#591.5532

## Exponential ##
Exp = smf.ols('log_sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
#466.2479

## Quadratic ##
Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
#475.5618

## Additive seasonality ##
add_sea = smf.ols('Sales~Q1_+Q2_+Q3_',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1_','Q2_','Q3_']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
#1860.0238

## Additive Seasonality Quadratic ##
add_sea_Quad = smf.ols('Sales~t+t_squared+Q1_+Q2_+Q3_',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1_','Q2_','Q3_','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
#301.7380

## Multiplicative Seasonality ##
Mul_sea = smf.ols('log_sales~Q1_+Q2_+Q3_',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
#1963.3896

##Multiplicative Additive Seasonality ##
Mul_Add_sea = smf.ols('log_sales~t+Q1_+Q2_+Q3_',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
#225.5243

## RMSE RESULTS FOR FORECASTING MODEL ##
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)

# RMSE of Multiplicative Additive Seasonality (rmse_Mult_add_sea) has the least value among the models prepared so far 
# So Multiplicative Additive Seasonality model can be used for FORECASTING



