#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importiing dataset 
plst_s = pd.read_csv("PlasticSales.csv")
months =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
p = plst_s["Month"][0]

plst_s['months']= 0

for i in range(60):
    p = plst_s["Month"][i]
    plst_s['months'][i]= p[0:3]

month_dummies = pd.DataFrame(pd.get_dummies(plst_s['months']))
plst_s= pd.concat([plst_s,month_dummies],axis = 1)

plst_s["t"] = np.arange(1,61)
plst_s["t_squared"] = plst_s["t"]*plst_s["t"]
plst_s["log_sales"] = np.log(plst_s["Sales"])
plst_s.columns

# dividing the dataset into train and test data
Train = plst_s.head(52)
Test = plst_s.tail(8)

plt.plot(plst_s.iloc[:,1])

# to change the index value in pandas data frame 
Test.set_index(np.arange(1,9),inplace=True)

### BUILDING VARIOUS FORECASTING MODELS AND IMPLEMENTIING ###

import statsmodels.formula.api as smf 

## L I N E A R ##
linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
#242.0224

## Exponential ##
Exp = smf.ols('log_sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
#244.9181

## Quadratic ##
Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
#259.0646

## Additive seasonality ##
add_sea = smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
#221.5208

## Additive Seasonality Quadratic ##
add_sea_Quad = smf.ols('Sales~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
#224.9209

## Multiplicative Seasonality ##
Mul_sea = smf.ols('log_sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
#225.8696

##Multiplicative Additive Seasonality ##
Mul_Add_sea = smf.ols('log_sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
#204.6104

## RMSE RESULTS FOR FORECASTING MODEL ##
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)

# RMSE of Multiplicative Additive Seasonality (rmse_Mult_add_sea) has the least value among the models prepared so far 
# So Multiplicative Additive Seasonality model can be used for FORECASTING



