#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importiing dataset 
air_d = pd.read_csv("Airlines+Data.csv")
month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
p = air_d["Month"][0]

air_d['months']= 0

for i in range(96):
    p = air_d["Month"][i]
    air_d['months'][i]= p[0:3]

month_dummies = pd.DataFrame(pd.get_dummies(air_d['months']))
air_d= pd.concat([air_d,month_dummies],axis = 1)

air_d["t"] = np.arange(1,97)
air_d["t_squared"] = air_d["t"]*air_d["t"]
air_d["log_Psgr"] = np.log(air_d["Passengers"])
air_d.columns

# dividing the dataset into train and test data
Train = air_d.head(86)
Test = air_d.tail(10)

plt.plot(air_d.iloc[:,1])

# to change the index value in pandas data frame 
Test.set_index(np.arange(1,11),inplace=True)

### BUILDING VARIOUS FORECASTING MODELS AND IMPLEMENTIING ###

import statsmodels.formula.api as smf 

## L I N E A R ##
linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
#58.6431

## Exponential ##
Exp = smf.ols('log_Psgr~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
#49.9031

## Quadratic ##
Quad = smf.ols('Passengers~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
#53.9143

## Additive seasonality ##
add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
#136.7901

## Additive Seasonality Quadratic ##
add_sea_Quad = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
#29.1045

## Multiplicative Seasonality ##
Mul_sea = smf.ols('log_Psgr~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
#144.3849

##Multiplicative Additive Seasonality ##
Mul_Add_sea = smf.ols('log_Psgr~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
#11.2649

## RMSE RESULTS FOR FORECASTING MODEL ##
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)

# RMSE of Multiplicative Additive Seasonality (rmse_Mult_add_sea) has the least value among the models prepared so far 
# So Multiplicative Additive Seasonality model can be used for FORECASTING


