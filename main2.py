import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt


# df.describe() 看数据
#plot_data=df.copy()
#plt.plot(plot_data['ds'],plot_data['y'],'.')
#plot_data=df[(df['ds']>='')&(df['ds']<='')]  看一段数据
#plt.plot(plot_data['ds'],plot_data['y'],'.-')
#plt.grid(axis='x')
#ax=plt,gca()
#x_major_locator=plt.MultipleLocator(7)
#ax.xaxis.set_major_locator(x_major_locator) 自己看看周期
#plt.xticks(rotation=90)


#分段： from prophet.plt import add_changepoints_to_plot  a=add_changepoints_to_plot(fig.gca(),m,forecast) 看分段点
#m.add_seasonality(name='',period=30, fourier_order=3, prior_scale=0.1) 自己加一个季节性
#m.add_country_holidays(country_name='') prophet自己的节假日函数
#名字=pd.DataFrame=[{'holiday':'',  'ds':pd.to_datatime(['日期'，，，，，，])，'lower_window'=0,'upper_window'=1}] holidays=pd.concat() holidays=holidays
#df_new=df.copy()  df_new.loc[(df['ds']>='日期')&(df['ds']<='日期'),'y']=np.nan 处理异常值


df=pd.read_csv('E:\\Python 000\\ddd\\test.csv')
df.head()
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
plt(m, forecast)
plt(m, forecast)


#趋势性：growth='linear'(线性分段) ‘logistic’df['cap']=上限，df['floor']=下限，训练和future都要写上  （线性不分段） ‘flat’（无）
#changepoint_range=0.8   0~1默认0.8,最好用[0.8,0.95]
#n_changepoints=25  整数，最好不调   （changepoints=['日期 ','日期']用此参数忽略前两个）
#changepoint_prior_scale=0.05 波动程度[0.001,0.5]最好
#interval_width=0.8  0~1 置信区间
#yearly_seasonality=10 weekly与daily同理 0或False则不拟合
#seasonality_prior_scale=10 [0.01,10]
#seasonality_mode='additive'(+) 'multiplicative'(*)
#mcmc_samples=10 季节性置信区间，最好还是算了吧
#holidays_prior_scale=0.1 [0.1,10]


# from prophet.diagnostics import cross_validation   模型效果验证
# df_cv=cross_validation(m,initial= ,period= ,horizon= )
# from prophet.diagnostics import performance_metrics
# df_p=performance_metrics(df_cv)
# df_p.head()
# from prophet.plot import plot_cross_validation_metric
# fig=plot_cross_validation_metric(df_cv,metric='mape')
