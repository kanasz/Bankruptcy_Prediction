import pandas as pd
import numpy as np
df = pd.read_csv('./results/agriculture_RESULT.csv')
df.columns=['Sector','Year','Run','C','Weight','Number_Of_Features','Geom_Mean']
print(df.head())


c_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
weight_range = [50, 100, 200, 300, 400, 500, 600]
years = [13]


for c in c_range:
    for weight in weight_range:
        for year in years:
            df_result = df[(df['C']==c) & (df['Weight']==weight) & (df['Year']==year)]
            mean = (df_result['Geom_Mean'].mean())
            #means = np.append(means,[mean])
            print("c:{} w:{} y:{} gm:{}".format(c,weight,year,mean))
