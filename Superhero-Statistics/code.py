# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data=pd.read_csv(path)
data['Gender'].replace('_','Agender',inplace=True)
gender_count=data['Gender'].value_counts()
#Code starts here 




# --------------
# #Code starts here
lis=data['Alignment'].unique()
alignment=data['Alignment'].value_counts()
plt.pie(alignment,labels=lis)


# --------------
#Code starts here
sc_df=data[['Strength','Combat']]
sc_covariance=sc_df['Strength'].cov(sc_df['Combat'])
sc_strength=sc_df['Strength'].std()
sc_combat=sc_df['Combat'].std()
sc_pearson=sc_covariance/(sc_strength*sc_combat)

ic_df=data[['Intelligence','Combat']]
ic_covariance=ic_df['Intelligence'].cov(ic_df['Combat'])
ic_intelligence=ic_df['Intelligence'].std()
ic_combat=ic_df['Combat'].std()
ic_pearson=ic_covariance/(ic_intelligence*ic_combat)


# --------------
#Code starts here
total_high=data['Total'].quantile(0.99)
super_best=data[data['Total']>total_high]
super_best_names=[super_best['Name']]



# --------------
#Code starts here
fig = plt.figure()

ax_1 = fig.add_subplot(131)
plt.boxplot(data['Intelligence'])
plt.title('Intelligence')

ax_2 = fig.add_subplot(132)
plt.boxplot(data['Speed'])
plt.title('Speed')

ax_3 = fig.add_subplot(133)
plt.boxplot(data['Power'])
plt.title('Power')


