# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df=pd.read_csv(path)
total=len(df)
p_a=len(df[df['fico']>700])/total
p_b=len(df[df['purpose']=='debt_consolidation'])/total
df1=df[df['purpose']=='debt_consolidation']
total1=len(df1)
p_a_b=len(df1[(df1['purpose']=='debt_consolidation') & (df1['fico']>700)])/total1
p_b_a=(p_a_b * p_a)/p_b
result=p_b_a==p_a
result
# code ends here


# --------------
# code starts here
prob_lp=len(df[df['paid.back.loan']=='Yes'])/total
prob_cs=len(df[df['credit.policy']=='Yes'])/total
new_df=df[df['paid.back.loan']=='Yes']
tot=len(new_df)
prob_pd_cs=len(new_df[(new_df['paid.back.loan']=='Yes') & (new_df['credit.policy']=='Yes')])/tot
bayes=(prob_pd_cs*prob_lp)/prob_cs

# code ends here


# --------------
# code starts here
data=df['purpose'].value_counts()
data.plot(kind='bar')
df1=df[df['paid.back.loan']=='No']
df1.plot(kind='bar')

# code ends here


# --------------
# code starts here
inst_median=df['installment'].median()
inst_mean=df['installment'].mean()
plt.hist(df['installment'])
plt.hist(df['log.annual.inc'])
# code ends here


