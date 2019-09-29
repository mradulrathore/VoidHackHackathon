# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""import csv1 file"""
dataset=pd.read_csv('csv1.csv')
x=dataset.iloc[:, :].values
x = np.delete(x, (9), axis=0)
x = np.delete(x, (9), axis=0)



from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(x[:, 1:11])
 
SimpleImputer(copy=True, fill_value=None, missing_values='NaN',
strategy='mean', verbose=0)

x[:, 1:11]=imp_mean.transform(x[:, 1:11])


"""importing csv2 file"""
dataset1=pd.read_csv('csv2.csv')
y=dataset1.iloc[:, :].values
imp_mean1 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean1.fit(y[:, 1:11])
 
SimpleImputer(copy=True, fill_value=None, missing_values='NaN',
strategy='mean', verbose=0)

y[:, 1:11]=imp_mean1.transform(y[:, 1:11])


pd.DataFrame(x).to_csv("out1.csv")
"""
myCsvRow=['2010',x["1"].mean(),x["2"].mean(),x["3"].mean(),x["4"].mean(),x["5"].mean(),x["6"].mean(),x["7"].mean(),x["8"].mean(),x["9"].mean()]
with open('out1.csv','a') as fd:
    fd.write(myCsvRow)
    """
pd.DataFrame(y).to_csv("out2.csv")


from pandas import DataFrame

df = pd.DataFrame(x)

df.columns = ["0","1","2","3","4","5","6","7","8","9","10"]
print df
df=df.drop('0',1)
print df


dfmd = pd.DataFrame(x)

dfmd.columns = ["0","1","2","3","4","5","6","7","8","9","10"]
print dfmd
dfmd=dfmd.drop('0',1)
dfmd=dfmd.drop('1',1)
dfmd=dfmd.drop('3',1)
dfmd=dfmd.drop('5',1)
dfmd=dfmd.drop('7',1)
dfmd=dfmd.drop('9',1)

print dfmd

dfmd.sort()


pd.DataFrame(dfmd.idxmax(axis=1)).to_csv("death.csv")
death=pd.read_csv('death.csv')
deatharray=death.iloc[:, :].values
j=2
for i in range(9):
    if deatharray[i][1]==2:
            deathname="Acute Diarrhoeal Disease"
    elif deatharray[i][1]==4:
            deathname="Malaria"
    elif deatharray[i][1]==6:
            deathname="Acute Respiratory Infection"
    elif deatharray[i][1]==8:
            deathname="Japanese Encephalitis"
    elif deatharray[i][1]==10:
            deathname="Viral Hepatitis"
    print deathname
    
    
"""death of years prediction"""
    
dfmds = pd.DataFrame(y)

dfmds.columns = ["0","1","2","3","4","5","6","7","8","9","10"]
print dfmds
dfmds=dfmds.drop('0',1)
dfmds=dfmds.drop('1',1)
dfmds=dfmds.drop('3',1)
dfmds=dfmds.drop('5',1)
dfmds=dfmds.drop('7',1)
dfmds=dfmds.drop('9',1)

print dfmds

dfmds.sort()


pd.DataFrame(dfmds.idxmax(axis=1)).to_csv("deathstate.csv")
death=pd.read_csv('deathstate.csv')
deatharraystate=death.iloc[:, :].values
j=2
for i in range(34):
     if deatharraystate[i][1]==2:
            deathname="Acute Diarrhoeal Disease"
     elif deatharraystate[i][1]==4:
            deathname="Malaria"
     elif deatharraystate[i][1]==6:
            deathname="Acute Respiratory Infection"
     elif deatharraystate[i][1]==8:
            deathname="Japanese Encephalitis"
     elif deatharraystate[i][1]==10:
            deathname="Viral Hepatitis"
     print deathname
    
    
"""Complete death max"""
"""
maxx=0;j=0
import csv
for i in range(5):
    percentarray=((x[10][i+2]/x[10][i+1])*100)-((x[0][i+2]/x[0][i+1])*100)
    if abs(percentarray)>maxx:
        j=i
    row = [i, percentarray]
    with open('percentagepre.csv', 'a') as csvFile:
        writer= csv.writer(csvFile)
        writer.writerow(row)
csvFile.close()
print j
"""

plt.plot((x[:,0]),x[:,2])
plt.plot((x[:,0]),x[:,4])
plt.plot((x[:,0]),x[:,6])
plt.plot((x[:,0]),x[:,8])
plt.plot((x[:,0]),x[:,10])
plt.ylabel("Deaths")
plt.xlabel("Year")

plt.savefig('death.png')
plt.show()
fig = plt.figure()

####

yeararray = np.arange(11)

###LINEAR REGRESSION###

DATA=pd.read_csv('out1.csv')
print (DATA.shape)
DATA.head()
xi=DATA['0'].values
print xi
mean_xi=np.mean(xi)
n=len(xi)
max_xi=np.max(xi)+100
min_xi=np.min(xi)-100
xii=np.linspace(min_xi,max_xi,1000)
print mean_xi

yi=DATA['1'].values

mean_yi=np.mean(yi)

numer=0
denum=0
i=0

for i in range(n):
    numer+=(xi[i]-mean_xi)*(yi[i]-mean_yi)
    denum+=(xi[i]-mean_xi)**2
print numer,denum
m=numer/denum
c=mean_yi-(m*mean_xi)
print (m,c)

yii=c+m*xii
plt.plot(xii,yii,color='red',label='regressionline')
plt.scatter(xii,yii,color='black',label='scatterplot')
plt.xlabel('Year')
plt.ylabel('Cases')
plt.legend()
plt.show()

ss_t=0
ss_r=0
i=0
for i in range(n):
    y_pred=c+m*xi[i]
    ss_t+=(yi[i]-mean_yi)**2
    ss_r+=(yi[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print (r2)

yeararray[1]=c+m*mean_xi

##LR Column-1####


yi=DATA['2'].values

mean_yi=np.mean(yi)

numer=0
denum=0
i=0
for i in range(n):
    numer+=(xi[i]-mean_xi)*(yi[i]-mean_yi)
    denum+=(xi[i]-mean_xi)**2
print numer,denum
m=numer/denum
c=mean_yi-(m*mean_xi)
print (m,c)

yii=c+m*xii
plt.plot(xii,yii,color='red',label='regressionline')
plt.scatter(xii,yii,color='black',label='scatterplot')
plt.xlabel('Year')
plt.ylabel('Cases')
plt.legend()
plt.show()

ss_t=0
ss_r=0
i=0
for i in range(n):
    y_pred=c+m*xi[i]
    ss_t+=(yi[i]-mean_yi)**2
    ss_r+=(yi[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print (r2)

yeararray[2]=c+m*mean_xi

##LR Column-2####


yi=DATA['3'].values

mean_yi=np.mean(yi)

numer=0
denum=0
i=0
for i in range(n):
    numer+=(xi[i]-mean_xi)*(yi[i]-mean_yi)
    denum+=(xi[i]-mean_xi)**2
print numer,denum
m=numer/denum
c=mean_yi-(m*mean_xi)
print (m,c)

yii=c+m*xii
plt.plot(xii,yii,color='red',label='regressionline')
plt.scatter(xii,yii,color='black',label='scatterplot')
plt.xlabel('Year')
plt.ylabel('Cases')
plt.legend()
plt.show()

ss_t=0
ss_r=0
i=0
for i in range(n):
    y_pred=c+m*xi[i]
    ss_t+=(yi[i]-mean_yi)**2
    ss_r+=(yi[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print (r2)
yeararray[3]=c+m*mean_xi

##LR Column-3####


yi=DATA['4'].values

mean_yi=np.mean(yi)

numer=0
denum=0
i=0
for i in range(n):
    numer+=(xi[i]-mean_xi)*(yi[i]-mean_yi)
    denum+=(xi[i]-mean_xi)**2
print numer,denum
m=numer/denum
c=mean_yi-(m*mean_xi)
print (m,c)

yii=c+m*xii
plt.plot(xii,yii,color='red',label='regressionline')
plt.scatter(xii,yii,color='black',label='scatterplot')
plt.xlabel('Year')
plt.ylabel('Cases')
plt.legend()
plt.show()

ss_t=0
ss_r=0
i=0
for i in range(n):
    y_pred=c+m*xi[i]
    ss_t+=(yi[i]-mean_yi)**2
    ss_r+=(yi[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print (r2)

yeararray[4]=c+m*mean_xi

##LR Column-4####


yi=DATA['5'].values

mean_yi=np.mean(yi)

numer=0
denum=0
i=0
for i in range(n):
    numer+=(xi[i]-mean_xi)*(yi[i]-mean_yi)
    denum+=(xi[i]-mean_xi)**2
print numer,denum
m=numer/denum
c=mean_yi-(m*mean_xi)
print (m,c)

yii=c+m*xii
plt.plot(xii,yii,color='red',label='regressionline')
plt.scatter(xii,yii,color='black',label='scatterplot')
plt.xlabel('Year')
plt.ylabel('Cases')
plt.legend()
plt.show()

ss_t=0
ss_r=0
i=0
for i in range(n):
    y_pred=c+m*xi[i]
    ss_t+=(yi[i]-mean_yi)**2
    ss_r+=(yi[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print (r2)
yeararray[5]=c+m*mean_xi


##LR Column-5####


yi=DATA['6'].values

mean_yi=np.mean(yi)

numer=0
denum=0
i=0
for i in range(n):
    numer+=(xi[i]-mean_xi)*(yi[i]-mean_yi)
    denum+=(xi[i]-mean_xi)**2
print numer,denum
m=numer/denum
c=mean_yi-(m*mean_xi)
print (m,c)

yii=c+m*xii
plt.plot(xii,yii,color='red',label='regressionline')
plt.scatter(xii,yii,color='black',label='scatterplot')
plt.xlabel('Year')
plt.ylabel('Cases')
plt.legend()
plt.show()

ss_t=0
ss_r=0
i=0
for i in range(n):
    y_pred=c+m*xi[i]
    ss_t+=(yi[i]-mean_yi)**2
    ss_r+=(yi[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print (r2)

yeararray[6]=c+m*mean_xi

##LR Column-6####


yi=DATA['7'].values

mean_yi=np.mean(yi)

numer=0
denum=0
i=0
for i in range(n):
    numer+=(xi[i]-mean_xi)*(yi[i]-mean_yi)
    denum+=(xi[i]-mean_xi)**2
print numer,denum
m=numer/denum
c=mean_yi-(m*mean_xi)
print (m,c)

yii=c+m*xii
plt.plot(xii,yii,color='red',label='regressionline')
plt.scatter(xii,yii,color='black',label='scatterplot')
plt.xlabel('Year')
plt.ylabel('Cases')
plt.legend()
plt.show()

ss_t=0
ss_r=0
i=0
for i in range(n):
    y_pred=c+m*xi[i]
    ss_t+=(yi[i]-mean_yi)**2
    ss_r+=(yi[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print (r2)
yeararray[7]=c+m*mean_xi


##LR Column-7####


yi=DATA['8'].values

mean_yi=np.mean(yi)

numer=0
denum=0
i=0
for i in range(n):
    numer+=(xi[i]-mean_xi)*(yi[i]-mean_yi)
    denum+=(xi[i]-mean_xi)**2
print numer,denum
m=numer/denum
c=mean_yi-(m*mean_xi)
print (m,c)

yii=c+m*xii
plt.plot(xii,yii,color='red',label='regressionline')
plt.scatter(xii,yii,color='black',label='scatterplot')
plt.xlabel('Year')
plt.ylabel('Cases')
plt.legend()
plt.show()

ss_t=0
ss_r=0
i=0
for i in range(n):
    y_pred=c+m*xi[i]
    ss_t+=(yi[i]-mean_yi)**2
    ss_r+=(yi[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print (r2)

yeararray[8]=c+m*mean_xi


##LR Column-8####


yi=DATA['9'].values

mean_yi=np.mean(yi)

numer=0
denum=0
i=0
for i in range(n):
    numer+=(xi[i]-mean_xi)*(yi[i]-mean_yi)
    denum+=(xi[i]-mean_xi)**2
print numer,denum
m=numer/denum
c=mean_yi-(m*mean_xi)
print (m,c)

yii=c+m*xii
plt.plot(xii,yii,color='red',label='regressionline')
plt.scatter(xii,yii,color='black',label='scatterplot')
plt.xlabel('Year')
plt.ylabel('Cases')
plt.legend()
plt.show()

ss_t=0
ss_r=0
i=0
for i in range(n):
    y_pred=c+m*xi[i]
    ss_t+=(yi[i]-mean_yi)**2
    ss_r+=(yi[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print (r2)

yeararray[9]=c+m*mean_xi


##LR Column-9####


yi=DATA['10'].values

mean_yi=np.mean(yi)

numer=0
denum=0
i=0
for i in range(n):
    numer+=(xi[i]-mean_xi)*(yi[i]-mean_yi)
    denum+=(xi[i]-mean_xi)**2
print numer,denum
m=numer/denum
c=mean_yi-(m*mean_xi)
print (m,c)

yii=c+m*xii
plt.plot(xii,yii,color='red',label='regressionline')
plt.scatter(xii,yii,color='black',label='scatterplot')
plt.xlabel('Year')
plt.ylabel('Cases')
plt.legend()
plt.show()

ss_t=0
ss_r=0
i=0
for i in range(n):
    y_pred=c+m*xi[i]
    ss_t+=(yi[i]-mean_yi)**2
    ss_r+=(yi[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print (r2)
yeararray[10]=c+m*mean_xi


##LR Column-10###

print (yeararray)