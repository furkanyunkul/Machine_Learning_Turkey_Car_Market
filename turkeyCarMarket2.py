import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
#Genek Ozellikler
#------------------------------------------------------------------------------
dataset = pd.read_csv('turkey_car_market.csv')
print('shape')
print(dataset.shape)
print('head')
print(dataset.head())
print('tail')
print(dataset.tail())
print('describe')
print(dataset.describe())
print('info')
print(dataset.info())
#dataset['Model Yıl'] = dataset['Model Yıl'].astype('int64')

degisken=dataset.columns
satirSayisi=len(dataset)
#Sadece 2.el fiyat tahmin edilecegi icin 
dataset = dataset.loc[dataset["Durum"] == "2. El"]

print("İkinci el")

print('head')
print(dataset.head())
print('tail')
print(dataset.tail())
print('describe')
print(dataset.describe())
print('info')
print(dataset.info())

print(dataset['Fiyat'].describe())




#İlk 10 deger
#------------------------------------------------------------------------------
print('Marka en cok 10')
print(dataset['Marka'].value_counts().head(10))

print('Arac Tip Grubu en cok 10')
print(dataset['Arac Tip Grubu'].value_counts().head(10))

print('Arac Tip en cok 10')
print(dataset['Arac Tip'].value_counts().head(10))

print('Model Yıl en cok 10')
print(dataset['Model Yıl'].value_counts().head(10))

print('Yakıt Türü en cok 10')
print(dataset['Yakıt Turu'].value_counts().head(10))

print('Vites en cok 10')
print(dataset['Vites'].value_counts().head(10))

print('CCM en cok 10')
print(dataset['CCM'].value_counts().head(10))

print('Beygir Gucu en cok 10')
print(dataset['Beygir Gucu'].value_counts().head(10))

print('Renk en cok 10')
print(dataset['Renk'].value_counts().head(10))

print('Kasa Tipi en cok 10')
print(dataset['Kasa Tipi'].value_counts().head(10))

print('Kimden en cok 10')
print(dataset['Kimden'].value_counts().head(10))

print('Durum en cok 10')
print(dataset['Durum'].value_counts().head(10))

print('Km en cok 10')
print(dataset['Km'].value_counts().head(10))

print('Fiyat en cok 10')
print(dataset['Fiyat'].value_counts().head(10))

#Null ve bilinmeyen degerler
#------------------------------------------------------------------------------

print("Degiskenler")
print(degisken)

#Null data olup olmadigi kontrol gene dataya orani
print(dataset.isnull().sum()/len(dataset))

print('Bilinmeyen Sayisi')
for i in degisken:
    print(i,len(dataset.loc[dataset[i] == "Bilmiyorum"]) ,len(dataset.loc[dataset[i] == "Bilmiyorum"])/len(dataset))
    
print('Bilinmeyen Sayisi')
for i in degisken:
    print(i,len(dataset.loc[dataset[i] == "-"]) ,len(dataset.loc[dataset[i] == "-"])/len(dataset))

beygirGucu=dataset.loc[dataset['Beygir Gucu']!='Bilmiyorum']
print(beygirGucu.head())

q1 = dataset["Fiyat"].quantile(0.25)
q3 = dataset["Fiyat"].quantile(0.75)
IOC = q3 - q1
print(IOC)
altSinir = q1 - 1.5*IOC
ustSinir = q3 + 1.5*IOC


kontrol = (dataset["Fiyat"] < altSinir) | (dataset["Fiyat"] > ustSinir)
dataset["Asırı_Deger"] = kontrol
print(dataset["Asırı_Deger"].value_counts())

datasetAsiriFiyatFark = dataset.loc[dataset["Asırı_Deger"] == True]

#pivot
print(round(pd.pivot_table(data = dataset, columns = "Marka", values = "Fiyat")).T)

dataset3=dataset.loc[dataset["Asırı_Deger"] == False]

#CCM- olan 1 adet oldugu icin ilgili satir silindi
dataset3.drop(dataset3.loc[dataset3['CCM']=='-'].index, inplace=True)
dataset3.drop(dataset3.loc[dataset3['Arac Tip']=='-'].index, inplace=True)
dataset3.drop(dataset3.loc[dataset3['CCM']=='Bilmiyorum'].index, inplace=True)
#-lerin hepsi bilinmiyora donsturulup veri on isleme yapilacak
#İlan 2 aylik tarihi kisiti kaldirilip 2020 
#İlan 2 aylik tarihi kisiti kaldirilip 2020 
dataset3.drop(['İlan Tarihi'],axis=1,inplace=True)

#dataset3 ozet
#-----------------------------------------------------------------------
dataset3.drop(['Asırı_Deger'],axis=1,inplace=True)
print("dataset3 aciklama")
print(dataset3.describe())
#Beygir gucunu kaldirma
#-------------------------------------------------------------------------------
dataset3.drop(['Beygir Gucu'],axis=1,inplace=True)
#Fytat icin korelasyon
corrCar=dataset3.corr()
print(corrCar)
#-----------------------------------------------------------------------------
#Box plot
#Model yili

attrib = 'Model Yıl'
data = pd.concat([dataset3['Fiyat'], dataset3[attrib]], axis=1)
f, ax = plt.subplots(figsize=(20, 12))
fig = sns.boxplot(x=attrib, y="Fiyat", data=data)
fig.axis(ymin=0, ymax=450000)
#Markasina gore
attrib = 'Marka'
data = pd.concat([dataset3['Fiyat'], dataset3[attrib]], axis=1)
f, ax = plt.subplots(figsize=(20, 12))
fig = sns.boxplot(x=attrib, y="Fiyat", data=data)
fig.axis(ymin=0, ymax=450000)
#Yakıt Türüne Göre Fiyat
attrib = 'Yakıt Turu'
data = pd.concat([dataset3['Fiyat'], dataset3[attrib]], axis=1)
f, ax = plt.subplots(figsize=(20, 12))
fig = sns.boxplot(x=attrib, y="Fiyat", data=data)
fig.axis(ymin=0, ymax=450000)
#Vites Türüne Göre Fiyatlar
attrib = 'Vites'
data = pd.concat([dataset3['Fiyat'], dataset3[attrib]], axis=1)
f, ax = plt.subplots(figsize=(20, 12))
fig = sns.boxplot(x=attrib, y="Fiyat", data=data)
fig.axis(ymin=0, ymax=450000)
#Satıcıya Göre Fiyatlar
attrib = 'Kimden'
data = pd.concat([dataset3['Fiyat'], dataset3[attrib]], axis=1)
f, ax = plt.subplots(figsize=(20, 12))
fig = sns.boxplot(x=attrib, y="Fiyat", data=data)
fig.axis(ymin=0, ymax=450000)
#Renk icin
attrib = 'Renk'
data = pd.concat([dataset3['Fiyat'], dataset3[attrib]], axis=1)
f, ax = plt.subplots(figsize=(20, 12))
fig = sns.boxplot(x=attrib, y="Fiyat", data=data)
fig.axis(ymin=0, ymax=450000);




#model kurma
#-----------------------------------------------------------------------------
#Arac Tipi ve Rwnk çıkarılıd
cols = ['Marka', 'Arac Tip Grubu', 'Arac Tip','Model Yıl','Yakıt Turu', 'Vites', 'CCM', 'Renk', 'Kasa Tipi',
       'Kimden','Km']
X = dataset3[['Marka', 'Arac Tip Grubu','Model Yıl','Yakıt Turu', 'Vites', 'CCM', 
       'Kimden','Km']]
Y = dataset3.iloc[:,12:13]
X = pd.get_dummies(data=X)
#------------------------------------------------------------------------------


#Train tes bolme
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)

from sklearn import neighbors
# the value of n_neighbors will be changed when we plot the histogram showing the lowest RMSE value
knn = neighbors.KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, Y_train)

predicted = knn.predict(X_test)
residual = Y_test - predicted

fig = plt.figure(figsize=(30,30))
ax1 = plt.subplot(211)
sns.distplot(residual, color ='teal')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Residual counts',fontsize=35)
plt.xlabel('Residual',fontsize=25)
plt.ylabel('Count',fontsize=25)

ax2 = plt.subplot(212)
plt.scatter(predicted, residual, color ='teal')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Predicted',fontsize=25)
plt.ylabel('Residual',fontsize=25)
plt.axhline(y=0)
plt.title('Residual vs. Predicted',fontsize=35)

plt.show()

from sklearn.metrics import mean_squared_error,mean_absolute_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)
print('MSE:')
print(mean_squared_error(Y_test, predicted))
print('MAE:')
print(mean_absolute_error(Y_test, predicted))


from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))

rmse_l = []
num = []
for n in range(2, 16):
    knn = neighbors.KNeighborsRegressor(n_neighbors=n)
    knn.fit(X_train, Y_train)
    predicted = knn.predict(X_test)
    rmse_l.append(np.sqrt(mean_squared_error(Y_test, predicted)))
    num.append(n)
    
df_plt = pd.DataFrame()
df_plt['rmse'] = rmse_l
df_plt['n_neighbors'] = num
ax = plt.figure(figsize=(15,7))
sns.barplot(data = df_plt, x = 'n_neighbors', y = 'rmse')
plt.show()

#------------------------------------------------------------------------------
#Decision Tree Regresyon
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_features='auto')
dtr.fit(X_train, Y_train)
predicted = dtr.predict(X_test)
predicted=pd.DataFrame(predicted,columns={'Fiyat'})


#Duzeltilecek
residual = Y_test - predicted

fig = plt.figure(figsize=(30,30))
ax1 = plt.subplot(211)
sns.distplot(residual, color ='orange')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Residual counts',fontsize=35)
plt.xlabel('Residual',fontsize=25)
plt.ylabel('Count',fontsize=25)




from sklearn.metrics import mean_squared_error,mean_absolute_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)
print('MSE:')
print(mean_squared_error(Y_test, predicted))
print('MAE:')
print(mean_absolute_error(Y_test, predicted))


print('Variance score: %.2f' % r2_score(Y_test, predicted))

#----------------------------------------------------------------------------------
#Linear Regression


from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

predicted = regr.predict(X_test)
residual = Y_test - predicted

fig = plt.figure(figsize=(30,30))
ax1 = plt.subplot(211)
sns.distplot(residual, color ='teal')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Residual counts',fontsize=35)
plt.xlabel('Residual',fontsize=25)
plt.ylabel('Count',fontsize=25)

ax2 = plt.subplot(212)
plt.scatter(predicted, residual, color ='teal')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Predicted',fontsize=25)
plt.ylabel('Residual',fontsize=25)
plt.axhline(y=0)
plt.title('Residual vs. Predicted',fontsize=35)

plt.show()


from sklearn.metrics import mean_squared_error,mean_absolute_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)
print('MSE:')
print(mean_squared_error(Y_test, predicted))
print('MAE:')
print(mean_absolute_error(Y_test, predicted))


print('Variance score: %.2f' % r2_score(Y_test, predicted))

#------------------------------------------------------------------
#Random Forest

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=20, random_state=0)
rfr.fit(X_train, Y_train)
predicted = rfr.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error

rmse = np.sqrt(mean_squared_error(Y_test, predicted))

print('RMSE:')
print(rmse)
print('MSE:')
print(mean_squared_error(Y_test, predicted))
print('MAE:')
print(mean_absolute_error(Y_test, predicted))

print('Variance score: %.2f' % r2_score(Y_test, predicted))


