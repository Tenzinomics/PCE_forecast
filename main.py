import dataimport 
import pca_decomp as pca
import pandas as pd
import matplotlib.pyplot as plt


#Importing the datacode 
data_code = pd.read_excel("C:/Users/tchozin/Desktop/Python projects/Dual Mandate Forecast/data_pce_lead.xlsx")
df_main_pca, df_main_reg, df_raw = dataimport.data_main(data_code ,2000)


#Making sure both input and output series are same length
df_clean_pca =  df_main_pca[12:len(df_main_pca)-1].fillna(method='bfill').fillna(method='ffill')
df_clean_reg =  df_main_reg[12:len(df_main_reg)-1].fillna(method='bfill').fillna(method='ffill')


principal_comp, eigenvalues = pca.pca_model(df_clean_pca)
principal_comp_firstfour = principal_comp.iloc[:, :4] 


#Plotting the eigen values
plt.bar(range(1, len(eigenvalues) + 1), eigenvalues)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Eigenvalues')
plt.show()

# plt.plot(principal_comp_firstfour["PC1"])
# plt.show()



#regression
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#combining the data frame for input variables and greating lagged value
lagg_for_depend = 1
lagg_for_pca = 1

#the input variable has a lagged output variable
#df_combined_lagged = pd.concat([df_clean_reg.shift(lagg_for_depend), principal_comp_firstfour.shift(lagg_for_pca)], axis=1).dropna()
df_combined_lagged = pd.concat([principal_comp_firstfour.shift(lagg_for_pca)], axis=1).dropna()

#main data resizing to compensate for the na values
df_clean_reg_resize = df_clean_reg[lagg_for_depend:]

# Split data into training and testing sets (80% training, 20% testing)
train_size = int(len(df_combined_lagged) * 0.3)
X_train, X_test = df_combined_lagged[:train_size], df_combined_lagged[train_size:]
Y_train, Y_test = df_clean_reg_resize[:train_size], df_clean_reg_resize[train_size:]


lr = LinearRegression()
lr.fit(X_train, Y_train)
y_pred = lr.predict(X_test)

y_pred = pd.DataFrame(y_pred, index=Y_test.index.values, columns=["Predicted"])

import matplotlib.pyplot as plt

plt.plot(Y_test, label="Actual Values")
plt.plot(y_pred, label="Predicted Values")
plt.legend()
plt.show()

#summary statistics
import statsmodels.api as sm
log_clf =sm.OLS(Y_test,X_test)
classifier = log_clf.fit()
print(classifier.summary2())


fig, ax1 = plt.subplots()
ax1.plot(principal_comp.iloc[:, :4], color='gray')
ax2 = ax1.twinx()
ax2.plot(df_clean_reg , color='red')
plt.show()





# # Concatenate the shifted data with the new rows (this ensures the rows are added without losing data)
# pcshifted = pd.concat([principal_comp_firstfour.iloc[:, :1].shift(n), new_rows], axis=1)
# #del pcshifted['extrarow'] 

# fig, ax1 = plt.subplots()
# ax1.plot(pcshifted, color='gray')
# ax2 = ax1.twinx()
# ax2.plot(df_clean_reg , color='red')
# plt.show()