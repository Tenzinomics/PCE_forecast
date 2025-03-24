<h1>US PCE Forecast Using Principal Component Analysis</h1>

Principal Component Analysis (PCA) reduces the dimensionality of a dataset without losing much of the variation in the dataset.
<br>
<br>
To forecast Personal Consumption Expenditures (PCE), I used nearly 80 leading economic data series. I applied PCA to decompose the dataset into eigenvectors and eigenvalues.
<br>
 <br>
![image](https://github.com/user-attachments/assets/2cae9aca-5522-4422-97ea-da9f621c7ce1)
 <br>
  <br>
Each eigenvector, more commonly known as a principal component (PC), corresponds to an eigenvalue that tells you its explanatory power - the bigger the eigenvalue, the more variation it explains in the dataset.
 <br>
  <br>
Following a rule of thumb, I selected the top four principal components with the highest explanatory power (highest eigenvalues), which together account for the majority of the variation in the dataset.
<br>
 <br>

![image](https://github.com/user-attachments/assets/e599aeb8-7507-40b1-835e-1d583f1affeb)
 <br>
  <br>
Next, I used these principal components, lagged by two months, as inputs to Ordinary Least Squares (OLS) regression. The chart below compares the model’s projections to the PCE values.
<br>
 <br>
![image](https://github.com/user-attachments/assets/fffcb444-28a9-4fa2-a143-a7043542e94d)
 <br>
  <br>
All in all, the model performs reasonably well in capturing the general trend of PCE. However, for precise point forecasts, the model needs modification.

<br>





