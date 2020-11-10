# CSCI-550-HW3 ![Python package](https://github.com/davidkelly-wk/CSCI-550-HW3/workflows/Python%20package/badge.svg)
## Datamining HW3
Required packages for this application can be installed with the pip. Run pip install -r requirements.txt to install.

Note: if you have pip configured as your python 2 package manager, you may have to use pip3 instead as this application requires python 3.

## THIS PROGRAM TAKES ~30 MINUTES TO RUN
k-NN is inherently slow, and using a larger value such as 41 (the sqrt of the dataset length) causes it to take even longer. To run without k-NN, simply comment out lines 105-107 in main.py 

## k-NN results

|FIELD1|dataset|k  |fold_number|method|accuracy          |precision         |recall            |F1-score          |
|------|-------|---|-----------|------|------------------|------------------|------------------|------------------|
|0     |car    |3  |1          |KNN   |0.9046242772227606|0.9046242772227605|0.9046242772227605|0.9046242772227605|
|1     |car    |3  |2          |KNN   |0.9682080922149087|0.9682080922149086|0.9682080922149086|0.9682080922149086|
|2     |car    |3  |3          |KNN   |0.9306358379843408|0.9306358379013665|0.8702702700701241|0.8994413405589713|
|3     |car    |3  |4          |KNN   |0.7999999999130434|0.799999999826087 |0.5714285713989944|0.6666666665861515|
|4     |car    |3  |5          |KNN   |0.8405797100462087|0.84057970994749  |0.6373626373022582|0.7249999998874999|
|5     |car    |5  |1          |KNN   |0.9075144506314946|0.9075144506314945|0.9075144506314945|0.9075144506314945|
|6     |car    |5  |2          |KNN   |0.8872832367703566|0.8872832367703565|0.8872832367703565|0.8872832367703565|
|7     |car    |5  |3          |KNN   |0.8930635835878246|0.8930635835878244|0.8930635835878244|0.8930635835878244|
|8     |car    |5  |4          |KNN   |0.7971014491892459|0.7971014491031296|0.5670103092507174|0.6626506023312527|
|9     |car    |5  |5          |KNN   |0.8289855071510186|0.8289855070556605|0.6177105831025008|0.7079207919762769|
|10    |car    |41 |1          |KNN   |0.9364161847188347|0.9364161847188345|0.9364161847188345|0.9364161847188345|
|11    |car    |41 |2          |KNN   |0.9017341038140266|0.9017341038140264|0.9017341038140264|0.9017341038140264|
|12    |car    |41 |3          |KNN   |0.9161849708576967|0.9161849708576965|0.9161849708576965|0.9161849708576965|
|13    |car    |41 |4          |KNN   |0.8057971013606384|0.8057971012720018|0.5803757828474423|0.6747572814685645|
|14    |car    |41 |5          |KNN   |0.7275362318181054|0.7275362317521529|0.4709193245887733|0.5717539862998843|

## Decision Tree results

|FIELD1|dataset|k  |fold_number|method|accuracy          |precision         |recall             |F1-score          |
|------|-------|---|-----------|------|------------------|------------------|-------------------|------------------|
|0     |car    |N/A|1          |DTree |0.8063583813258045|0.8063583813258045|0.8063583813258045 |0.8063583813258045|
|1     |car    |N/A|2          |DTree |0.7832369940559324|0.7832369940559324|0.7832369940559324 |0.7832369940559324|
|2     |car    |N/A|3          |DTree |0.7630057801947944|0.7630057801947944|0.7630057801947944 |0.7630057801947943|
|3     |car    |N/A|4          |DTree |0.6376811593803823|0.6376811593404746|0.36974789920344614|0.4680851063965596|
|4     |car    |N/A|5          |DTree |0.6840579709611426|0.6840579709077925|0.4191829485189404 |0.5198237885375226|

