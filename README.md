**COVID-19 Spread in Africa Using Machine Learning**

This project analyzes COVID-19 cases and deaths across African countries from 2020 to 2022. It uses exploratory data analysis, K-Means clustering, PCA, Decision Tree Regression, and Random Forest Regression to study regional patterns and predict COVID-19 mortality.
The dataset was obtained from Our World in Data and filtered to include African countries.

**Tools Used**
•	Python
•	Jupyter Notebook
•	Pandas
•	NumPy
•	Matplotlib
•	Seaborn
•	Scikit-learn

**Project Steps** 
1. Cleaned and filtered the dataset
2. Compared COVID-19 cases and deaths across African regions
3. Used K-Means clustering to group similar countries
4. Used PCA to visualize the clusters
5. Built Decision Tree and Random Forest regression models
6. Evaluated the models using MSE, RMSE, and R²

**Results**
Model	RMSE	R2 Score
Decision Tree Regressor	1.1321	0.67849
Random Forest Regressor	1.3257	0.5524

The Decision Tree Regressor performed better and explained about 67% of the variation in COVID-19 deaths.
The clustering results also showed that countries with similar COVID-19 outcomes were not always located in the same region.


