# hapiness_score_analyse
use machine leaning(unsupervised learning) to analyse hapiness_score  
## Context
The World Happiness Report is a landmark survey of the state of global happiness. The first report was published in 2012, the second in 2013, the third in 2015, and the fourth in the 2016 Update. The World Happiness 2017, which ranks 155 countries by their happiness levels, was released at the United Nations at an event celebrating International Day of Happiness on March 20th. The report continues to gain global recognition as governments, organizations and civil society increasingly use happiness indicators to inform their policy-making decisions. Leading experts across fields – economics, psychology, survey analysis, national statistics, health, public policy and more – describe how measurements of well-being can be used effectively to assess the progress of nations. The reports review the state of happiness in the world today and show how the new science of happiness explains personal and national variations in happiness.
## Data understanding
We conducted an exploratory analysis using the provided dataset to uncover patterns in the happiness ranking and gain insights into the uniqueness of each country. The analysis focused on parameters like ```Economy (GDP per Capita), Family (Social factor), Health (Life Expectancy), Freedom, Generosity, and Trust (Perception of Government Corruption) ```. 

## Data Cleaning and Preprocessing
1. Drop unwanted columns like the Hapiness Rank, Standard Error, Lower Confidence Interval, Dystopia Residual and more features because they're useless to our analysis.

2. Add a year column to be used in the data integration part in order differentiate between each dataset we have

3. Rename some columns in each dataset.
   
5. check the missing value and outier value
## Exploratory Data Analysis (EDA)
### statistical analysis
![image](https://github.com/kehanantoineLIU/hapiness_score_analyse/assets/125217787/9b71eda8-eea8-495d-93b9-e8b171258531)   
We can notice that the Distribution of the happiness score is near to the normal distribution.   
![image](https://github.com/kehanantoineLIU/hapiness_score_analyse/assets/125217787/053ca4b8-ca11-4cda-bf22-a6784e7238fc)   
we can notice several information from this plot:
Eastern Asia is the happiest region in Asia
Sub-Saharan Africa has Freedom greater than the life Expectancy
Central and Eastern Europe have the lowest Trust for the Government   
![image](https://github.com/kehanantoineLIU/hapiness_score_analyse/assets/125217787/e2cf451e-6fbd-4bf5-9e65-52f8edab7f3c)   
We find that with the globalization of economic development, there is an overall upward trend in the global happiness index. Except for a brief decline in 2017. We suspect there are two main reasons for this. One reason is that some European and Latin American countries  have faced an economic downturn in 2017. The second reason is terrorist attacks. events such as the Manchester Arena bombing and the Barcelona attacks in 2017 caused a large number of casualties and panic, affecting people's lives and well-being.
![image](https://github.com/kehanantoineLIU/hapiness_score_analyse/assets/125217787/4ca1e6b1-68e7-43f9-868e-36906457ff27)   
![image](https://github.com/kehanantoineLIU/hapiness_score_analyse/assets/125217787/1019186c-56e2-41ff-81ee-0b5b91acf948)   
We find that countries with high happiness indexes tend to be developed countries. And countries with low happiness indexes are concentrated in Africa


### heatmap
![image](https://github.com/kehanantoineLIU/hapiness_score_analyse/assets/125217787/78b22d39-2765-4f22-a443-ab9479936a1b)   
From the correlation matrix, we can notice that the GDP per capita, Family and the Life Expectancy are strongly correlated with the Happiness Score variable.
We can say that countries that have high GDP per capita, have the high services and their citizens tends to make a family are the happiest countries.

## use two types of unsupervised learning operations were performed
### Data downgrading  
To enhance the readability of our analysis and visualizations, we employed techniques to reduce the number of features in the dataset. One of these techniques was Principal Component Analysis (PCA), a statistical method that simplifies the dataset. Additionally, we used the TSNE Technique (t-distributed stochastic neighbor embedding) as a non-linear approach to validate our analysis, serving a similar purpose to PCA
#### PCA
![image](https://github.com/kehanantoineLIU/hapiness_score_analyse/assets/125217787/86293372-d28e-4b52-a3b7-3e58f4a3bd00)   
we will choose the number of components based on the explained variance we want to retain. here we choose 90% of the variance
So here we choose to keep the four constituent parts
![image](https://github.com/kehanantoineLIU/hapiness_score_analyse/assets/125217787/7ddc6dbf-9236-4195-b08b-f8c4018ef9a8)
We found that although we were able to downscale the data to four factors, these four factors did not have good interpretability

#### TSNE
![image](https://github.com/kehanantoineLIU/hapiness_score_analyse/assets/125217787/e4ed735e-4c70-4ccb-914a-a723a201c4aa)
![image](https://github.com/kehanantoineLIU/hapiness_score_analyse/assets/125217787/a8d8dbfa-24b2-48b3-bfdb-7b576ab66db6)   
From the two previous visualizations, it's evident that Component 1 (Poverty), has a significantly negative correlation with the Happiness Score, GDP, Family, and Life
Expectancy. This indicates that poverty plays a crucial role in a country's overall happiness. Conversely, Component 2 (Generosity) tends to have a positive influence on a country's Happiness Score. A



### clustering
#### Metrics
Calinski-Harabas (HC): Also known as the Variance Ratio Criterion, this index measures the ratio of between-cluster variance to within-cluster variance. Higher value of CH index means the clusters are dense and well separated, although there is no “acceptable” cut-off value. We need to choose that solution which gives a peak or at least an abrupt elbow on the line plot of CH indices.   
Silhouette: Silhouette Score is a metric to evaluate the performance of clustering algorithm. It uses compactness of individual clusters(intra cluster distance) and separation amongst clusters (inter cluster distance) to measure an overall representative score of how well our clustering algorithm has performed. The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters

#### K-means  
The elbow method is a graphical representation of finding the optimal 'K' in a K-means clustering. It works by finding WCSS (Within-Cluster Sum of Square) i.e. the sum of the square distance between points in a cluster and the cluster centroid.   
Plot the Elbow Method to find the optimal K  
![image](https://github.com/kehanantoineLIU/hapiness_score_analyse/assets/125217787/0763bfb2-7778-467a-b27a-17f3f8d4db1d)

![image](https://github.com/kehanantoineLIU/hapiness_score_analyse/assets/125217787/0f9913bd-7ee1-4fed-8370-620a51ec339c)
the best result with clusters number of 3.
#### Gaussian mixture model 
Gaussian Mixture Models (GMMs) assume that there are a certain number of Gaussian distributions, and each of these distributions represent a cluster. Hence, a Gaussian Mixture Model tends to group the data points belonging to a single distribution together.   

![image](https://github.com/kehanantoineLIU/hapiness_score_analyse/assets/125217787/65e2b842-24c1-4633-8bd6-0b0d963fc6a7)

#### hierarchical clustering 
In the this methods, we will only use data of 2019 because of the 2 following reasons:
Reduce calculation.
We can see in the results of K-Means and Gaussian Mixture Model, the silhouette scores are high, so that means the clustering has the low seperation level. Therefore, the data combined from other years could make some noise in the result.
![image](https://github.com/kehanantoineLIU/hapiness_score_analyse/assets/125217787/d10aef9b-cf06-4133-a50c-f6ee6cfab864)


#### mean shift
Mean Shift is a clustering algorithm that doesn't require specifying the number of clusters in advance. It automatically discovers clusters in a dataset based on data density.   

![image](https://github.com/kehanantoineLIU/hapiness_score_analyse/assets/125217787/ee772ac6-dbd1-4802-a011-c7f6abe96747)   
The mean shift method gave us the clusters number when silhouette score is 4. However, when we visualized in the figure, the cluster 4 seem to be the outliers. So we still have the same result with other methods.
#### Clustering Conclusion:
1. We acquired the clusters number is 3 with all 4 methods althought the seperation level of clusters is not too high (it's better to have a silhouette score > 0.6).
2. The number of clusters is similar with our expection when we did PCA. Generally, we have 3 clusters:
3. Countries that are rich and generous are frequently located in North America and Western Europe;
4. Countries that are poor and generous are frequently located in Sub-Saharan Africa;
5. And the rest of the countries that don't fit on the previous two groups are less generous.
## Finial
One notable finding is the correlation between components like "Poverty" and "Generosity" and the Happiness Score. For instance, the negative correlation of the "Poverty" component with the Happiness Score, GDP, Family, and Life Expectancy underscores its significant role in overall happiness. The clustering of countries based on their characteristics further revealed geographical patterns, highlighting the association between wealth, generosity, and geographic location.
