# NYC-Online-Taxi-Clustering

In this project, I will be performing an unsupervised clustering of data on the customer's records from New York City taxi online. Customer segmentation is the practice of separating customers into groups that reflect similarities among customers in each cluster. Dividing customers into segments optimizing the significance of each customer to the business and also helps the business to cater to the concerns of different types of customers. To modify products according to distinct needs and behaviours of the customers.

## Importing Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
```
## Loading Dataset

``` python
df = pd.read_csv('ONLINETAXI_TRANSACTION.csv')
df.head()
```
| index | VENDOR	| passenger_count |	trip_distance |	rate_code |	store_and_fwd_flag | payment_type |	fare_amount |	extra |	mta_tax	| tip_amount |	tolls_amount |	imp_surcharge |	total_amount |	pickup_location_id |	dropoff_location_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 |	2 |	2	| 7.22 |	1 |	N |	1 |	22.5 |	0.5 |	0.5 |	4.76 |	0.00 |	0.3 |	28.56 |	132 |	28 |
| 1 |	1 |	1 |	7.50 |	1 |	N |	1 |	25.0 |	0.5 |	0.5 |	4.00 |	0.00 |	0.3 |	30.30 |	230 |	33 |
| 2 |	2 |	1 |	9.74 |	1 |	N |	1 |	34.0 |	0.5 |	0.5 |	6.16 |	5.76 |	0.3 |	47.22 |	138 |	249 |
| 3 |	2 |	5 |	10.92 |	1 |	N |	1 |	31.5 |	0.5 |	0.5 |	7.71 |	5.76 |	0.3 |	46.27 |	138 |	161 |
| 4 |	1 |	1 |	10.50 |	1 |	N |	1 |	32.5 |	0.5 |	0.5 |	5.07 |	0.00 |	0.3 |	38.87 |	148 |	165  |

## Features Understanding

- Vendor: A code indicating the TPEP provider that provided the record. TPEP is a vendor who has [contracted with] been authorized by the Commission to install and maintain the Taxicab Technology System in Taxicabs.
- passenger_count: The number of passengers in the vehicle.
- trip_distance: The elapsed trip distance in miles reported by the taximeter
- rate_code: The final rate code in effect at the end of the trip.
    - 1 = Standard Rate
    - 2 = JFK
    - 3 = Newark
    - 4 = Nassau or Westchester
    - 5 = Negotiated fare
    - 6 = Group ride


- store_and_fwd_flag: This flag indicates whether the trip record was held in vehicle memory before sending to the vendor, aka “store and forward,” because the vehicle did not have a connection to the server.
    - Y = store and forward trip
    - N = not a store and forward trip

- payment_type: A numeric code signifying how the passenger paid for the trip. 
- fare_amount: The time-and-distance fare calculated by the meter.
- extra: Miscellaneous extras and surcharges.
- mta_tax: Tax for the Metropolitan Transportation Authority
- tip_amount: This field is automatically populated for credit card tips. Cash tips are not included.
- tolls_amount: Total amount of all tolls paid in trip.
- imp_surcharge: 0.30 improvement surcharge assessed trips.
- total_amount: The total amount charged to passengers. Does not include cash tips.
- pickup_location_id: TLC Taxi Zone in which the taximeter was engaged
- dropoff_location_id: TLC Taxi Zone in which the taximeter was disengaged

###### Source: 
- https://www.kaggle.com/datasets/anandaramg/taxi-trip-data-nyc?select=trip_data_dictionary.pdf
- https://portal.311.nyc.gov/article/?kanumber=KA-01245

## Data Preprocessing

### Check data Anomalies

#### - Trip Distance
```python
df[df['trip_distance']<=0.0]
```
<img width="584" alt="image" src="https://user-images.githubusercontent.com/99155979/182074624-692c06ae-5223-4cec-8fc9-03422b08c1fb.png">

```python
((len(df[df['trip_distance']<=0.0]))/len(df))*100
```
0.6988281736223483
Drop column "trip_distance" with value 0 or less. It's odd with 0 trip distance, means the taxi doesn't move at all but there's a fare amount for it. Also it's only 0.69% of data so it's really a small amount of data.

```python
df = df[df['trip_distance']>0.0]
```

#### - Total Amount
```python
df[df['total_amount']<=0]
```
<img width="577" alt="image" src="https://user-images.githubusercontent.com/99155979/182074977-0323fc76-2fc9-493b-85ae-92ed72796b6e.png">

```python
((len(df[df['total_amount']<=0.0]))/len(df))*100
```
0.05058735047632692
Drop column "total_amount" with value 0 or less. It's odd customer doesn't pay any amount of fee, even some of the data have minus value. Also it's only 0.05% of data so it's really a small amount of data.

```python
df = df[df['total_amount']>0]
```

## Clustering

I'm using KMeans, because K-means is the simplest and easy to understand. To implement and to run. other algorithms are much harder to implement efficiently and have much more parameters to set.

### Import KMeans package
```python
from sklearn.cluster import KMeans
```
```python
X = df[['total_amount', 'trip_distance']].values
```
Based on the features I use ``total_amount`` and ``trip_distance`` as an X value. trip_distance because well the further the distance the more expensive the fare. total_amount because it's the total fee/fare passenger need to pay.

```python
plt.figure(figsize=(13,7))
sns.scatterplot(x='total_amount', y='trip_distance', data=df)
plt.show()
```
![download](https://user-images.githubusercontent.com/99155979/182075507-c0f7ce84-5203-4249-9009-18e195d596ef.png)

### Elbow Analysis
- Used for get an ideal number for clustering
- Inertia => sum of squared distance data point to the closer centroid
- The most optimal K is when the decrease in intertia is no longer significant

```python
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(X)
Elbow_M.show()
```
![__results___35_1](https://user-images.githubusercontent.com/99155979/182077056-b984405e-c5c3-4b21-9d70-41749e2263b3.png)

Based on Elbow Analysis the number of clusters is 4

```python
model_KM = KMeans(n_clusters = 4, random_state=42)
cluster = model_KM.fit_predict(X)
df['segment'] = cluster
df.head()
```
<img width="578" alt="image" src="https://user-images.githubusercontent.com/99155979/182077341-2c786a85-a28b-46c6-b30c-c76c7739c79d.png">

```python
centroid = model_KM.cluster_centers_

plt.figure(figsize=(15,8))
plt.scatter(x=X[:,0], y=X[:,1], c=cluster, s=30)
plt.scatter(x=centroid[:,0], y=centroid[:,1], c='r', s=100)
plt.xlabel("Total Amount")
plt.ylabel("Trip Distance")
plt.show()
```
![download](https://user-images.githubusercontent.com/99155979/182077512-239c55e8-6d95-41d7-9ee7-6dc960c342ac.png)

## Evaluating Model

![download](https://user-images.githubusercontent.com/99155979/182077550-c7fa5330-8d97-4a8a-a149-31b9ce95f03f.png)
Based on graph above, it is known that cluster 0 has the highest rate rate, followed by cluster 3, 2, 1

![download](https://user-images.githubusercontent.com/99155979/182077578-79dd242c-81d3-440b-9fc5-75f024a29666.png)
- group 0: short and average trip distance & low total fee amount
- group 1: short and long trip distance & high total fee amount
- group 2: shot trip distance & average total fee amount
- group 3: average trip distance & low total fee amount

![download](https://user-images.githubusercontent.com/99155979/182077651-bfabd68c-b41a-4a4c-a4df-2ea850922267.png)
- group 0, 3, 2: Standart rate
- group 1: JFK final rate

![download](https://user-images.githubusercontent.com/99155979/182077693-bc44184d-2b40-404c-83ed-ec9d2a427e32.png)
![download](https://user-images.githubusercontent.com/99155979/182077707-4850654a-6761-44e6-bd15-c59e80ae1d9a.png)
- group 0: the lowest tip given
- group 1: the highest tip given
- group 2: average tip given between the lowest and the highest
- group 3: low tip given

## Profiling

##### Group 0:
- Short and average trip distance & low total fee amount
- Standart rate
- The lowest tip given

##### Group 1:
- Short and long trip distance & high total fee amount
- JFK final rate
- The highest tip given

##### Group 2:
- Shot trip distance & average total fee amount
- Standart rate
- Average tip given between the lowest and the highest

##### Group 3:
- Average trip distance & low total fee amount
- Standart rate
- Low tip given

## Segmenting

##### Group 0:
- Low Income
- Within city mobility

##### Group 1:
- Highest Income
- Within city and and long trip mobility including to JFK(airport)

##### Group 2:
- Average income
- Within city mobility

##### Group 3:
- Low income
- Within city mobility
