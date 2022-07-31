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

## Data Cleaning


