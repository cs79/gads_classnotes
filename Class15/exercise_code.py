from pandas import DataFrame, Series
import pandas as pd

url = 'https://raw.githubusercontent.com/justmarkham/DAT4/master/data/used_vehicles.csv'
cols = ['price', 'year', 'miles', 'doors', 'type']
data = pd.read_csv(url)

data.groupby(data.type).describe()
# avg price difference ~5.5k

data.groupby(data.doors).describe()
# 2k

data.groupby(data.doors).describe()

# split miles at 100000
miles_split = []

for i in data.miles:
    if i > 100000:
        miles_split.append(1)
    else:
        miles_split.append(0)

data['miles_split'] = miles_split
# grouped by this, price diff is 9.5k - looks like a good initial split

data.groupby([data.miles_split, data.type]).describe()

scatter(data.year, data.price) # 2006/2007 looks like an elbow

years_split = []

for i in data.year:
    if i > 2006:
        years_split.append(1)
    else:
        years_split.append(0)

data['years_split'] = years_split   # looks like a good second split

data.groupby([data.miles_split, data.years_split]).describe()

data_copy = data[['miles_split', 'years_split', 'doors', 'type', 'price']] # quick cleanup

data_copy.groupby([data.miles_split, data.years_split, data.doors]).describe()
# decent third split
