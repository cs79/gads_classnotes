## Class 05 - pandas

### Movielens example
```python

#reading in data from movielens, in appropriate format
cols= ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_table('u.user', sep='|', header=None, names=cols, index_col='user_id')

# new in pandas vs. the version I was using:
users.describe(include='all') # will call describe per-col type

# mult-column select
users[['age', 'gender']] # subsets on just the desired cols, passed in a list


users.sort_index(by='age') # can sort by a column

users.sort_index(by=['age', 'occupation']) # or multiple columns

```

## Drinks practice exercise

```python

drinks = pd.read_csv('drinks.csv', na_filter=False) #use the one from before in gads folder

drinks.head(10)

drinks.dtypes

drinks.beer_servings

drinks.beer_servings.mean()

drinks[drinks.continent=='EU']

drinks[drinks.continent=='EU'].beer_servings.mean()

drinks[(drinks.continent=='EU') & (drinks.wine_servings > 300)]

drinks.sort_index(by='total_litres_of_pure_alcohol', ascending=False)[:10].country
drinks.sort_index(by='total_litres_of_pure_alcohol').tail(10).country #another, shorter way

drinks.beer_servings.max().country

drinks[drinks.beer_servings == drinks.beer_servings.max()].country

drinks.continent.value_counts()

```
## Other cool things:

```python

drinks.loc[192, 'beer_servings':'wine_servings'] = np.nan #sets a range of cols AT a specific loc (index NAME); .iloc refers to things by position

```

## Split - Apply - Combine

re-check class notes, and read Ch. 09 in McKinney
