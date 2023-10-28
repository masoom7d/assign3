import pulp
import pandas as pd
import numpy as np
from tqdm import tqdm

data = pd.read_csv('data.csv')
data = data.dropna(subset=['Geopoint'])

#compactness of the district is going to be average distance between the individual counties in the district.
#Optimize compactness of redistricting
# Calculate all pairwise distance between counties using their lat lon

def degrees_to_radians(x):
     return((np.pi/180)*x)
     
def lon_lat_distance_miles(lon_a,lat_a,lon_b,lat_b):
    radius_of_earth = 24872/(2*np.pi)
    c = np.sin((degrees_to_radians(lat_a) - \
    degrees_to_radians(lat_b))/2)**2 + \
    np.cos(degrees_to_radians(lat_a)) * \
    np.cos(degrees_to_radians(lat_b)) * \
    np.sin((degrees_to_radians(lon_a) - \
    degrees_to_radians(lon_b))/2)**2
    return(2 * radius_of_earth * (np.arcsin(np.sqrt(c))))    

def lon_lat_distance_meters (lon_a,lat_a,lon_b,lat_b):
    return(lon_lat_distance_miles(lon_a,lat_a,lon_b,lat_b) * 1609.34) 

# added lat lon columns from Geopoint which is string
data[['lat', 'lon']] = data['Geopoint'].str.split(',', expand=True).astype(float)

#creating a data frame with lat and long values and pairwise distance between counties
latlon = data[['lat', 'lon']].values
n = data.shape[0]
dists = np.zeros((n,n))
for i in range(n):
    for j in range(i+1,n):
        dists[i,j] = lon_lat_distance_meters(latlon[i,1], latlon[i,0], latlon[j,1], latlon[j,0])
        dists[j,i] = dists[i,j]

dists_df = pd.DataFrame(dists, columns=data['Name'].values, index=data['Name'])
pops_df = data.set_index('Name')['Pop2023'].astype(float)


max_distance = 10000
max_district = 11
max_district_size = 13 #this means that 4 counties can be assigned to one district or represented by one district 
max_pop_diff = 10000000

counties = data['Name'].values.tolist()[:133]

#Define compactness function 
def compactness(district):
    """
    Find the compactness of the district
    - by calculating the average distance between the counties in districts
    - will likely need to make a matrix and average across
    - could be good to make a matrix and then index into it
    """
    k = len(district)
    if k == 1:
        return 0
    cur = dists_df.loc[district, district].values
    return cur[np.triu_indices(k, 1)].mean()

def distance(district):
    return np.max(dists_df.loc[district, district].values) <= max_distance

def population(district):
    min_pop = min(pops_df.loc[list(district)])
    max_pop = max(pops_df.loc[list(district)])
    return max_pop - min_pop <= max_pop_diff 

#create list of all possible districts
possible_districts = []
for county in tqdm(counties):
    neighbors = dists_df.loc[county,counties]
    neighbors = neighbors[neighbors <= max_distance].index.tolist()
    possible_districts += [tuple(sorted(c)) for c in pulp.allcombinations(neighbors, max_district_size) if distance(c) and population(c)]
possible_districts = list(set(possible_districts))
print(len(possible_districts))

#create a binary variable to state that a district is used
x = pulp.LpVariable.dicts('district', possible_districts, 
                            lowBound = 0,
                            upBound = 1,
                            cat = pulp.LpInteger)

redistrict_model = pulp.LpProblem("Redistricting Model", pulp.LpMinimize)

redistrict_model += sum([compactness(district) * x[district] for district in possible_districts])

#specify the maximum number of districts
redistrict_model += sum([x[district] for district in possible_districts]) <= max_district, "Maximum_number_of_districts"


#A county can be assigned to one and only one district
for county in counties:
    redistrict_model += sum([x[district] for district in possible_districts
                                if county in district]) == 1, "Must_zone_%s"%county

    
redistrict_model.solve()

if redistrict_model.status != -1:
    print("The choosen districts are out of a total of %s:"%len(possible_districts))
    for district in possible_districts:
        if x[district].value() == 1.0:
            print(district)
            
        
