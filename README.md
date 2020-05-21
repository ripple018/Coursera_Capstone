# Food around the Universities in Canada

### Applied Data Science Capstone
### Coursera Capstone Project - The Battle of Neighborhoods

___by Ripple Shi, 20 May 2020___

----

## PART 1 INTRODUCTION

I believe food is an important part of life. Food provides you with nutrition, energy and ideally, satisfaction. As someone who will soon enroll in a university in Canada, I am really curious about what types of food I can get there. I was born and raised in Asia, where the diets are quite different from those in North America, so it will be a great relief if I know whether I could easily find my familiar types of meals nearby.

I am sure this is also a concern for others who are seeking to study, work or live in a different country. However, limit to the scale of the project, we will only explore different food suppliers around Canadian universities. They could be restaurants, café or any other venues that could satisfy people’s needs in food.

I hope this project could provide the readers with some insights on this subject. Although here we will focus on food, universities and Canada, I think the idea behind this project can also be applied to any similar intention.


To carry out the project, we are going to rely on the recommendations provided by Foursquare’s API. By specifying the coordinate of the university, Foursquare will return us some recommended venues that are in the food section within the limit and distance we set. Meanwhile, the category of a venue will also be returned, so we could use that to guess the main cuisine of the venue.

Also, we will use K-Means, a clustering algorithm, to cluster the universities in groups and explore the features of each group. Hopefully, we could get some interesting findings out of that.

## PART 2 DATA

The data used in this project come from three sources.

We start by getting a list of universities in Canada from Wikipedia. The link is https://en.wikipedia.org/wiki/List_of_universities_in_Canada.

Although we are not sure whether the list is complete, it should be enough to represent the population we are interested in.

Next, we will use the name and the province of the universities to get their coordinates. We obtain the latitude and longitude of each university using the dataset on http://py4edata.dr-chuck.net/. This is a subset of data from the Google Geocoding API, established by Dr. Charles R. Severance from University of Michigan. This data set is built to facilitate the study of Python courses taught by Dr. Chuck. Please note that this dataset is not my first choice to get the coordinates. We will further explain this problem in the Data Preprocessing section afterwards. Anyway, the coordinates retrieving from this data set allow us to specify the location in the search queries of Foursquare.

Finally, we use Foursquare’s API to get the recommendations in the food section and do the analysis. According to the documentation of Foursquare’s API, by using the endpoint “explore” we could get a list of recommended venues near the current location. The list includes much information, but we only need the venue name and the venue type. This information will be enough for us to summarize what types of venues we could find around the universities. 

Based on that, we will build our variables of what percent a venue category is taking among all the recommended venues that meet the conditions we set. These self-created variables will be  used to train a model using K-Means algorithm to get the clusters of the universities.

## PART 3 Methodology

### 3.1. Data Preprocessing

#### 3.1.1. List of the Universities


```python
import numpy as np
import pandas as pd
```


```python
!conda install -c conda-forge lxml --yes
url_wiki = "https://en.wikipedia.org/wiki/List_of_universities_in_Canada"
tables = pd.read_html(url_wiki)
```

    Collecting package metadata (current_repodata.json): done
    Solving environment: done
    
    # All requested packages already installed.
    


From the wikipedia page we could see there are 10 tables, each is a list of public universities of a province, and 1 table of private universities.

The format of the tables are slightly different. Lists of public universities missed one column named _province_ compared with list of private universities. We will consider this difference when reading the tables.


```python
province_ls = ["Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland and Labrador", "Nova Scotia", "Ontario", "Prince Edward Island", "Quebec", "Saskatchewan"]

df_public = pd.DataFrame()
i = 0

for table, province in zip(tables, province_ls):
    table.insert(2, "Province", province)
    df_public = pd.concat([df_public, table], axis=0)

df_public
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Name</th>
      <th>City</th>
      <th>Province</th>
      <th>Language</th>
      <th>Est.</th>
      <th colspan="3" halign="left">Students</th>
      <th>Notes</th>
    </tr>
    <tr>
      <th></th>
      <th>Name</th>
      <th>City</th>
      <th></th>
      <th>Language</th>
      <th>Est.</th>
      <th>Undergrad.</th>
      <th>Postgrad.</th>
      <th>Total</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alberta University of the Arts</td>
      <td>Calgary</td>
      <td>Alberta</td>
      <td>English</td>
      <td>1926</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1323</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Athabasca University</td>
      <td>Athabasca, Calgary, Edmonton</td>
      <td>Alberta</td>
      <td>English</td>
      <td>1970</td>
      <td>36240.0</td>
      <td>3460.0</td>
      <td>39700</td>
      <td>[38]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MacEwan University</td>
      <td>Edmonton</td>
      <td>Alberta</td>
      <td>English</td>
      <td>1971</td>
      <td>18897.0</td>
      <td>0.0</td>
      <td>18897</td>
      <td>[39]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mount Royal University</td>
      <td>Calgary</td>
      <td>Alberta</td>
      <td>English</td>
      <td>1910</td>
      <td>24768.0</td>
      <td>0.0</td>
      <td>24768</td>
      <td>[40]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>University of Alberta</td>
      <td>Edmonton, Camrose, Calgary</td>
      <td>Alberta</td>
      <td>Bilingual</td>
      <td>1906</td>
      <td>31904.0</td>
      <td>7598.0</td>
      <td>39502</td>
      <td>[41]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Université du Québec à Rimouski[note 3]</td>
      <td>Rimouski and Lévis</td>
      <td>Quebec</td>
      <td>French</td>
      <td>1969</td>
      <td>4620.0</td>
      <td>810.0</td>
      <td>5430</td>
      <td>[75]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Université du Québec à Trois-Rivières[note 3]</td>
      <td>Trois-Rivières</td>
      <td>Quebec</td>
      <td>French</td>
      <td>1969</td>
      <td>9160.0</td>
      <td>1450.0</td>
      <td>10610</td>
      <td>[76]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Université Laval</td>
      <td>Quebec City</td>
      <td>Quebec</td>
      <td>French</td>
      <td>1663</td>
      <td>27530.0</td>
      <td>10270.0</td>
      <td>37800</td>
      <td>[77]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>University of Regina</td>
      <td>Regina, Saskatoon, Swift Current</td>
      <td>Saskatchewan</td>
      <td>Bilingual</td>
      <td>1911</td>
      <td>10690.0</td>
      <td>1480.0</td>
      <td>12170</td>
      <td>[78]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>University of Saskatchewan</td>
      <td>Saskatoon</td>
      <td>Saskatchewan</td>
      <td>English</td>
      <td>1907</td>
      <td>16430.0</td>
      <td>2190.0</td>
      <td>18620</td>
      <td>[79]</td>
    </tr>
  </tbody>
</table>
<p>75 rows × 9 columns</p>
</div>




```python
df_private = tables[-4]
df_private
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>City</th>
      <th>Province</th>
      <th>Language</th>
      <th>Established</th>
      <th>Undergraduates</th>
      <th>Post-graduates</th>
      <th>Total students</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fairleigh Dickinson University (branch)</td>
      <td>Vancouver</td>
      <td>British Columbia</td>
      <td>English</td>
      <td>2007</td>
      <td>78[failed verification]</td>
      <td>50.0</td>
      <td>78[failed verification]</td>
      <td>[80]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>New York Institute of Technology (branch)</td>
      <td>Vancouver</td>
      <td>British Columbia</td>
      <td>English</td>
      <td>2007</td>
      <td>70[failed verification]</td>
      <td>40.0</td>
      <td>70[failed verification]</td>
      <td>[81]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Quest University</td>
      <td>Squamish</td>
      <td>British Columbia</td>
      <td>English</td>
      <td>2007</td>
      <td>700</td>
      <td>0.0</td>
      <td>700</td>
      <td>[82]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Niagara University (branch)</td>
      <td>Vaughan</td>
      <td>Ontario</td>
      <td>English</td>
      <td>2019</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[83]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Trinity Western University</td>
      <td>Langley</td>
      <td>British Columbia</td>
      <td>English</td>
      <td>1962</td>
      <td>2130</td>
      <td>730.0</td>
      <td>2860</td>
      <td>[84]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>University Canada West</td>
      <td>Victoria</td>
      <td>British Columbia</td>
      <td>English</td>
      <td>2005</td>
      <td>350[needs update]</td>
      <td>0.0</td>
      <td>350[needs update]</td>
      <td>[85]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Booth University College</td>
      <td>Winnipeg</td>
      <td>Manitoba</td>
      <td>English</td>
      <td>1982</td>
      <td>250</td>
      <td>0.0</td>
      <td>250</td>
      <td>[86]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Canadian Mennonite University</td>
      <td>Winnipeg</td>
      <td>Manitoba</td>
      <td>English</td>
      <td>1944</td>
      <td>600</td>
      <td>0.0</td>
      <td>600</td>
      <td>[56]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Kingswood University</td>
      <td>Sussex</td>
      <td>New Brunswick</td>
      <td>English</td>
      <td>1945</td>
      <td>300</td>
      <td>0.0</td>
      <td>300</td>
      <td>[87][needs update]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Crandall University</td>
      <td>Moncton</td>
      <td>New Brunswick</td>
      <td>English</td>
      <td>1949</td>
      <td>685</td>
      <td>0.0</td>
      <td>685</td>
      <td>[88][needs update]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>St. Stephen's University</td>
      <td>St. Stephen</td>
      <td>New Brunswick</td>
      <td>English</td>
      <td>1975</td>
      <td>100</td>
      <td>0.0</td>
      <td>100</td>
      <td>[89][needs update]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>University of Fredericton</td>
      <td>Fredericton</td>
      <td>New Brunswick</td>
      <td>English</td>
      <td>2005</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[59][needs update]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Atlantic School of Theology</td>
      <td>Halifax</td>
      <td>Nova Scotia</td>
      <td>English</td>
      <td>1971</td>
      <td>0</td>
      <td>124.0</td>
      <td>124</td>
      <td>[59]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Tyndale University</td>
      <td>Toronto</td>
      <td>Ontario</td>
      <td>English</td>
      <td>1894</td>
      <td>850</td>
      <td>0.0</td>
      <td>850</td>
      <td>[90]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Redeemer University College</td>
      <td>Ancaster</td>
      <td>Ontario</td>
      <td>English</td>
      <td>1982</td>
      <td>955</td>
      <td>0.0</td>
      <td>955</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>The King's University</td>
      <td>Edmonton</td>
      <td>Alberta</td>
      <td>English</td>
      <td>1979</td>
      <td>790</td>
      <td>0.0</td>
      <td>790</td>
      <td>[91]</td>
    </tr>
  </tbody>
</table>
</div>



To combine the two dataframes, we should adjust the format of the df_public to keep it consistent with that of df_private.


```python
# Drop additional column index
df_public.columns = pd.MultiIndex.droplevel(df_public.columns,level=1)

# Unify the column names
df_public.columns = df_private.columns

df_public.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>City</th>
      <th>Province</th>
      <th>Language</th>
      <th>Established</th>
      <th>Undergraduates</th>
      <th>Post-graduates</th>
      <th>Total students</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alberta University of the Arts</td>
      <td>Calgary</td>
      <td>Alberta</td>
      <td>English</td>
      <td>1926</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1323</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Athabasca University</td>
      <td>Athabasca, Calgary, Edmonton</td>
      <td>Alberta</td>
      <td>English</td>
      <td>1970</td>
      <td>36240.0</td>
      <td>3460.0</td>
      <td>39700</td>
      <td>[38]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MacEwan University</td>
      <td>Edmonton</td>
      <td>Alberta</td>
      <td>English</td>
      <td>1971</td>
      <td>18897.0</td>
      <td>0.0</td>
      <td>18897</td>
      <td>[39]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mount Royal University</td>
      <td>Calgary</td>
      <td>Alberta</td>
      <td>English</td>
      <td>1910</td>
      <td>24768.0</td>
      <td>0.0</td>
      <td>24768</td>
      <td>[40]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>University of Alberta</td>
      <td>Edmonton, Camrose, Calgary</td>
      <td>Alberta</td>
      <td>Bilingual</td>
      <td>1906</td>
      <td>31904.0</td>
      <td>7598.0</td>
      <td>39502</td>
      <td>[41]</td>
    </tr>
  </tbody>
</table>
</div>



Now we are able to concatenate the two lists and clean the tables altogether.


```python
df_univ = pd.concat([df_public, df_private], axis=0)
df_univ
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>City</th>
      <th>Province</th>
      <th>Language</th>
      <th>Established</th>
      <th>Undergraduates</th>
      <th>Post-graduates</th>
      <th>Total students</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alberta University of the Arts</td>
      <td>Calgary</td>
      <td>Alberta</td>
      <td>English</td>
      <td>1926</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1323</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Athabasca University</td>
      <td>Athabasca, Calgary, Edmonton</td>
      <td>Alberta</td>
      <td>English</td>
      <td>1970</td>
      <td>36240</td>
      <td>3460.0</td>
      <td>39700</td>
      <td>[38]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MacEwan University</td>
      <td>Edmonton</td>
      <td>Alberta</td>
      <td>English</td>
      <td>1971</td>
      <td>18897</td>
      <td>0.0</td>
      <td>18897</td>
      <td>[39]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mount Royal University</td>
      <td>Calgary</td>
      <td>Alberta</td>
      <td>English</td>
      <td>1910</td>
      <td>24768</td>
      <td>0.0</td>
      <td>24768</td>
      <td>[40]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>University of Alberta</td>
      <td>Edmonton, Camrose, Calgary</td>
      <td>Alberta</td>
      <td>Bilingual</td>
      <td>1906</td>
      <td>31904</td>
      <td>7598.0</td>
      <td>39502</td>
      <td>[41]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>University of Fredericton</td>
      <td>Fredericton</td>
      <td>New Brunswick</td>
      <td>English</td>
      <td>2005</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[59][needs update]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Atlantic School of Theology</td>
      <td>Halifax</td>
      <td>Nova Scotia</td>
      <td>English</td>
      <td>1971</td>
      <td>0</td>
      <td>124.0</td>
      <td>124</td>
      <td>[59]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Tyndale University</td>
      <td>Toronto</td>
      <td>Ontario</td>
      <td>English</td>
      <td>1894</td>
      <td>850</td>
      <td>0.0</td>
      <td>850</td>
      <td>[90]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Redeemer University College</td>
      <td>Ancaster</td>
      <td>Ontario</td>
      <td>English</td>
      <td>1982</td>
      <td>955</td>
      <td>0.0</td>
      <td>955</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>The King's University</td>
      <td>Edmonton</td>
      <td>Alberta</td>
      <td>English</td>
      <td>1979</td>
      <td>790</td>
      <td>0.0</td>
      <td>790</td>
      <td>[91]</td>
    </tr>
  </tbody>
</table>
<p>91 rows × 9 columns</p>
</div>



There are some columns not relevant to the location, so we drop some columns and clean the name of the universities for later processes.


```python
# Drop columns
df_univ.drop(df_univ.columns[3:], axis=1, inplace=True)

# Reset the index
df_univ.reset_index(drop=True, inplace=True)

# Remove the notes in the names
df_univ["Name"] = [name.replace("[note 3]","") for name in df_univ["Name"]]

# Escape the 's in the names
df_univ["Name"] = [name.replace("'s","\'s") for name in df_univ["Name"]]

df_univ.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>City</th>
      <th>Province</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alberta University of the Arts</td>
      <td>Calgary</td>
      <td>Alberta</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Athabasca University</td>
      <td>Athabasca, Calgary, Edmonton</td>
      <td>Alberta</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MacEwan University</td>
      <td>Edmonton</td>
      <td>Alberta</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mount Royal University</td>
      <td>Calgary</td>
      <td>Alberta</td>
    </tr>
    <tr>
      <th>4</th>
      <td>University of Alberta</td>
      <td>Edmonton, Camrose, Calgary</td>
      <td>Alberta</td>
    </tr>
  </tbody>
</table>
</div>



#### 3.1.2. Coordinates of the Universities

At this stage, we are ready to retrieve the coordinates of the universities.


```python
import requests
import urllib.parse
```


```python
lat_ls = []
lng_ls = []
address_ls = []
api_key = 42
serviceurl = "http://py4e-data.dr-chuck.net/json?"

for name, province in zip(df_univ["Name"], df_univ["Province"]):
    
    # Define the location of the university
    lc = "{}, {}, Canada".format(name, province)
    print("Searching for {} ......".format(name))
    
    # Set the parameters to be sent together with the url
    parms = dict()
    parms["address"] = lc
    parms['key'] = api_key
    url = serviceurl + urllib.parse.urlencode(parms)

    json = requests.get(url).json()
    
    if 'status' not in json or json['status'] == 'ZERO_RESULTS' :
        print('==== Failure To Retrieve ====')
        address_ls.append(np.nan)
        lat_ls.append(np.nan)
        lng_ls.append(np.nan)
        continue
    
    latlng = json['results'][0]['geometry']['location']  
    address = json['results'][0]['formatted_address']
    print("Result: ", latlng, address)
    address_ls.append(address)
    lat_ls.append(latlng['lat'])
    lng_ls.append(latlng['lng'])

print("Done!")
```

    Searching for Alberta University of the Arts ......
    Result:  {'lat': 51.0615707, 'lng': -114.0920983} 1407 14 Ave NW, Calgary, AB T2N 4R3, Canada
    Searching for Athabasca University ......
    Result:  {'lat': 54.714955, 'lng': -113.3085451} 1 University Dr, Athabasca, AB T9S 3A3, Canada
    Searching for MacEwan University ......
    Result:  {'lat': 53.5470544, 'lng': -113.506372} 10700 104 Ave NW, Edmonton, AB T5J 4S2, Canada
    Searching for Mount Royal University ......
    Result:  {'lat': 51.01101, 'lng': -114.129778} 30 Mt Royal Cir SW, Calgary, AB T3E 7C9, Canada
    Searching for University of Alberta ......
    Result:  {'lat': 53.5232189, 'lng': -113.5263186} 116 St & 85 Ave, Edmonton, AB T6G 2R3, Canada
    Searching for University of Calgary ......
    Result:  {'lat': 51.159473, 'lng': -114.214827} 11877 85 St NW, Calgary, AB T3R 1J3, Canada
    Searching for University of Lethbridge ......
    Result:  {'lat': 49.6786156, 'lng': -112.8601177} 4401 University Dr W, Lethbridge, AB T1K 3M4, Canada
    Searching for Capilano University ......
    Result:  {'lat': 49.317754, 'lng': -123.019085} 2055 Purcell Way, North Vancouver, BC V7J 3H5, Canada
    Searching for Emily Carr University of Art and Design ......
    Result:  {'lat': 49.267579, 'lng': -123.0925322} 520 E 1st Ave, Vancouver, BC V5T 0H2, Canada
    Searching for Kwantlen Polytechnic University ......
    Result:  {'lat': 49.1326553, 'lng': -122.8714821} 12666 72 Ave, Surrey, BC V3W 2M8, Canada
    Searching for Royal Roads University ......
    Result:  {'lat': 48.4342047, 'lng': -123.4739689} 2005 Sooke Rd, Victoria, BC V9B 5Y2, Canada
    Searching for Simon Fraser University ......
    Result:  {'lat': 49.2780937, 'lng': -122.9198833} 8888 University Dr, Burnaby, BC V5A 1S6, Canada
    Searching for Thompson Rivers University ......
    Result:  {'lat': 50.6712202, 'lng': -120.3666264} Thompson Rivers University, Kamloops, BC V2C, Canada
    Searching for University of British Columbia ......
    Result:  {'lat': 49.26060520000001, 'lng': -123.2459938} Vancouver, BC V6T 1Z4, Canada
    Searching for University of Victoria ......
    Result:  {'lat': 48.4634067, 'lng': -123.3116935} Victoria, BC V8P 5C2, Canada
    Searching for University of the Fraser Valley ......
    Result:  {'lat': 49.029051, 'lng': -122.285434} 33844 King Rd, Abbotsford, BC V2S 7M8, Canada
    Searching for University of Northern British Columbia ......
    Result:  {'lat': 53.8922034, 'lng': -122.8133607} 3333 University Way, Prince George, BC V2N 4Z9, Canada
    Searching for Vancouver Island University ......
    Result:  {'lat': 49.1573024, 'lng': -123.9664322} 900 Fifth St, Nanaimo, BC V9R 5S5, Canada
    Searching for Brandon University ......
    Result:  {'lat': 49.845352, 'lng': -99.962159} 270 18th St, Brandon, MB R7A 6A9, Canada
    Searching for University College of the North ......
    Result:  {'lat': 53.8193494, 'lng': -101.2364889} 436 7 St E, The Pas, MB R9A 1T4, Canada
    Searching for University of Manitoba ......
    Result:  {'lat': 49.8075008, 'lng': -97.1366259} 66 Chancellors Cir, Winnipeg, MB R3T 2N2, Canada
    Searching for University of Winnipeg ......
    Result:  {'lat': 49.89125430000001, 'lng': -97.153487} 515 Portage Ave, Winnipeg, MB R3B 2E9, Canada
    Searching for Mount Allison University ......
    Result:  {'lat': 45.8983184, 'lng': -64.3731004} 62 York St, Sackville, NB E4L 1E2, Canada
    Searching for St. Thomas University ......
    Result:  {'lat': 45.94401999999999, 'lng': -66.6462213} 51 Dineen Dr, Fredericton, NB E3B 5G3, Canada
    Searching for University of New Brunswick ......
    Result:  {'lat': 45.9455704, 'lng': -66.6408264} 3 Bailey Dr, Fredericton, NB E3B 5A3, Canada
    Searching for Université de Moncton ......
    Result:  {'lat': 46.1050904, 'lng': -64.78176189999999} 18 Antonine-Maillet Ave, Moncton, NB E1A 3E9, Canada
    Searching for Memorial University of Newfoundland ......
    Result:  {'lat': 47.5737975, 'lng': -52.7329053} 230 Elizabeth Ave, St. John's, NL A1C 5S7, Canada
    Searching for Acadia University ......
    Result:  {'lat': 45.09051609999999, 'lng': -64.364525} 10 Highland Ave, Wolfville, NS B4P 2R6, Canada
    Searching for Cape Breton University ......
    Result:  {'lat': 46.1706301, 'lng': -60.09349979999999} 1250 Grand Lake Rd, Sydney, NS B1P 6L2, Canada
    Searching for Dalhousie University ......
    Result:  {'lat': 44.63658119999999, 'lng': -63.59165549999999} 6299 South St, Halifax, NS B3H 4R2, Canada
    Searching for Mount Saint Vincent University ......
    Result:  {'lat': 44.6712012, 'lng': -63.6456043} 166 Bedford Hwy, Halifax, NS B3M 2J6, Canada
    Searching for Nova Scotia College of Art and Design University ......
    Result:  {'lat': 44.6495475, 'lng': -63.57415839999999} 5163 Duke St, Halifax, NS B3J 3J6, Canada
    Searching for Saint Francis Xavier University ......
    Result:  {'lat': 45.61773669999999, 'lng': -61.9953913} 4130 University Ave, Antigonish, NS B2G 2W5, Canada
    Searching for Saint Mary's University ......
    Result:  {'lat': 44.6313301, 'lng': -63.581457} 923 Robie St, Halifax, NS B3H 3C3, Canada
    Searching for Université Sainte-Anne ......
    Result:  {'lat': 44.3326798, 'lng': -66.1168753} 1695 Route 1, Church Point, NS B0W 1M0, Canada
    Searching for Algoma University ......
    Result:  {'lat': 46.5015289, 'lng': -84.2878694} 1520 Queen St E, Sault Ste. Marie, ON P6A 2G4, Canada
    Searching for Brock University ......
    Result:  {'lat': 43.1175731, 'lng': -79.2476925} 1812 Sir Isaac Brock Way, St. Catharines, ON L2S 3A1, Canada
    Searching for Carleton University ......
    Result:  {'lat': 45.3830819, 'lng': -75.6983121} Carleton University, Ottawa, ON, Canada
    Searching for Lakehead University ......
    Result:  {'lat': 48.42111080000001, 'lng': -89.2606994} 955 Oliver Rd, Thunder Bay, ON P7B 5E1, Canada
    Searching for Laurentian University ......
    Result:  {'lat': 46.4667708, 'lng': -80.9742332} 935 Ramsey Lake Rd, Sudbury, ON P3E 2C6, Canada
    Searching for McMaster University ......
    Result:  {'lat': 43.260879, 'lng': -79.9192254} 1280 Main St W, Hamilton, ON L8S 4L8, Canada
    Searching for Nipissing University ......
    Result:  {'lat': 46.3432094, 'lng': -79.49230179999999} 100 College Dr, North Bay, ON P1B 8L7, Canada
    Searching for Ontario College of Art and Design University ......
    Result:  {'lat': 43.65299359999999, 'lng': -79.39121659999999} 100 McCaul St, Toronto, ON M5T 1W1, Canada
    Searching for Queen's University at Kingston ......
    Result:  {'lat': 44.2252795, 'lng': -76.49514119999999} 99 University Ave, Kingston, ON K7L 3N6, Canada
    Searching for Royal Military College of Canada ......
    Result:  {'lat': 44.2338812, 'lng': -76.4675051} 13 General Crerar Crescent, Kingston, ON K7K 7B4, Canada
    Searching for Ryerson University ......
    Result:  {'lat': 43.6576585, 'lng': -79.3788017} 350 Victoria St, Toronto, ON M5B 2K3, Canada
    Searching for Trent University ......
    ==== Failure To Retrieve ====
    Searching for Université de l'Ontario français ......
    Result:  {'lat': 43.6444854, 'lng': -79.3690622} 7 Lower Jarvis St, Toronto, ON M5E 1Z2, Canada
    Searching for University of Guelph ......
    Result:  {'lat': 43.5327217, 'lng': -80.22618039999999} 50 Stone Rd E, Guelph, ON N1G 2W1, Canada
    Searching for Ontario Tech University ......
    Result:  {'lat': 43.9457579, 'lng': -78.8960092} 2000 Simcoe St N, Oshawa, ON L1G 0C5, Canada
    Searching for University of Ottawa ......
    Result:  {'lat': 45.42310639999999, 'lng': -75.68313289999999} 75 Laurier Ave E, Ottawa, ON K1N 6N5, Canada
    Searching for University of Toronto ......
    Result:  {'lat': 43.6628917, 'lng': -79.39565640000001} 27 King's College Cir, Toronto, ON M5S, Canada
    Searching for University of Waterloo ......
    Result:  {'lat': 43.4722854, 'lng': -80.5448576} 200 University Ave W, Waterloo, ON N2L 3G1, Canada
    Searching for University of Western Ontario ......
    Result:  {'lat': 43.0095971, 'lng': -81.2737336} 1151 Richmond St, London, ON N6A 3K7, Canada
    Searching for University of Windsor ......
    Result:  {'lat': 42.3043142, 'lng': -83.06603899999999} 401 Sunset Ave, Windsor, ON N9B 3P4, Canada
    Searching for Wilfrid Laurier University ......
    Result:  {'lat': 43.4739562, 'lng': -80.5277749} 75 University Ave W, Waterloo, ON N2L 3C5, Canada
    Searching for York University ......
    Result:  {'lat': 43.7734535, 'lng': -79.50186839999999} 4700 Keele St, Toronto, ON M3J 1P3, Canada
    Searching for University of Prince Edward Island ......
    Result:  {'lat': 46.257492, 'lng': -63.1375074} 550 University Ave, Charlottetown, PE C1A 4P3, Canada
    Searching for Bishop's University ......
    Result:  {'lat': 45.3628528, 'lng': -71.8456569} 2600 Rue College, Sherbrooke, QC J1M 1Z7, Canada
    Searching for Concordia University ......
    Result:  {'lat': 45.4945643, 'lng': -73.5773775} 1455 Boulevard de Maisonneuve O, Montréal, QC H3G 1M8, Canada
    Searching for École de technologie supérieure ......
    Result:  {'lat': 45.4945877, 'lng': -73.5622815} 1100 Rue Notre-Dame Ouest, Montréal, QC H3C 1K3, Canada
    Searching for École nationale d'administration publique ......
    Result:  {'lat': 46.8138009, 'lng': -71.2224769} 555 Boulevard Charest E, Québec, QC G1K 9E5, Canada
    Searching for Institut national de la recherche scientifique ......
    Result:  {'lat': 46.8130159, 'lng': -71.2240248} 490 Rue de la Couronne, Québec, QC G1K 9A9, Canada
    Searching for McGill University ......
    Result:  {'lat': 45.50478469999999, 'lng': -73.5771511} 845 Rue Sherbrooke Ouest, Montréal, QC H3A 0G4, Canada
    Searching for Université de Montréal ......
    Result:  {'lat': 45.5231104, 'lng': -73.6196605} 1375 Avenue Thérèse-Lavoie-Roux, Montréal, QC H2V 0B3, Canada
    Searching for Université de Sherbrooke ......
    Result:  {'lat': 45.3779433, 'lng': -71.929385} 2500 Boulevard de l'Université, Sherbrooke, QC J1K 2R1, Canada
    Searching for Université du Québec en Abitibi-Témiscamingue ......
    Result:  {'lat': 48.23057379999999, 'lng': -79.0082905} 445 Boulevard de l'Université, Rouyn-Noranda, QC J9X 5E4, Canada
    Searching for Université du Québec en Outaouais ......
    Result:  {'lat': 45.422466, 'lng': -75.7387016} 283 Boul Alexandre-Taché, Gatineau, QC J8X 3X7, Canada
    Searching for Université du Québec à Chicoutimi ......
    Result:  {'lat': 48.419008, 'lng': -71.052621} 555 Boulevard de l'Université, Chicoutimi, QC G7H 2B1, Canada
    Searching for Université du Québec à Montréal ......
    Result:  {'lat': 45.5125995, 'lng': -73.56059549999999} 405 Rue Sainte-Catherine Est, Montréal, QC H2L 2C4, Canada
    Searching for Université du Québec à Rimouski ......
    Result:  {'lat': 48.45256029999999, 'lng': -68.51213179999999} 300 Allée des Ursulines, Rimouski, QC G5L 3A1, Canada
    Searching for Université du Québec à Trois-Rivières ......
    Result:  {'lat': 46.3472006, 'lng': -72.57709369999999} 3351 Boulevard des Forges, Trois-Rivières, QC G8Z 4M3, Canada
    Searching for Université Laval ......
    Result:  {'lat': 46.78174629999999, 'lng': -71.2747424} 2325 Rue de l'Université, Québec, QC G1V 0A6, Canada
    Searching for University of Regina ......
    Result:  {'lat': 50.4154542, 'lng': -104.5878302} 3737 Wascana Pkwy, Regina, SK S4S 0A2, Canada
    Searching for University of Saskatchewan ......
    Result:  {'lat': 52.1334003, 'lng': -106.6313582} Saskatoon, SK S7N, Canada
    Searching for Fairleigh Dickinson University (branch) ......
    Result:  {'lat': 49.277696, 'lng': -123.11555} 842 Cambie St, Vancouver, BC V6B 2P6, Canada
    Searching for New York Institute of Technology (branch) ......
    Result:  {'lat': 49.2615238, 'lng': -123.0420669} 2955 Virtual Way, Vancouver, BC V5M 4X3, Canada
    Searching for Quest University ......
    Result:  {'lat': 49.7381913, 'lng': -123.1004121} 3200 University Blvd, Squamish, BC V8B 0N8, Canada
    Searching for Niagara University (branch) ......
    Result:  {'lat': 43.1369816, 'lng': -79.0349955} 5795 Lewiston Rd, Niagara University, NY 14109, USA
    Searching for Trinity Western University ......
    Result:  {'lat': 49.14069569999999, 'lng': -122.6019912} 7600 Glover Rd, Langley City, BC V2Y 1Y1, Canada
    Searching for University Canada West ......
    Result:  {'lat': 49.2842378, 'lng': -123.1144255} 626 W Pender St #100, Vancouver, BC V6B 1V9, Canada
    Searching for Booth University College ......
    Result:  {'lat': 49.8929267, 'lng': -97.1515364} 447 Webb Pl, Winnipeg, MB R3B 2P2, Canada
    Searching for Canadian Mennonite University ......
    Result:  {'lat': 49.86001359999999, 'lng': -97.2321798} 500 Shaftesbury Blvd, Winnipeg, MB R3P 2N2, Canada
    Searching for Kingswood University ......
    Result:  {'lat': 45.7253572, 'lng': -65.5241285} 26 Western St, Sussex, NB E4E 1E6, Canada
    Searching for Crandall University ......
    Result:  {'lat': 46.1346326, 'lng': -64.8612334} 333 Gorge Rd, Moncton, NB E1G 3H9, Canada
    Searching for St. Stephen's University ......
    Result:  {'lat': 45.1929557, 'lng': -67.2819203} 8 Main St, Saint Stephen, NB E3L 3E2, Canada
    Searching for University of Fredericton ......
    Result:  {'lat': 45.9455704, 'lng': -66.6408264} 3 Bailey Dr, Fredericton, NB E3B 5A3, Canada
    Searching for Atlantic School of Theology ......
    Result:  {'lat': 44.6268264, 'lng': -63.5805033} 660 Francklyn St, Halifax, NS B3H 3B6, Canada
    Searching for Tyndale University ......
    Result:  {'lat': 43.7968511, 'lng': -79.3921865} 3377 Bayview Ave, North York, ON M2M 3S4, Canada
    Searching for Redeemer University College ......
    Result:  {'lat': 43.2086769, 'lng': -79.94914039999999} 777 Garner Rd E, Ancaster, ON L9K 1J4, Canada
    Searching for The King's University ......
    Result:  {'lat': 53.5254179, 'lng': -113.4167524} 9125 50 St NW, Edmonton, AB T6B 2H3, Canada
    Done!



```python
df_univ['Address'] = address_ls
df_univ['Latitude'] = lat_ls
df_univ['Longitude'] = lng_ls
df_univ
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>City</th>
      <th>Province</th>
      <th>Address</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alberta University of the Arts</td>
      <td>Calgary</td>
      <td>Alberta</td>
      <td>1407 14 Ave NW, Calgary, AB T2N 4R3, Canada</td>
      <td>51.061571</td>
      <td>-114.092098</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Athabasca University</td>
      <td>Athabasca, Calgary, Edmonton</td>
      <td>Alberta</td>
      <td>1 University Dr, Athabasca, AB T9S 3A3, Canada</td>
      <td>54.714955</td>
      <td>-113.308545</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MacEwan University</td>
      <td>Edmonton</td>
      <td>Alberta</td>
      <td>10700 104 Ave NW, Edmonton, AB T5J 4S2, Canada</td>
      <td>53.547054</td>
      <td>-113.506372</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mount Royal University</td>
      <td>Calgary</td>
      <td>Alberta</td>
      <td>30 Mt Royal Cir SW, Calgary, AB T3E 7C9, Canada</td>
      <td>51.011010</td>
      <td>-114.129778</td>
    </tr>
    <tr>
      <th>4</th>
      <td>University of Alberta</td>
      <td>Edmonton, Camrose, Calgary</td>
      <td>Alberta</td>
      <td>116 St &amp; 85 Ave, Edmonton, AB T6G 2R3, Canada</td>
      <td>53.523219</td>
      <td>-113.526319</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>86</th>
      <td>University of Fredericton</td>
      <td>Fredericton</td>
      <td>New Brunswick</td>
      <td>3 Bailey Dr, Fredericton, NB E3B 5A3, Canada</td>
      <td>45.945570</td>
      <td>-66.640826</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Atlantic School of Theology</td>
      <td>Halifax</td>
      <td>Nova Scotia</td>
      <td>660 Francklyn St, Halifax, NS B3H 3B6, Canada</td>
      <td>44.626826</td>
      <td>-63.580503</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Tyndale University</td>
      <td>Toronto</td>
      <td>Ontario</td>
      <td>3377 Bayview Ave, North York, ON M2M 3S4, Canada</td>
      <td>43.796851</td>
      <td>-79.392186</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Redeemer University College</td>
      <td>Ancaster</td>
      <td>Ontario</td>
      <td>777 Garner Rd E, Ancaster, ON L9K 1J4, Canada</td>
      <td>43.208677</td>
      <td>-79.949140</td>
    </tr>
    <tr>
      <th>90</th>
      <td>The King's University</td>
      <td>Edmonton</td>
      <td>Alberta</td>
      <td>9125 50 St NW, Edmonton, AB T6B 2H3, Canada</td>
      <td>53.525418</td>
      <td>-113.416752</td>
    </tr>
  </tbody>
</table>
<p>91 rows × 6 columns</p>
</div>



We could see that we failed to retrieve the coordinate of Trent University, so we remove that record.


```python
df_univ.dropna(axis=0, how='any', inplace=True)

df_univ
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>City</th>
      <th>Province</th>
      <th>Address</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alberta University of the Arts</td>
      <td>Calgary</td>
      <td>Alberta</td>
      <td>1407 14 Ave NW, Calgary, AB T2N 4R3, Canada</td>
      <td>51.061571</td>
      <td>-114.092098</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Athabasca University</td>
      <td>Athabasca, Calgary, Edmonton</td>
      <td>Alberta</td>
      <td>1 University Dr, Athabasca, AB T9S 3A3, Canada</td>
      <td>54.714955</td>
      <td>-113.308545</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MacEwan University</td>
      <td>Edmonton</td>
      <td>Alberta</td>
      <td>10700 104 Ave NW, Edmonton, AB T5J 4S2, Canada</td>
      <td>53.547054</td>
      <td>-113.506372</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mount Royal University</td>
      <td>Calgary</td>
      <td>Alberta</td>
      <td>30 Mt Royal Cir SW, Calgary, AB T3E 7C9, Canada</td>
      <td>51.011010</td>
      <td>-114.129778</td>
    </tr>
    <tr>
      <th>4</th>
      <td>University of Alberta</td>
      <td>Edmonton, Camrose, Calgary</td>
      <td>Alberta</td>
      <td>116 St &amp; 85 Ave, Edmonton, AB T6G 2R3, Canada</td>
      <td>53.523219</td>
      <td>-113.526319</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>86</th>
      <td>University of Fredericton</td>
      <td>Fredericton</td>
      <td>New Brunswick</td>
      <td>3 Bailey Dr, Fredericton, NB E3B 5A3, Canada</td>
      <td>45.945570</td>
      <td>-66.640826</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Atlantic School of Theology</td>
      <td>Halifax</td>
      <td>Nova Scotia</td>
      <td>660 Francklyn St, Halifax, NS B3H 3B6, Canada</td>
      <td>44.626826</td>
      <td>-63.580503</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Tyndale University</td>
      <td>Toronto</td>
      <td>Ontario</td>
      <td>3377 Bayview Ave, North York, ON M2M 3S4, Canada</td>
      <td>43.796851</td>
      <td>-79.392186</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Redeemer University College</td>
      <td>Ancaster</td>
      <td>Ontario</td>
      <td>777 Garner Rd E, Ancaster, ON L9K 1J4, Canada</td>
      <td>43.208677</td>
      <td>-79.949140</td>
    </tr>
    <tr>
      <th>90</th>
      <td>The King's University</td>
      <td>Edmonton</td>
      <td>Alberta</td>
      <td>9125 50 St NW, Edmonton, AB T6B 2H3, Canada</td>
      <td>53.525418</td>
      <td>-113.416752</td>
    </tr>
  </tbody>
</table>
<p>90 rows × 6 columns</p>
</div>




```python
df_univ.reset_index(drop=True, inplace=True)
df_univ
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>City</th>
      <th>Province</th>
      <th>Address</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alberta University of the Arts</td>
      <td>Calgary</td>
      <td>Alberta</td>
      <td>1407 14 Ave NW, Calgary, AB T2N 4R3, Canada</td>
      <td>51.061571</td>
      <td>-114.092098</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Athabasca University</td>
      <td>Athabasca, Calgary, Edmonton</td>
      <td>Alberta</td>
      <td>1 University Dr, Athabasca, AB T9S 3A3, Canada</td>
      <td>54.714955</td>
      <td>-113.308545</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MacEwan University</td>
      <td>Edmonton</td>
      <td>Alberta</td>
      <td>10700 104 Ave NW, Edmonton, AB T5J 4S2, Canada</td>
      <td>53.547054</td>
      <td>-113.506372</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mount Royal University</td>
      <td>Calgary</td>
      <td>Alberta</td>
      <td>30 Mt Royal Cir SW, Calgary, AB T3E 7C9, Canada</td>
      <td>51.011010</td>
      <td>-114.129778</td>
    </tr>
    <tr>
      <th>4</th>
      <td>University of Alberta</td>
      <td>Edmonton, Camrose, Calgary</td>
      <td>Alberta</td>
      <td>116 St &amp; 85 Ave, Edmonton, AB T6G 2R3, Canada</td>
      <td>53.523219</td>
      <td>-113.526319</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>85</th>
      <td>University of Fredericton</td>
      <td>Fredericton</td>
      <td>New Brunswick</td>
      <td>3 Bailey Dr, Fredericton, NB E3B 5A3, Canada</td>
      <td>45.945570</td>
      <td>-66.640826</td>
    </tr>
    <tr>
      <th>86</th>
      <td>Atlantic School of Theology</td>
      <td>Halifax</td>
      <td>Nova Scotia</td>
      <td>660 Francklyn St, Halifax, NS B3H 3B6, Canada</td>
      <td>44.626826</td>
      <td>-63.580503</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Tyndale University</td>
      <td>Toronto</td>
      <td>Ontario</td>
      <td>3377 Bayview Ave, North York, ON M2M 3S4, Canada</td>
      <td>43.796851</td>
      <td>-79.392186</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Redeemer University College</td>
      <td>Ancaster</td>
      <td>Ontario</td>
      <td>777 Garner Rd E, Ancaster, ON L9K 1J4, Canada</td>
      <td>43.208677</td>
      <td>-79.949140</td>
    </tr>
    <tr>
      <th>89</th>
      <td>The King's University</td>
      <td>Edmonton</td>
      <td>Alberta</td>
      <td>9125 50 St NW, Edmonton, AB T6B 2H3, Canada</td>
      <td>53.525418</td>
      <td>-113.416752</td>
    </tr>
  </tbody>
</table>
<p>90 rows × 6 columns</p>
</div>



Here we quickly explain why we give up using the module `geocoder` to retrieve the latitude and longitude of a university.


```python
!conda install -c conda-forge geocoder --yes
import geocoder
```

    Collecting package metadata (current_repodata.json): done
    Solving environment: done
    
    # All requested packages already installed.
    



```python
# Get some coordinates of the universities as an example for clarification
uni_dc = {}
count = 0
for name, province in zip(df_univ["Name"], df_univ["Province"]):
    lc = "{}, {}, Canada".format(name, province)
    print("Searching for {} ......".format(name))
    g = geocoder.arcgis(lc)
    latlng = g.latlng
    print("Result: ", latlng)
    uni_dc[name] = latlng
    count += 1
    if count > 5:
        break
print("Done!")
pd.DataFrame(uni_dc).T
```

    Searching for Alberta University of the Arts ......
    Result:  [51.06438000000003, -114.09211999999997]
    Searching for Athabasca University ......
    Result:  [53.522890000000075, -113.52626999999995]
    Searching for MacEwan University ......
    Result:  [53.53948006665105, -113.49235997778297]
    Searching for Mount Royal University ......
    Result:  [51.01228000000003, -114.13238999999999]
    Searching for University of Alberta ......
    Result:  [53.522890000000075, -113.52626999999995]
    Searching for University of Calgary ......
    Result:  [51.07663000000008, -114.13208999999995]
    Done!





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alberta University of the Arts</th>
      <td>51.06438</td>
      <td>-114.09212</td>
    </tr>
    <tr>
      <th>Athabasca University</th>
      <td>53.52289</td>
      <td>-113.52627</td>
    </tr>
    <tr>
      <th>MacEwan University</th>
      <td>53.53948</td>
      <td>-113.49236</td>
    </tr>
    <tr>
      <th>Mount Royal University</th>
      <td>51.01228</td>
      <td>-114.13239</td>
    </tr>
    <tr>
      <th>University of Alberta</th>
      <td>53.52289</td>
      <td>-113.52627</td>
    </tr>
    <tr>
      <th>University of Calgary</th>
      <td>51.07663</td>
      <td>-114.13209</td>
    </tr>
  </tbody>
</table>
</div>



We could see from the result that Athabasca University and University of Alberta share the same coordinate. It is impossible. We further check this location on the map, and it is clear that [53.52289, -113.52627] is the coordinate of University of Alberta.


```python
import folium
map = folium.Map(location=[53.52289, -113.52627], zoom_start=14)
folium.Marker(location=[53.52289, -113.52627]).add_to(map)
map
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfMWM0NzY1ZmQxOGY3NDI5Y2E0NzBjMmQyOTRiZGJhZGQgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzFjNDc2NWZkMThmNzQyOWNhNDcwYzJkMjk0YmRiYWRkIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF8xYzQ3NjVmZDE4Zjc0MjljYTQ3MGMyZDI5NGJkYmFkZCA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF8xYzQ3NjVmZDE4Zjc0MjljYTQ3MGMyZDI5NGJkYmFkZCcsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNTMuNTIyODksLTExMy41MjYyN10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB6b29tOiAxNCwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1heEJvdW5kczogYm91bmRzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbGF5ZXJzOiBbXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KTsKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfMzQ4NmI2OTBjOTNlNDg0NjljOGE1OGFkNWMyNGVlY2MgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICdodHRwczovL3tzfS50aWxlLm9wZW5zdHJlZXRtYXAub3JnL3t6fS97eH0ve3l9LnBuZycsCiAgICAgICAgICAgICAgICB7CiAgImF0dHJpYnV0aW9uIjogbnVsbCwKICAiZGV0ZWN0UmV0aW5hIjogZmFsc2UsCiAgIm1heFpvb20iOiAxOCwKICAibWluWm9vbSI6IDEsCiAgIm5vV3JhcCI6IGZhbHNlLAogICJzdWJkb21haW5zIjogImFiYyIKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMWM0NzY1ZmQxOGY3NDI5Y2E0NzBjMmQyOTRiZGJhZGQpOwogICAgICAgIAogICAgCgogICAgICAgICAgICB2YXIgbWFya2VyX2IxMmMxZDZjNDk1OTQ0Y2RiMjA4YWI0MTZjNmRmYjllID0gTC5tYXJrZXIoCiAgICAgICAgICAgICAgICBbNTMuNTIyODksLTExMy41MjYyN10sCiAgICAgICAgICAgICAgICB7CiAgICAgICAgICAgICAgICAgICAgaWNvbjogbmV3IEwuSWNvbi5EZWZhdWx0KCkKICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzFjNDc2NWZkMThmNzQyOWNhNDcwYzJkMjk0YmRiYWRkKTsKICAgICAgICAgICAgCjwvc2NyaXB0Pg== onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



We use the coordinate we got for Athabasca University and display it on the map. In fact, these two universities are quite far from each other.


```python
# Add the marker pinned to the coordinate of Athabasca University
map = folium.Map(location=[53.52289, -113.52627], zoom_start=6)
folium.Marker(location=[53.52289, -113.52627], popup='University of Alberta').add_to(map)
folium.Marker(location=[54.714955, -113.308545], popup='Athabasca University').add_to(map)
map
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfMWMxYTY2NDUyZjBhNDFkMzg5OTgwZTBlMjYxZWU1OWMgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzFjMWE2NjQ1MmYwYTQxZDM4OTk4MGUwZTI2MWVlNTljIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF8xYzFhNjY0NTJmMGE0MWQzODk5ODBlMGUyNjFlZTU5YyA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF8xYzFhNjY0NTJmMGE0MWQzODk5ODBlMGUyNjFlZTU5YycsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNTMuNTIyODksLTExMy41MjYyN10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB6b29tOiA2LAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWF4Qm91bmRzOiBib3VuZHMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBsYXllcnM6IFtdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgd29ybGRDb3B5SnVtcDogZmFsc2UsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjcnM6IEwuQ1JTLkVQU0czODU3CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pOwogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgdGlsZV9sYXllcl80NWIzNWM0MmE2NjU0YTkwYmEyNzY1YjZkMzlmMjAwNCA9IEwudGlsZUxheWVyKAogICAgICAgICAgICAgICAgJ2h0dHBzOi8ve3N9LnRpbGUub3BlbnN0cmVldG1hcC5vcmcve3p9L3t4fS97eX0ucG5nJywKICAgICAgICAgICAgICAgIHsKICAiYXR0cmlidXRpb24iOiBudWxsLAogICJkZXRlY3RSZXRpbmEiOiBmYWxzZSwKICAibWF4Wm9vbSI6IDE4LAogICJtaW5ab29tIjogMSwKICAibm9XcmFwIjogZmFsc2UsCiAgInN1YmRvbWFpbnMiOiAiYWJjIgp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xYzFhNjY0NTJmMGE0MWQzODk5ODBlMGUyNjFlZTU5Yyk7CiAgICAgICAgCiAgICAKCiAgICAgICAgICAgIHZhciBtYXJrZXJfNWE1MmEzYzgwMWY4NGMwZTg5ZWJhODk3M2E5NTIxOTYgPSBMLm1hcmtlcigKICAgICAgICAgICAgICAgIFs1My41MjI4OSwtMTEzLjUyNjI3XSwKICAgICAgICAgICAgICAgIHsKICAgICAgICAgICAgICAgICAgICBpY29uOiBuZXcgTC5JY29uLkRlZmF1bHQoKQogICAgICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMWMxYTY2NDUyZjBhNDFkMzg5OTgwZTBlMjYxZWU1OWMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTUxY2M0ZmRhODFkNDZlMzk1ZWZkMDc2NGExODIwOTkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzJkOWRiNmZjNWYzNDlmMTllNjNiZmRlNTNjMTkzOWUgPSAkKCc8ZGl2IGlkPSJodG1sX2MyZDlkYjZmYzVmMzQ5ZjE5ZTYzYmZkZTUzYzE5MzllIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIEFsYmVydGE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk1MWNjNGZkYTgxZDQ2ZTM5NWVmZDA3NjRhMTgyMDk5LnNldENvbnRlbnQoaHRtbF9jMmQ5ZGI2ZmM1ZjM0OWYxOWU2M2JmZGU1M2MxOTM5ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgbWFya2VyXzVhNTJhM2M4MDFmODRjMGU4OWViYTg5NzNhOTUyMTk2LmJpbmRQb3B1cChwb3B1cF85NTFjYzRmZGE4MWQ0NmUzOTVlZmQwNzY0YTE4MjA5OSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAoKICAgICAgICAgICAgdmFyIG1hcmtlcl9mZjYwZDM1ZDZhODU0ZTkwYmVkNWY3OWY3NjQ1YmU1OSA9IEwubWFya2VyKAogICAgICAgICAgICAgICAgWzU0LjcxNDk1NSwtMTEzLjMwODU0NV0sCiAgICAgICAgICAgICAgICB7CiAgICAgICAgICAgICAgICAgICAgaWNvbjogbmV3IEwuSWNvbi5EZWZhdWx0KCkKICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzFjMWE2NjQ1MmYwYTQxZDM4OTk4MGUwZTI2MWVlNTljKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYwNDg4YzFkZWYwNTRjY2FhYTZjZjhhMTQ2NWMxNzc2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzMwZTI0ZjBiMDk3NzQzYjU5OGJmNzRmYzNmNzg3NTJhID0gJCgnPGRpdiBpZD0iaHRtbF8zMGUyNGYwYjA5Nzc0M2I1OThiZjc0ZmMzZjc4NzUyYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QXRoYWJhc2NhIFVuaXZlcnNpdHk8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYwNDg4YzFkZWYwNTRjY2FhYTZjZjhhMTQ2NWMxNzc2LnNldENvbnRlbnQoaHRtbF8zMGUyNGYwYjA5Nzc0M2I1OThiZjc0ZmMzZjc4NzUyYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgbWFya2VyX2ZmNjBkMzVkNmE4NTRlOTBiZWQ1Zjc5Zjc2NDViZTU5LmJpbmRQb3B1cChwb3B1cF82MDQ4OGMxZGVmMDU0Y2NhYWE2Y2Y4YTE0NjVjMTc3Nik7CgogICAgICAgICAgICAKICAgICAgICAKPC9zY3JpcHQ+ onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



The result told us that using this way to find the coordinate is not accurate. In fact, I have checked the complete result of the coordinates of all the universities obtained using the `geocoder`, and have noticed many duplicate values.

This is not the fault of the module. The reason is that I did not provide the formatted address but only the name of a university to the API. According to the documentation of the `geocoder`, the address is necessary to get the correct result.

#### 3.1.3. Recommendations of Food around the Universities

After obtaining the coordinates of the universities, we use Foursquare’s API to get the recommendations in the food section.


```python
CLIENT_ID = 'XPY14AS3GHQDHDO3HTF1CNG5SOXOWZARINMBXYAU4UOYXIZF'
CLIENT_SECRET = 'YXGDSZG1BMYTEZFHLJL5CQZ0O3F2LMBJWNQGEFPPXSMFDTZO'
VERSION = '20200516'
```


```python
# Define a function to help us get the recommendations

def university_food(university, lat, lng, section = 'food', radius = 500, limit = 10):
    url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&section={}&limit={}'\
                                   .format(CLIENT_ID, CLIENT_SECRET, VERSION, lat, lng, radius, section, limit)
    json = requests.get(url).json()
    
    try:
        df = pd.json_normalize(json['response']['groups'][0]['items'])
        df['venue.categories'] = [info[0]['name'] for info in df['venue.categories']]
        df = df[['venue.name', 'venue.location.lat', 'venue.location.lng', 'venue.location.distance', 'venue.categories']]
        df.columns = [col.replace("."," ").capitalize() for col in df.columns]
        df.insert(0, "University", university)
    except:
        df = pd.DataFrame([{"University":university}])
            
    return df
```


```python
df_recomm = pd.DataFrame()
for university, lat, lng in zip(df_univ['Name'], df_univ['Latitude'], df_univ['Longitude']):
    print("Searching for {} ....".format(university))
    df = university_food(university, lat, lng, radius = 3000)
    df_recomm = pd.concat([df_recomm, df],axis=0)
df_recomm
```

    Searching for Alberta University of the Arts ....
    Searching for Athabasca University ....
    Searching for MacEwan University ....
    Searching for Mount Royal University ....
    Searching for University of Alberta ....
    Searching for University of Calgary ....
    Searching for University of Lethbridge ....
    Searching for Capilano University ....
    Searching for Emily Carr University of Art and Design ....
    Searching for Kwantlen Polytechnic University ....
    Searching for Royal Roads University ....
    Searching for Simon Fraser University ....
    Searching for Thompson Rivers University ....
    Searching for University of British Columbia ....
    Searching for University of Victoria ....
    Searching for University of the Fraser Valley ....
    Searching for University of Northern British Columbia ....
    Searching for Vancouver Island University ....
    Searching for Brandon University ....
    Searching for University College of the North ....
    Searching for University of Manitoba ....
    Searching for University of Winnipeg ....
    Searching for Mount Allison University ....
    Searching for St. Thomas University ....
    Searching for University of New Brunswick ....
    Searching for Université de Moncton ....
    Searching for Memorial University of Newfoundland ....
    Searching for Acadia University ....
    Searching for Cape Breton University ....
    Searching for Dalhousie University ....
    Searching for Mount Saint Vincent University ....
    Searching for Nova Scotia College of Art and Design University ....
    Searching for Saint Francis Xavier University ....
    Searching for Saint Mary's University ....
    Searching for Université Sainte-Anne ....
    Searching for Algoma University ....
    Searching for Brock University ....
    Searching for Carleton University ....
    Searching for Lakehead University ....
    Searching for Laurentian University ....
    Searching for McMaster University ....
    Searching for Nipissing University ....
    Searching for Ontario College of Art and Design University ....
    Searching for Queen's University at Kingston ....
    Searching for Royal Military College of Canada ....
    Searching for Ryerson University ....
    Searching for Université de l'Ontario français ....
    Searching for University of Guelph ....
    Searching for Ontario Tech University ....
    Searching for University of Ottawa ....
    Searching for University of Toronto ....
    Searching for University of Waterloo ....
    Searching for University of Western Ontario ....
    Searching for University of Windsor ....
    Searching for Wilfrid Laurier University ....
    Searching for York University ....
    Searching for University of Prince Edward Island ....
    Searching for Bishop's University ....
    Searching for Concordia University ....
    Searching for École de technologie supérieure ....
    Searching for École nationale d'administration publique ....
    Searching for Institut national de la recherche scientifique ....
    Searching for McGill University ....
    Searching for Université de Montréal ....
    Searching for Université de Sherbrooke ....
    Searching for Université du Québec en Abitibi-Témiscamingue ....
    Searching for Université du Québec en Outaouais ....
    Searching for Université du Québec à Chicoutimi ....
    Searching for Université du Québec à Montréal ....
    Searching for Université du Québec à Rimouski ....
    Searching for Université du Québec à Trois-Rivières ....
    Searching for Université Laval ....
    Searching for University of Regina ....
    Searching for University of Saskatchewan ....
    Searching for Fairleigh Dickinson University (branch) ....
    Searching for New York Institute of Technology (branch) ....
    Searching for Quest University ....
    Searching for Niagara University (branch) ....
    Searching for Trinity Western University ....
    Searching for University Canada West ....
    Searching for Booth University College ....
    Searching for Canadian Mennonite University ....
    Searching for Kingswood University ....
    Searching for Crandall University ....
    Searching for St. Stephen's University ....
    Searching for University of Fredericton ....
    Searching for Atlantic School of Theology ....
    Searching for Tyndale University ....
    Searching for Redeemer University College ....
    Searching for The King's University ....





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>University</th>
      <th>Venue name</th>
      <th>Venue location lat</th>
      <th>Venue location lng</th>
      <th>Venue location distance</th>
      <th>Venue categories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alberta University of the Arts</td>
      <td>Jimmy's A&amp;A Deli</td>
      <td>51.070299</td>
      <td>-114.092472</td>
      <td>971.0</td>
      <td>Mediterranean Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alberta University of the Arts</td>
      <td>Vendome Cafe</td>
      <td>51.055138</td>
      <td>-114.083323</td>
      <td>943.0</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alberta University of the Arts</td>
      <td>Hayden Block</td>
      <td>51.052595</td>
      <td>-114.088226</td>
      <td>1035.0</td>
      <td>BBQ Joint</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alberta University of the Arts</td>
      <td>Wow Chicken</td>
      <td>51.054881</td>
      <td>-114.085833</td>
      <td>864.0</td>
      <td>Korean Restaurant</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alberta University of the Arts</td>
      <td>Peppino</td>
      <td>51.052509</td>
      <td>-114.090946</td>
      <td>1011.0</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>The King's University</td>
      <td>A&amp;W</td>
      <td>53.541615</td>
      <td>-113.417087</td>
      <td>1803.0</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>6</th>
      <td>The King's University</td>
      <td>Sabu Sushi Bar</td>
      <td>53.518032</td>
      <td>-113.442067</td>
      <td>1866.0</td>
      <td>Sushi Restaurant</td>
    </tr>
    <tr>
      <th>7</th>
      <td>The King's University</td>
      <td>Subway</td>
      <td>53.525481</td>
      <td>-113.444089</td>
      <td>1809.0</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>8</th>
      <td>The King's University</td>
      <td>Fargo's</td>
      <td>53.540761</td>
      <td>-113.424622</td>
      <td>1785.0</td>
      <td>Steakhouse</td>
    </tr>
    <tr>
      <th>9</th>
      <td>The King's University</td>
      <td>Sawmill Banquet &amp; Catering Centre</td>
      <td>53.512940</td>
      <td>-113.401028</td>
      <td>1735.0</td>
      <td>Breakfast Spot</td>
    </tr>
  </tbody>
</table>
<p>856 rows × 6 columns</p>
</div>




```python
df_recomm.reset_index(drop=True, inplace=True)
df_recomm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>University</th>
      <th>Venue name</th>
      <th>Venue location lat</th>
      <th>Venue location lng</th>
      <th>Venue location distance</th>
      <th>Venue categories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alberta University of the Arts</td>
      <td>Jimmy's A&amp;A Deli</td>
      <td>51.070299</td>
      <td>-114.092472</td>
      <td>971.0</td>
      <td>Mediterranean Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alberta University of the Arts</td>
      <td>Vendome Cafe</td>
      <td>51.055138</td>
      <td>-114.083323</td>
      <td>943.0</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alberta University of the Arts</td>
      <td>Hayden Block</td>
      <td>51.052595</td>
      <td>-114.088226</td>
      <td>1035.0</td>
      <td>BBQ Joint</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alberta University of the Arts</td>
      <td>Wow Chicken</td>
      <td>51.054881</td>
      <td>-114.085833</td>
      <td>864.0</td>
      <td>Korean Restaurant</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alberta University of the Arts</td>
      <td>Peppino</td>
      <td>51.052509</td>
      <td>-114.090946</td>
      <td>1011.0</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>851</th>
      <td>The King's University</td>
      <td>A&amp;W</td>
      <td>53.541615</td>
      <td>-113.417087</td>
      <td>1803.0</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>852</th>
      <td>The King's University</td>
      <td>Sabu Sushi Bar</td>
      <td>53.518032</td>
      <td>-113.442067</td>
      <td>1866.0</td>
      <td>Sushi Restaurant</td>
    </tr>
    <tr>
      <th>853</th>
      <td>The King's University</td>
      <td>Subway</td>
      <td>53.525481</td>
      <td>-113.444089</td>
      <td>1809.0</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>854</th>
      <td>The King's University</td>
      <td>Fargo's</td>
      <td>53.540761</td>
      <td>-113.424622</td>
      <td>1785.0</td>
      <td>Steakhouse</td>
    </tr>
    <tr>
      <th>855</th>
      <td>The King's University</td>
      <td>Sawmill Banquet &amp; Catering Centre</td>
      <td>53.512940</td>
      <td>-113.401028</td>
      <td>1735.0</td>
      <td>Breakfast Spot</td>
    </tr>
  </tbody>
</table>
<p>856 rows × 6 columns</p>
</div>




```python
# Avoid the data in API changed
df_recomm.to_csv("Recommended Places@0512.csv", encoding="utf_8_sig")
```

Currently, each row is a recommendation for the specific university. To facilitate our analysis, we transform the table to be each row is an unique university, with the counts of each type of venues as columns.


```python
# Get dummies by venue categories
df_recomm_dummies = pd.get_dummies(data=df_recomm, columns=['Venue categories'], dummy_na=True, prefix="", prefix_sep="")

# Calculate the number of each venue type of each university
df_recomm_count = df_recomm_dummies.groupby(by=['University'], axis=0, sort=False).sum()

# Drop columns with unmeaningful values
df_recomm_count = df_recomm_count.iloc[:,3:]

df_recomm_count.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Afghan Restaurant</th>
      <th>American Restaurant</th>
      <th>Asian Restaurant</th>
      <th>BBQ Joint</th>
      <th>Bagel Shop</th>
      <th>Bakery</th>
      <th>Belgian Restaurant</th>
      <th>Bistro</th>
      <th>Brazilian Restaurant</th>
      <th>Breakfast Spot</th>
      <th>...</th>
      <th>Sushi Restaurant</th>
      <th>Taco Place</th>
      <th>Tapas Restaurant</th>
      <th>Thai Restaurant</th>
      <th>Theme Restaurant</th>
      <th>Turkish Restaurant</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Vietnamese Restaurant</th>
      <th>Wings Joint</th>
      <th>nan</th>
    </tr>
    <tr>
      <th>University</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alberta University of the Arts</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Athabasca University</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>MacEwan University</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mount Royal University</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>University of Alberta</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 68 columns</p>
</div>




```python
# Calculate the percentage of each venue type of each university
df_recomm_perc = df_recomm_count.div(df_recomm_count.sum(axis=1), axis='index', fill_value=None)
df_recomm_perc.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Afghan Restaurant</th>
      <th>American Restaurant</th>
      <th>Asian Restaurant</th>
      <th>BBQ Joint</th>
      <th>Bagel Shop</th>
      <th>Bakery</th>
      <th>Belgian Restaurant</th>
      <th>Bistro</th>
      <th>Brazilian Restaurant</th>
      <th>Breakfast Spot</th>
      <th>...</th>
      <th>Sushi Restaurant</th>
      <th>Taco Place</th>
      <th>Tapas Restaurant</th>
      <th>Thai Restaurant</th>
      <th>Theme Restaurant</th>
      <th>Turkish Restaurant</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Vietnamese Restaurant</th>
      <th>Wings Joint</th>
      <th>nan</th>
    </tr>
    <tr>
      <th>University</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alberta University of the Arts</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Athabasca University</th>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.166667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>MacEwan University</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Mount Royal University</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>University of Alberta</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 68 columns</p>
</div>



With these two tables df_recomm_count and df_recomm_perc we can process our analysis

### 3.2. Exploratory Data Analysis

Before carrying out the clustering, we take two universities and obtain the recommendations nearby to get some insights.


```python
for university in df_recomm_count.iloc[0:2,:].iterrows():
    print("------- {} ---------".format(university[0]))
    venues = university[1].sort_values(ascending=False)
    venues = venues[venues>0]
    print(venues)
```

    ------- Alberta University of the Arts ---------
    Gastropub                   1
    Diner                       1
    BBQ Joint                   1
    Italian Restaurant          1
    Bakery                      1
    Japanese Restaurant         1
    Korean Restaurant           1
    Sushi Restaurant            1
    Café                        1
    Mediterranean Restaurant    1
    Name: Alberta University of the Arts, dtype: uint8
    ------- Athabasca University ---------
    Sandwich Place         2
    American Restaurant    1
    Asian Restaurant       1
    Gastropub              1
    Burger Joint           1
    Name: Athabasca University, dtype: uint8



```python
for university in df_recomm_perc.iloc[0:2,:].iterrows():
    print("------- {} ---------".format(university[0]))
    venues = university[1].sort_values(ascending=False)
    venues = venues[venues>0]
    print(venues)
```

    ------- Alberta University of the Arts ---------
    Gastropub                   0.1
    Diner                       0.1
    BBQ Joint                   0.1
    Italian Restaurant          0.1
    Bakery                      0.1
    Japanese Restaurant         0.1
    Korean Restaurant           0.1
    Sushi Restaurant            0.1
    Café                        0.1
    Mediterranean Restaurant    0.1
    Name: Alberta University of the Arts, dtype: float64
    ------- Athabasca University ---------
    Sandwich Place         0.333333
    American Restaurant    0.166667
    Asian Restaurant       0.166667
    Gastropub              0.166667
    Burger Joint           0.166667
    Name: Athabasca University, dtype: float64


From these two outcomes, we could see that the counts and the percentages are equally important. Although the percentage of each venue type near Alberta University of the Arts is lower than that of Athabasca University, in return it means that there are more food options available near Alberta University of the Arts.

But still, we will use the percentages to train our model. Notice that the counts are not continuous and the number of venues returned by the API for each university varies. Hence, using percentages allow us to compare the similarities of food options available near the universities between groups, so we will get better performance from the model and the result is more meaningful.

In addition, recall that we meant to use the name of the category to guess the main cuisine of the venue. However, it turns out that some venues only have vague category names like diner or restaurant. Even though, I think it is still worthwhile to proceed.

### 3.3. Model training

We already got the percentage of each venue category near each university in previous steps. At this stage, we are going to use the K-Means clustering algorithm to label the universities.


```python
from sklearn.cluster import KMeans
```


```python
# We set the number of groups to be 6, and set the random seed to be 0 to avoid the result changes everytime we rerun the program
n_clusters = 6
km = KMeans(n_clusters = n_clusters, init='k-means++', random_state = 0)
```


```python
km.fit(df_recomm_perc)
labels = km.labels_
```


```python
df_univ['Label'] = labels
df_univ.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>City</th>
      <th>Province</th>
      <th>Address</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alberta University of the Arts</td>
      <td>Calgary</td>
      <td>Alberta</td>
      <td>1407 14 Ave NW, Calgary, AB T2N 4R3, Canada</td>
      <td>51.061571</td>
      <td>-114.092098</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Athabasca University</td>
      <td>Athabasca, Calgary, Edmonton</td>
      <td>Alberta</td>
      <td>1 University Dr, Athabasca, AB T9S 3A3, Canada</td>
      <td>54.714955</td>
      <td>-113.308545</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MacEwan University</td>
      <td>Edmonton</td>
      <td>Alberta</td>
      <td>10700 104 Ave NW, Edmonton, AB T5J 4S2, Canada</td>
      <td>53.547054</td>
      <td>-113.506372</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mount Royal University</td>
      <td>Calgary</td>
      <td>Alberta</td>
      <td>30 Mt Royal Cir SW, Calgary, AB T3E 7C9, Canada</td>
      <td>51.011010</td>
      <td>-114.129778</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>University of Alberta</td>
      <td>Edmonton, Camrose, Calgary</td>
      <td>Alberta</td>
      <td>116 St &amp; 85 Ave, Edmonton, AB T6G 2R3, Canada</td>
      <td>53.523219</td>
      <td>-113.526319</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



## PART 4 RESULTS

Now we could display the name of universities under each group.


```python
for table in df_univ.groupby(['Label'], axis=0):
    print("-------- Label {} --------".format(table[0]))
    print(table[1][['Name','Province']])
```

    -------- Label 0 --------
                                                  Name          Province
    13                  University of British Columbia  British Columbia
    14                          University of Victoria  British Columbia
    21                          University of Winnipeg          Manitoba
    23                           St. Thomas University     New Brunswick
    24                     University of New Brunswick     New Brunswick
    25                           Université de Moncton     New Brunswick
    27                               Acadia University       Nova Scotia
    37                             Carleton University           Ontario
    43                  Queen's University at Kingston           Ontario
    44                Royal Military College of Canada           Ontario
    45                              Ryerson University           Ontario
    46                Université de l'Ontario français           Ontario
    47                            University of Guelph           Ontario
    50                           University of Toronto           Ontario
    52                   University of Western Ontario           Ontario
    58                            Concordia University            Quebec
    59                 École de technologie supérieure            Quebec
    60       École nationale d'administration publique            Quebec
    61  Institut national de la recherche scientifique            Quebec
    62                               McGill University            Quebec
    63                          Université de Montréal            Quebec
    68                 Université du Québec à Montréal            Quebec
    69                 Université du Québec à Rimouski            Quebec
    85                       University of Fredericton     New Brunswick
    -------- Label 1 --------
                                                    Name  \
    3                             Mount Royal University   
    7                                Capilano University   
    9                    Kwantlen Polytechnic University   
    15                   University of the Fraser Valley   
    20                            University of Manitoba   
    26               Memorial University of Newfoundland   
    31  Nova Scotia College of Art and Design University   
    38                               Lakehead University   
    39                             Laurentian University   
    42      Ontario College of Art and Design University   
    49                              University of Ottawa   
    51                            University of Waterloo   
    54                        Wilfrid Laurier University   
    55                                   York University   
    66                 Université du Québec en Outaouais   
    70             Université du Québec à Trois-Rivières   
    73                        University of Saskatchewan   
    74           Fairleigh Dickinson University (branch)   
    77                       Niagara University (branch)   
    79                            University Canada West   
    80                          Booth University College   
    
                         Province  
    3                     Alberta  
    7            British Columbia  
    9            British Columbia  
    15           British Columbia  
    20                   Manitoba  
    26  Newfoundland and Labrador  
    31                Nova Scotia  
    38                    Ontario  
    39                    Ontario  
    42                    Ontario  
    49                    Ontario  
    51                    Ontario  
    54                    Ontario  
    55                    Ontario  
    66                     Quebec  
    70                     Quebec  
    73               Saskatchewan  
    74           British Columbia  
    77                    Ontario  
    79           British Columbia  
    80                   Manitoba  
    -------- Label 2 --------
                                 Name  Province
    81  Canadian Mennonite University  Manitoba
    -------- Label 3 --------
                                                 Name              Province
    5                           University of Calgary               Alberta
    18                             Brandon University              Manitoba
    28                         Cape Breton University           Nova Scotia
    32                Saint Francis Xavier University           Nova Scotia
    35                              Algoma University               Ontario
    36                               Brock University               Ontario
    41                           Nipissing University               Ontario
    56             University of Prince Edward Island  Prince Edward Island
    57                            Bishop's University                Quebec
    64                       Université de Sherbrooke                Quebec
    65  Université du Québec en Abitibi-Témiscamingue                Quebec
    72                           University of Regina          Saskatchewan
    82                           Kingswood University         New Brunswick
    83                            Crandall University         New Brunswick
    84                       St. Stephen's University         New Brunswick
    88                    Redeemer University College               Ontario
    -------- Label 4 --------
                                           Name          Province
    1                      Athabasca University           Alberta
    4                     University of Alberta           Alberta
    6                  University of Lethbridge           Alberta
    11                  Simon Fraser University  British Columbia
    16  University of Northern British Columbia  British Columbia
    19          University College of the North          Manitoba
    22                 Mount Allison University     New Brunswick
    34                   Université Sainte-Anne       Nova Scotia
    48                  Ontario Tech University           Ontario
    67        Université du Québec à Chicoutimi            Quebec
    78               Trinity Western University  British Columbia
    89                    The King's University           Alberta
    -------- Label 5 --------
                                             Name          Province
    0              Alberta University of the Arts           Alberta
    2                          MacEwan University           Alberta
    8     Emily Carr University of Art and Design  British Columbia
    10                     Royal Roads University  British Columbia
    12                 Thompson Rivers University  British Columbia
    17                Vancouver Island University  British Columbia
    29                       Dalhousie University       Nova Scotia
    30             Mount Saint Vincent University       Nova Scotia
    33                    Saint Mary's University       Nova Scotia
    40                        McMaster University           Ontario
    53                      University of Windsor           Ontario
    71                           Université Laval            Quebec
    75  New York Institute of Technology (branch)  British Columbia
    76                           Quest University  British Columbia
    86                Atlantic School of Theology       Nova Scotia
    87                         Tyndale University           Ontario


We could also figure out how many universities under each group.


```python
df_univ.groupby(by=['Label'], axis=0).count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>City</th>
      <th>Province</th>
      <th>Address</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
    <tr>
      <th>Label</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>5</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>



While only one universities were assigned to group 2, the other universities have been evenly assigned to different groups.

Now we summarize the clustering result according to the provinces.


```python
df_label_prov = df_univ.groupby(by=['Province', 'Label'], axis=0).count()
df_label_prov = df_label_prov[['Name']]
df_label_prov = df_label_prov.pivot_table(index='Label',columns='Province', fill_value=0)
df_label_prov.columns = df_label_prov.columns.droplevel(level=0)
df_label_prov
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Province</th>
      <th>Alberta</th>
      <th>British Columbia</th>
      <th>Manitoba</th>
      <th>New Brunswick</th>
      <th>Newfoundland and Labrador</th>
      <th>Nova Scotia</th>
      <th>Ontario</th>
      <th>Prince Edward Island</th>
      <th>Quebec</th>
      <th>Saskatchewan</th>
    </tr>
    <tr>
      <th>Label</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Subject to the total number of universities in each province, almost every province has three or more categories. It can be inferred that the smaller area unit such as neighbourhood or block matters more than the province.

Since there are 90 universities in total, the best way to present our findings is visualization. From the map generated using `folium`, we can see six different colours on the map, each representing a different cluster.


```python
# Define the colors to be used for different labels

from matplotlib import cm
from matplotlib import colors

colors_array = cm.autumn(np.linspace(0, 1, n_clusters))
color_ls = [colors.rgb2hex(i) for i in colors_array]
```


```python
# Determine the central point of the map
lat_sta = df_univ['Latitude'].mean()
lng_sta = df_univ['Longitude'].mean()

# Initialize the map
univ_map = folium.Map(location=[lat_sta, lng_sta], zoom_start=4)

# Add circle marks to indicate the location of the universities
count=0
for university, lat, lng, label in zip(df_univ['Name'], df_univ['Latitude'], df_univ['Longitude'], df_univ['Label']):
    folium.CircleMarker(location=[lat,lng],  
                    popup="{}, Cluster {}".format(university.replace("'"," "), label),
                    parse_html=True,
                    radius=12,
                    stroke=True,
                    color=color_ls[label],
                    weight=1,
                    opacity=0.8,
                    fill=True,   
                    ).add_to(univ_map)

univ_map
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZScsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDcuMDk4MjUyMjMzMzMzMzI2LC04Ny43NzExOTI2NjIyMjIyMl0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB6b29tOiA0LAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWF4Qm91bmRzOiBib3VuZHMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBsYXllcnM6IFtdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgd29ybGRDb3B5SnVtcDogZmFsc2UsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjcnM6IEwuQ1JTLkVQU0czODU3CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pOwogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgdGlsZV9sYXllcl9jNzllMTE4MTdiMGM0NWI3OGFlNWRhNDgyNTQ1NTVmOSA9IEwudGlsZUxheWVyKAogICAgICAgICAgICAgICAgJ2h0dHBzOi8ve3N9LnRpbGUub3BlbnN0cmVldG1hcC5vcmcve3p9L3t4fS97eX0ucG5nJywKICAgICAgICAgICAgICAgIHsKICAiYXR0cmlidXRpb24iOiBudWxsLAogICJkZXRlY3RSZXRpbmEiOiBmYWxzZSwKICAibWF4Wm9vbSI6IDE4LAogICJtaW5ab29tIjogMSwKICAibm9XcmFwIjogZmFsc2UsCiAgInN1YmRvbWFpbnMiOiAiYWJjIgp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGVjMDlhYzQ3ZjYzNGMyOGEwMmZlMjQxNDllNjNkZDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS4wNjE1NzA3LC0xMTQuMDkyMDk4M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZmZjAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmZmYwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNThiYzE1MjVjN2U4NDAwNjlmNjNmYzU4MmUwOTliYWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfY2U1YjdiMTU1MTFiNDRjMGI5MWNiZWNlNTcwNWFhZTAgPSAkKCc8ZGl2IGlkPSJodG1sX2NlNWI3YjE1NTExYjQ0YzBiOTFjYmVjZTU3MDVhYWUwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BbGJlcnRhIFVuaXZlcnNpdHkgb2YgdGhlIEFydHMsIENsdXN0ZXIgNTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNThiYzE1MjVjN2U4NDAwNjlmNjNmYzU4MmUwOTliYWMuc2V0Q29udGVudChodG1sX2NlNWI3YjE1NTExYjQ0YzBiOTFjYmVjZTU3MDVhYWUwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzhlYzA5YWM0N2Y2MzRjMjhhMDJmZTI0MTQ5ZTYzZGQxLmJpbmRQb3B1cChwb3B1cF81OGJjMTUyNWM3ZTg0MDA2OWY2M2ZjNTgyZTA5OWJhYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81NmZhZGE4MzhhMWQ0OWRkYmY0NDc0YmY0MGI0ZTQwZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzU0LjcxNDk1NSwtMTEzLjMwODU0NTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmY2MwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmNjMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzBjYTA0NTAxMGZlMzRkZGNhZmMwNjNhOGYxNzQyMjNjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzNlMTY2OWRlYTRiZTQwNTZiMTViZmZkYzNiZTJlMzQ3ID0gJCgnPGRpdiBpZD0iaHRtbF8zZTE2NjlkZWE0YmU0MDU2YjE1YmZmZGMzYmUyZTM0NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QXRoYWJhc2NhIFVuaXZlcnNpdHksIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMGNhMDQ1MDEwZmUzNGRkY2FmYzA2M2E4ZjE3NDIyM2Muc2V0Q29udGVudChodG1sXzNlMTY2OWRlYTRiZTQwNTZiMTViZmZkYzNiZTJlMzQ3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU2ZmFkYTgzOGExZDQ5ZGRiZjQ0NzRiZjQwYjRlNDBlLmJpbmRQb3B1cChwb3B1cF8wY2EwNDUwMTBmZTM0ZGRjYWZjMDYzYThmMTc0MjIzYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yMmMzYjRlYjRmODI0OWU1OGRjNzYyZDNiYTg3NDVjNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUzLjU0NzA1NDQsLTExMy41MDYzNzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmZmYwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmZmMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzliNjYyMTc2NDEyNTQ4ZGE5ZGQyOTVjM2Q5MzEyYmYzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzE2NzQxN2Y3ZWRhZjQwZWZhZGQ5Y2Q0MTMyMjYzM2U5ID0gJCgnPGRpdiBpZD0iaHRtbF8xNjc0MTdmN2VkYWY0MGVmYWRkOWNkNDEzMjI2MzNlOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWFjRXdhbiBVbml2ZXJzaXR5LCBDbHVzdGVyIDU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzliNjYyMTc2NDEyNTQ4ZGE5ZGQyOTVjM2Q5MzEyYmYzLnNldENvbnRlbnQoaHRtbF8xNjc0MTdmN2VkYWY0MGVmYWRkOWNkNDEzMjI2MzNlOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yMmMzYjRlYjRmODI0OWU1OGRjNzYyZDNiYTg3NDVjNi5iaW5kUG9wdXAocG9wdXBfOWI2NjIxNzY0MTI1NDhkYTlkZDI5NWMzZDkzMTJiZjMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjM5ZDIyNzBjZTE5NDZlZThiNzM1MWNhMmZjMWJjZWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS4wMTEwMSwtMTE0LjEyOTc3OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYzMzAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMzMwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjVmZTQyZjFmYTI5NGY5MWEyZGViY2Q1YzE0NjgwMTYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTA3NzIyYzM5NzhkNDdmZWFiM2ViMDgyYTJkNDFhY2QgPSAkKCc8ZGl2IGlkPSJodG1sXzUwNzcyMmMzOTc4ZDQ3ZmVhYjNlYjA4MmEyZDQxYWNkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Nb3VudCBSb3lhbCBVbml2ZXJzaXR5LCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I1ZmU0MmYxZmEyOTRmOTFhMmRlYmNkNWMxNDY4MDE2LnNldENvbnRlbnQoaHRtbF81MDc3MjJjMzk3OGQ0N2ZlYWIzZWIwODJhMmQ0MWFjZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mMzlkMjI3MGNlMTk0NmVlOGI3MzUxY2EyZmMxYmNlYi5iaW5kUG9wdXAocG9wdXBfYjVmZTQyZjFmYTI5NGY5MWEyZGViY2Q1YzE0NjgwMTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmVjMTRjMjVmZjdiNGMyODg3OWU5ZTk1Zjg3ZWExNmYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1My41MjMyMTg5LC0xMTMuNTI2MzE4Nl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZjYzAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmY2MwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOWUyMTg0OGRiOGY5NDNjNGJmY2M4MzAzYzM3ZTI0MmUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTlhOTc1YzBiMzBjNGYzM2JhMWQzMjRhODFjMDM4MWQgPSAkKCc8ZGl2IGlkPSJodG1sX2U5YTk3NWMwYjMwYzRmMzNiYTFkMzI0YTgxYzAzODFkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIEFsYmVydGEsIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWUyMTg0OGRiOGY5NDNjNGJmY2M4MzAzYzM3ZTI0MmUuc2V0Q29udGVudChodG1sX2U5YTk3NWMwYjMwYzRmMzNiYTFkMzI0YTgxYzAzODFkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZlYzE0YzI1ZmY3YjRjMjg4NzllOWU5NWY4N2VhMTZmLmJpbmRQb3B1cChwb3B1cF85ZTIxODQ4ZGI4Zjk0M2M0YmZjYzgzMDNjMzdlMjQyZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jYzE5YzA2ODZhNzc0OTAzYWFhNjE5NWUxNGI5MmM3MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjE1OTQ3MywtMTE0LjIxNDgyN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmY5OTAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmOTkwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjY0YTgzNTlhNjAyNDE1OGI2NDExOWI0YmM1MWI1MGIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDc2Yzk1YmY1YWY0NDAxN2I1ZmJhNjk4MThhOGM5ZjIgPSAkKCc8ZGl2IGlkPSJodG1sXzQ3NmM5NWJmNWFmNDQwMTdiNWZiYTY5ODE4YThjOWYyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIENhbGdhcnksIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjY0YTgzNTlhNjAyNDE1OGI2NDExOWI0YmM1MWI1MGIuc2V0Q29udGVudChodG1sXzQ3NmM5NWJmNWFmNDQwMTdiNWZiYTY5ODE4YThjOWYyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2NjMTljMDY4NmE3NzQ5MDNhYWE2MTk1ZTE0YjkyYzcwLmJpbmRQb3B1cChwb3B1cF9mNjRhODM1OWE2MDI0MTU4YjY0MTE5YjRiYzUxYjUwYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kNWEwZDRlYTg2ODQ0N2FiYWU5MjUwZjZjNzRjNDYwYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ5LjY3ODYxNTYsLTExMi44NjAxMTc3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmNjMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZjYzAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kMTRmZGU2YTRiNDY0YWQ2Yjg0ZTNjZGMxZDg4ODQ3MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wNTRiMDBjZTI5OGU0YjI1YThhYmExZmQwMjI1NTQyZiA9ICQoJzxkaXYgaWQ9Imh0bWxfMDU0YjAwY2UyOThlNGIyNWE4YWJhMWZkMDIyNTU0MmYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdHkgb2YgTGV0aGJyaWRnZSwgQ2x1c3RlciA0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kMTRmZGU2YTRiNDY0YWQ2Yjg0ZTNjZGMxZDg4ODQ3My5zZXRDb250ZW50KGh0bWxfMDU0YjAwY2UyOThlNGIyNWE4YWJhMWZkMDIyNTU0MmYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDVhMGQ0ZWE4Njg0NDdhYmFlOTI1MGY2Yzc0YzQ2MGEuYmluZFBvcHVwKHBvcHVwX2QxNGZkZTZhNGI0NjRhZDZiODRlM2NkYzFkODg4NDczKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzdjMDgzOTBkNTQ1ODQ2ZjhhYzE0N2Q4YjM4NGQwNjdiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDkuMzE3NzU0LC0xMjMuMDE5MDg1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjMzMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYzMzAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zNGZjMGI3MjAyNzY0OTZjOGFhNTljNzcyZjA3NWMzYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84ZDYzNmI0M2M5YTU0OGQ4Yjg5NmYzMjA5MWJkZTI4NiA9ICQoJzxkaXYgaWQ9Imh0bWxfOGQ2MzZiNDNjOWE1NDhkOGI4OTZmMzIwOTFiZGUyODYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNhcGlsYW5vIFVuaXZlcnNpdHksIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzRmYzBiNzIwMjc2NDk2YzhhYTU5Yzc3MmYwNzVjM2Iuc2V0Q29udGVudChodG1sXzhkNjM2YjQzYzlhNTQ4ZDhiODk2ZjMyMDkxYmRlMjg2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdjMDgzOTBkNTQ1ODQ2ZjhhYzE0N2Q4YjM4NGQwNjdiLmJpbmRQb3B1cChwb3B1cF8zNGZjMGI3MjAyNzY0OTZjOGFhNTljNzcyZjA3NWMzYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81ZjM2ZTIzNTVjOTg0Yzk4OWEzYTU2NTdjMzRmMmY5NCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ5LjI2NzU3OSwtMTIzLjA5MjUzMjJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmZmYwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmZmMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2EwMGFjNjVhNzI0NzQwYmM4MmI2NTBkNTZjYWI5MTZkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdkZWY5NWExYzk2MTQ1ZDk4YWQwMTEwMTkwZjcyZDQ5ID0gJCgnPGRpdiBpZD0iaHRtbF83ZGVmOTVhMWM5NjE0NWQ5OGFkMDExMDE5MGY3MmQ0OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RW1pbHkgQ2FyciBVbml2ZXJzaXR5IG9mIEFydCBhbmQgRGVzaWduLCBDbHVzdGVyIDU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2EwMGFjNjVhNzI0NzQwYmM4MmI2NTBkNTZjYWI5MTZkLnNldENvbnRlbnQoaHRtbF83ZGVmOTVhMWM5NjE0NWQ5OGFkMDExMDE5MGY3MmQ0OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81ZjM2ZTIzNTVjOTg0Yzk4OWEzYTU2NTdjMzRmMmY5NC5iaW5kUG9wdXAocG9wdXBfYTAwYWM2NWE3MjQ3NDBiYzgyYjY1MGQ1NmNhYjkxNmQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTk0Nzk1MmU0ZWE1NGRiNWFmZDcyMThmODExZWQwYTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OS4xMzI2NTUzLC0xMjIuODcxNDgyMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYzMzAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMzMwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTk4ZWYzMTUxNWRkNGI2M2JkOGViNjg0Mzg3NGVkYjEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjUwN2E0Y2Q4MzYwNDlkZWFiOWU2YzljNjY4M2IyYmMgPSAkKCc8ZGl2IGlkPSJodG1sX2Y1MDdhNGNkODM2MDQ5ZGVhYjllNmM5YzY2ODNiMmJjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ld2FudGxlbiBQb2x5dGVjaG5pYyBVbml2ZXJzaXR5LCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U5OGVmMzE1MTVkZDRiNjNiZDhlYjY4NDM4NzRlZGIxLnNldENvbnRlbnQoaHRtbF9mNTA3YTRjZDgzNjA0OWRlYWI5ZTZjOWM2NjgzYjJiYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81OTQ3OTUyZTRlYTU0ZGI1YWZkNzIxOGY4MTFlZDBhMS5iaW5kUG9wdXAocG9wdXBfZTk4ZWYzMTUxNWRkNGI2M2JkOGViNjg0Mzg3NGVkYjEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGFlOWY3ZDQ1MGM1NDYzNmE0YjMxNGU5NWI1N2RlNWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OC40MzQyMDQ3LC0xMjMuNDczOTY4OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZmZjAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmZmYwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTgwOTQ3NDkxNWJkNDFmNGI5MTY5Y2Q0ODI4ODFhNzggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmRhY2U5MDU2YmU4NDU5ODkwZTU1MmJiZDdkN2MwNjYgPSAkKCc8ZGl2IGlkPSJodG1sXzJkYWNlOTA1NmJlODQ1OTg5MGU1NTJiYmQ3ZDdjMDY2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb3lhbCBSb2FkcyBVbml2ZXJzaXR5LCBDbHVzdGVyIDU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE4MDk0NzQ5MTViZDQxZjRiOTE2OWNkNDgyODgxYTc4LnNldENvbnRlbnQoaHRtbF8yZGFjZTkwNTZiZTg0NTk4OTBlNTUyYmJkN2Q3YzA2Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wYWU5ZjdkNDUwYzU0NjM2YTRiMzE0ZTk1YjU3ZGU1YS5iaW5kUG9wdXAocG9wdXBfMTgwOTQ3NDkxNWJkNDFmNGI5MTY5Y2Q0ODI4ODFhNzgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzE5YzU1YTAwNGVkNDE3MWI3MzdhNTBhMzVkYjdlZmUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OS4yNzgwOTM3LC0xMjIuOTE5ODgzM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZjYzAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmY2MwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODdjMGUzZWU1ZWEwNDYzNWEyNTY3OGQ5NDc3MmZjOTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjZiNzVjOGVlMjc3NDJhMTg1OWU4N2EwNzJjYmM5ZDIgPSAkKCc8ZGl2IGlkPSJodG1sXzI2Yjc1YzhlZTI3NzQyYTE4NTllODdhMDcyY2JjOWQyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TaW1vbiBGcmFzZXIgVW5pdmVyc2l0eSwgQ2x1c3RlciA0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84N2MwZTNlZTVlYTA0NjM1YTI1Njc4ZDk0NzcyZmM5OC5zZXRDb250ZW50KGh0bWxfMjZiNzVjOGVlMjc3NDJhMTg1OWU4N2EwNzJjYmM5ZDIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzE5YzU1YTAwNGVkNDE3MWI3MzdhNTBhMzVkYjdlZmUuYmluZFBvcHVwKHBvcHVwXzg3YzBlM2VlNWVhMDQ2MzVhMjU2NzhkOTQ3NzJmYzk4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzViMzM2NDU1YTM1YTRjODc4NWQyYmVkMTZiNzc3NmYwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTAuNjcxMjIwMiwtMTIwLjM2NjYyNjRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmZmYwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmZmMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2MzYjk5ZjlkNGE1OTQ0MGFhYzRhZTBlNDJjMGVmZWUwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q1NDMwMTg4ODJhNDQyNjk4YzAyNTYzODhkZDk2N2IyID0gJCgnPGRpdiBpZD0iaHRtbF9kNTQzMDE4ODgyYTQ0MjY5OGMwMjU2Mzg4ZGQ5NjdiMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhvbXBzb24gUml2ZXJzIFVuaXZlcnNpdHksIENsdXN0ZXIgNTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzNiOTlmOWQ0YTU5NDQwYWFjNGFlMGU0MmMwZWZlZTAuc2V0Q29udGVudChodG1sX2Q1NDMwMTg4ODJhNDQyNjk4YzAyNTYzODhkZDk2N2IyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzViMzM2NDU1YTM1YTRjODc4NWQyYmVkMTZiNzc3NmYwLmJpbmRQb3B1cChwb3B1cF9jM2I5OWY5ZDRhNTk0NDBhYWM0YWUwZTQyYzBlZmVlMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80N2IyYmFjZDI1MDI0MjVlOGY0MmExYWRhZjcyZDBhMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ5LjI2MDYwNTIwMDAwMDAxLC0xMjMuMjQ1OTkzOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOWI2NGI4YTc3ZDE3NDgwMTg3YWM4N2JkNDE5OWEwNjcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjc5MGJhZDg1NDBhNGUxMmI3NTZhNWNjNTU4OTEwOGUgPSAkKCc8ZGl2IGlkPSJodG1sXzI3OTBiYWQ4NTQwYTRlMTJiNzU2YTVjYzU1ODkxMDhlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIEJyaXRpc2ggQ29sdW1iaWEsIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWI2NGI4YTc3ZDE3NDgwMTg3YWM4N2JkNDE5OWEwNjcuc2V0Q29udGVudChodG1sXzI3OTBiYWQ4NTQwYTRlMTJiNzU2YTVjYzU1ODkxMDhlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQ3YjJiYWNkMjUwMjQyNWU4ZjQyYTFhZGFmNzJkMGExLmJpbmRQb3B1cChwb3B1cF85YjY0YjhhNzdkMTc0ODAxODdhYzg3YmQ0MTk5YTA2Nyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xZjg4OGNhZTJlYjM0YmUyYjExNWUwY2JhMGZiMDczNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ4LjQ2MzQwNjcsLTEyMy4zMTE2OTM1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83ZDZlODBjOThmMmM0YWM5YTU4ODQ2MWMzN2ViNTA3MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80Y2U3ZTNhOWMzZmI0MzM2OWU0ZTdjMWIzYTZmNmNlZCA9ICQoJzxkaXYgaWQ9Imh0bWxfNGNlN2UzYTljM2ZiNDMzNjllNGU3YzFiM2E2ZjZjZWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdHkgb2YgVmljdG9yaWEsIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfN2Q2ZTgwYzk4ZjJjNGFjOWE1ODg0NjFjMzdlYjUwNzIuc2V0Q29udGVudChodG1sXzRjZTdlM2E5YzNmYjQzMzY5ZTRlN2MxYjNhNmY2Y2VkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFmODg4Y2FlMmViMzRiZTJiMTE1ZTBjYmEwZmIwNzM3LmJpbmRQb3B1cChwb3B1cF83ZDZlODBjOThmMmM0YWM5YTU4ODQ2MWMzN2ViNTA3Mik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85NjBjMWJlNjJiYWQ0NmRmOTlmM2ZiZDUwZjEyM2ZjMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ5LjAyOTA1MSwtMTIyLjI4NTQzNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYzMzAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMzMwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWFhMzdhZjhjYTkyNDEyNmEwMzE0NzdjNjc3MmFlNWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTcwOGRjMWZiZGQ4NDQ0Mjg2NWI1YjE0MWI0OTRkY2MgPSAkKCc8ZGl2IGlkPSJodG1sXzU3MDhkYzFmYmRkODQ0NDI4NjViNWIxNDFiNDk0ZGNjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIHRoZSBGcmFzZXIgVmFsbGV5LCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VhYTM3YWY4Y2E5MjQxMjZhMDMxNDc3YzY3NzJhZTViLnNldENvbnRlbnQoaHRtbF81NzA4ZGMxZmJkZDg0NDQyODY1YjViMTQxYjQ5NGRjYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85NjBjMWJlNjJiYWQ0NmRmOTlmM2ZiZDUwZjEyM2ZjMS5iaW5kUG9wdXAocG9wdXBfZWFhMzdhZjhjYTkyNDEyNmEwMzE0NzdjNjc3MmFlNWIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWQyNGJkNzVkOTBmNDljYThiNWIxYmU3ODQ2NjgyZDMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1My44OTIyMDM0LC0xMjIuODEzMzYwN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZjYzAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmY2MwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDIwYTAyZGI4YWIyNGJiMWI4MzU2YmVkMWIwMzgzY2IgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTI1ZjNhMzVkNThiNDNiYWE2NGM2MmQ1MGMwYjcwMDIgPSAkKCc8ZGl2IGlkPSJodG1sXzkyNWYzYTM1ZDU4YjQzYmFhNjRjNjJkNTBjMGI3MDAyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIE5vcnRoZXJuIEJyaXRpc2ggQ29sdW1iaWEsIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDIwYTAyZGI4YWIyNGJiMWI4MzU2YmVkMWIwMzgzY2Iuc2V0Q29udGVudChodG1sXzkyNWYzYTM1ZDU4YjQzYmFhNjRjNjJkNTBjMGI3MDAyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFkMjRiZDc1ZDkwZjQ5Y2E4YjViMWJlNzg0NjY4MmQzLmJpbmRQb3B1cChwb3B1cF8wMjBhMDJkYjhhYjI0YmIxYjgzNTZiZWQxYjAzODNjYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83OWFiMjFhMjQxNzQ0NGM4OGFkMDQyOWQ2MDU4NjBlYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ5LjE1NzMwMjQsLTEyMy45NjY0MzIyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmZmMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZmZjAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kN2UwYWYxNjE4YjM0NGVkODRkNzZmM2U3NmE3MjUyMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80NmJlZTQ4MGFlNzg0YzNiYjM1YjJmMzJlYmY3MzU1MiA9ICQoJzxkaXYgaWQ9Imh0bWxfNDZiZWU0ODBhZTc4NGMzYmIzNWIyZjMyZWJmNzM1NTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlZhbmNvdXZlciBJc2xhbmQgVW5pdmVyc2l0eSwgQ2x1c3RlciA1PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kN2UwYWYxNjE4YjM0NGVkODRkNzZmM2U3NmE3MjUyMi5zZXRDb250ZW50KGh0bWxfNDZiZWU0ODBhZTc4NGMzYmIzNWIyZjMyZWJmNzM1NTIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzlhYjIxYTI0MTc0NDRjODhhZDA0MjlkNjA1ODYwZWMuYmluZFBvcHVwKHBvcHVwX2Q3ZTBhZjE2MThiMzQ0ZWQ4NGQ3NmYzZTc2YTcyNTIyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE0NGYwZTZmM2Q1ZjQwZWRiZTRiNWZlZjAyODc2N2I1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDkuODQ1MzUyLC05OS45NjIxNTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmOTkwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjk5MDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzlmMzI3MWFhOGFlNTQ2ODA4NjcwZDU4NzZkZTVmYzk2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzU3YzgwMDZjM2UxNTRjZjM4NmZiMzJhOGZkOTRhZjkyID0gJCgnPGRpdiBpZD0iaHRtbF81N2M4MDA2YzNlMTU0Y2YzODZmYjMyYThmZDk0YWY5MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QnJhbmRvbiBVbml2ZXJzaXR5LCBDbHVzdGVyIDM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzlmMzI3MWFhOGFlNTQ2ODA4NjcwZDU4NzZkZTVmYzk2LnNldENvbnRlbnQoaHRtbF81N2M4MDA2YzNlMTU0Y2YzODZmYjMyYThmZDk0YWY5Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xNDRmMGU2ZjNkNWY0MGVkYmU0YjVmZWYwMjg3NjdiNS5iaW5kUG9wdXAocG9wdXBfOWYzMjcxYWE4YWU1NDY4MDg2NzBkNTg3NmRlNWZjOTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTdlZmU4N2Y0OGMwNDFiNzhlYjEzMjljNzlkOTFjNmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1My44MTkzNDk0LC0xMDEuMjM2NDg4OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZjYzAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmY2MwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzhiNzg4NjhjZmVhNDg3M2FlYTVjNmJkMzc1ZGIyYjggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWVlZjdmYTk5MWYwNGQ1YWJhMjZkZDE2Y2E2NjM2ZWYgPSAkKCc8ZGl2IGlkPSJodG1sXzVlZWY3ZmE5OTFmMDRkNWFiYTI2ZGQxNmNhNjYzNmVmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IENvbGxlZ2Ugb2YgdGhlIE5vcnRoLCBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM4Yjc4ODY4Y2ZlYTQ4NzNhZWE1YzZiZDM3NWRiMmI4LnNldENvbnRlbnQoaHRtbF81ZWVmN2ZhOTkxZjA0ZDVhYmEyNmRkMTZjYTY2MzZlZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81N2VmZTg3ZjQ4YzA0MWI3OGViMTMyOWM3OWQ5MWM2Yy5iaW5kUG9wdXAocG9wdXBfMzhiNzg4NjhjZmVhNDg3M2FlYTVjNmJkMzc1ZGIyYjgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWZiMGZlOTUyNzMyNDZiYzgzY2QwMDA5ODU3YjJhMjkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OS44MDc1MDA4LC05Ny4xMzY2MjU5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjMzMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYzMzAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iNTQzZDM5NjcwYWI0YjM5YjVmYmM5OGQzNzllODBmYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hMzAxNDUwMDBkZWY0MTFmOWE5MDlmNDQ4YTZjN2UzNCA9ICQoJzxkaXYgaWQ9Imh0bWxfYTMwMTQ1MDAwZGVmNDExZjlhOTA5ZjQ0OGE2YzdlMzQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdHkgb2YgTWFuaXRvYmEsIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjU0M2QzOTY3MGFiNGIzOWI1ZmJjOThkMzc5ZTgwZmEuc2V0Q29udGVudChodG1sX2EzMDE0NTAwMGRlZjQxMWY5YTkwOWY0NDhhNmM3ZTM0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FmYjBmZTk1MjczMjQ2YmM4M2NkMDAwOTg1N2IyYTI5LmJpbmRQb3B1cChwb3B1cF9iNTQzZDM5NjcwYWI0YjM5YjVmYmM5OGQzNzllODBmYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82OGI4OTY3MDg1NWM0ZGI3YjZlZGFhZGI3MjlmMjdmMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ5Ljg5MTI1NDMwMDAwMDAxLC05Ny4xNTM0ODddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNhZWM5YmI4OTZhMDQxNmE5NmQ4N2RmYmVmMTU5ZThhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzU4ZmMwNjQyN2JkODRjZGU4ODBjMWFiZDYxNWU3MGJjID0gJCgnPGRpdiBpZD0iaHRtbF81OGZjMDY0MjdiZDg0Y2RlODgwYzFhYmQ2MTVlNzBiYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VW5pdmVyc2l0eSBvZiBXaW5uaXBlZywgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zYWVjOWJiODk2YTA0MTZhOTZkODdkZmJlZjE1OWU4YS5zZXRDb250ZW50KGh0bWxfNThmYzA2NDI3YmQ4NGNkZTg4MGMxYWJkNjE1ZTcwYmMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjhiODk2NzA4NTVjNGRiN2I2ZWRhYWRiNzI5ZjI3ZjEuYmluZFBvcHVwKHBvcHVwXzNhZWM5YmI4OTZhMDQxNmE5NmQ4N2RmYmVmMTU5ZThhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M3OTQ2NjNhMjZmZTRjMDQ4NWVkODhkZWEyZTkwZDVlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDUuODk4MzE4NCwtNjQuMzczMTAwNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZjYzAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmY2MwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTVlNzkyMTAwMjdiNDRmMjhmZWE0NmY3MzFlNzA4ODQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODcxM2JhNjkwMGQwNGYwN2IzYWE2NWQwNDQ2N2MzOWIgPSAkKCc8ZGl2IGlkPSJodG1sXzg3MTNiYTY5MDBkMDRmMDdiM2FhNjVkMDQ0NjdjMzliIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Nb3VudCBBbGxpc29uIFVuaXZlcnNpdHksIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTVlNzkyMTAwMjdiNDRmMjhmZWE0NmY3MzFlNzA4ODQuc2V0Q29udGVudChodG1sXzg3MTNiYTY5MDBkMDRmMDdiM2FhNjVkMDQ0NjdjMzliKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M3OTQ2NjNhMjZmZTRjMDQ4NWVkODhkZWEyZTkwZDVlLmJpbmRQb3B1cChwb3B1cF81NWU3OTIxMDAyN2I0NGYyOGZlYTQ2ZjczMWU3MDg4NCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81YmYwNTE0Mzg5NzA0MDJmOWVjZDg1Y2RmODAwNjczNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ1Ljk0NDAxOTk5OTk5OTk5LC02Ni42NDYyMjEzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kMmE4OWVlZTdjZWM0N2JjOTQ5NzUwZWY2ZTMyMzdkMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80YThkZjI3YzZkMjU0MjM5OTQ5NjY4MzZjNGQwY2Y1MSA9ICQoJzxkaXYgaWQ9Imh0bWxfNGE4ZGYyN2M2ZDI1NDIzOTk0OTY2ODM2YzRkMGNmNTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBUaG9tYXMgVW5pdmVyc2l0eSwgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kMmE4OWVlZTdjZWM0N2JjOTQ5NzUwZWY2ZTMyMzdkMy5zZXRDb250ZW50KGh0bWxfNGE4ZGYyN2M2ZDI1NDIzOTk0OTY2ODM2YzRkMGNmNTEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNWJmMDUxNDM4OTcwNDAyZjllY2Q4NWNkZjgwMDY3MzQuYmluZFBvcHVwKHBvcHVwX2QyYTg5ZWVlN2NlYzQ3YmM5NDk3NTBlZjZlMzIzN2QzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE2YTYyYTFhMmE0MTQ5ZmI5MGRhYzkxYjYxYTM1OWJhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDUuOTQ1NTcwNCwtNjYuNjQwODI2NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzQwMDcyYTdlYmFkNGRiNmIyZjk0NDZhM2YzYmIxNjEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZWZlMzFjMjNhMTY0NDRjYmE1NzFkYmY1Yjg3MWIzMzIgPSAkKCc8ZGl2IGlkPSJodG1sX2VmZTMxYzIzYTE2NDQ0Y2JhNTcxZGJmNWI4NzFiMzMyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIE5ldyBCcnVuc3dpY2ssIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzQwMDcyYTdlYmFkNGRiNmIyZjk0NDZhM2YzYmIxNjEuc2V0Q29udGVudChodG1sX2VmZTMxYzIzYTE2NDQ0Y2JhNTcxZGJmNWI4NzFiMzMyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE2YTYyYTFhMmE0MTQ5ZmI5MGRhYzkxYjYxYTM1OWJhLmJpbmRQb3B1cChwb3B1cF83NDAwNzJhN2ViYWQ0ZGI2YjJmOTQ0NmEzZjNiYjE2MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lNTJjNjg5OTNkMGU0MWMwOGU5MmNjZDlkYWRiNzFiOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ2LjEwNTA5MDQsLTY0Ljc4MTc2MTg5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81YjZkNzVhNTQ2Njg0YzQ0YmMxZTM5N2RiZjFhYmViNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wNDU2M2ExNmJjMjE0NzBjODQwNDM4MThmOGViYzU5ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfMDQ1NjNhMTZiYzIxNDcwYzg0MDQzODE4ZjhlYmM1OWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdMOpIGRlIE1vbmN0b24sIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNWI2ZDc1YTU0NjY4NGM0NGJjMWUzOTdkYmYxYWJlYjYuc2V0Q29udGVudChodG1sXzA0NTYzYTE2YmMyMTQ3MGM4NDA0MzgxOGY4ZWJjNTlkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U1MmM2ODk5M2QwZTQxYzA4ZTkyY2NkOWRhZGI3MWI4LmJpbmRQb3B1cChwb3B1cF81YjZkNzVhNTQ2Njg0YzQ0YmMxZTM5N2RiZjFhYmViNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jNzhmZGE1MDUxNDE0NTk4YTQ0OWM0ZTIzNzlhNTNjMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ3LjU3Mzc5NzUsLTUyLjczMjkwNTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMzMwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjMzMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U1NTEwM2MyMjM5MTQ1YjU5NmQ0ZDViYTdhNzFkOGUzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2JiOTIzMzMzZGJmNTQ1ZjhhMDFmN2M4ZjA4NjA5MjUzID0gJCgnPGRpdiBpZD0iaHRtbF9iYjkyMzMzM2RiZjU0NWY4YTAxZjdjOGYwODYwOTI1MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWVtb3JpYWwgVW5pdmVyc2l0eSBvZiBOZXdmb3VuZGxhbmQsIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTU1MTAzYzIyMzkxNDViNTk2ZDRkNWJhN2E3MWQ4ZTMuc2V0Q29udGVudChodG1sX2JiOTIzMzMzZGJmNTQ1ZjhhMDFmN2M4ZjA4NjA5MjUzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M3OGZkYTUwNTE0MTQ1OThhNDQ5YzRlMjM3OWE1M2MxLmJpbmRQb3B1cChwb3B1cF9lNTUxMDNjMjIzOTE0NWI1OTZkNGQ1YmE3YTcxZDhlMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80M2E4ZjY3YWM4MjY0OWFkYTVhOTZiOTgyY2ZiZGY0OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ1LjA5MDUxNjA5OTk5OTk5LC02NC4zNjQ1MjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q1NmUyYjEyZjQwMjRhMTlhNjkwNDBjMmRjZDYyOWU5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZkNTQ0YjYzYWNlMjRkMmI4MThjOTAwZDlhZmZhMGQzID0gJCgnPGRpdiBpZD0iaHRtbF82ZDU0NGI2M2FjZTI0ZDJiODE4YzkwMGQ5YWZmYTBkMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QWNhZGlhIFVuaXZlcnNpdHksIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDU2ZTJiMTJmNDAyNGExOWE2OTA0MGMyZGNkNjI5ZTkuc2V0Q29udGVudChodG1sXzZkNTQ0YjYzYWNlMjRkMmI4MThjOTAwZDlhZmZhMGQzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQzYThmNjdhYzgyNjQ5YWRhNWE5NmI5ODJjZmJkZjQ4LmJpbmRQb3B1cChwb3B1cF9kNTZlMmIxMmY0MDI0YTE5YTY5MDQwYzJkY2Q2MjllOSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hMTViYTE2MDVmMTU0NWIyOGFjNDZjODQ1M2UxYzNiZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ2LjE3MDYzMDEsLTYwLjA5MzQ5OTc5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjk5MDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmY5OTAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wN2QzY2VhMGY3N2E0ZThkYTc4YTA5ZWQxZTI3MDhjZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80ZTE5ZTUyZGJhOGQ0NGVmOTliODBiNWQyZDU2ZDI4YSA9ICQoJzxkaXYgaWQ9Imh0bWxfNGUxOWU1MmRiYThkNDRlZjk5YjgwYjVkMmQ1NmQyOGEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNhcGUgQnJldG9uIFVuaXZlcnNpdHksIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDdkM2NlYTBmNzdhNGU4ZGE3OGEwOWVkMWUyNzA4Y2Yuc2V0Q29udGVudChodG1sXzRlMTllNTJkYmE4ZDQ0ZWY5OWI4MGI1ZDJkNTZkMjhhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ExNWJhMTYwNWYxNTQ1YjI4YWM0NmM4NDUzZTFjM2JkLmJpbmRQb3B1cChwb3B1cF8wN2QzY2VhMGY3N2E0ZThkYTc4YTA5ZWQxZTI3MDhjZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wZGM3MjZmOWJkOWU0ZTQ3YTM4ODQ5MjEwMTQ3ODhlNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ0LjYzNjU4MTE5OTk5OTk5LC02My41OTE2NTU0OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZmZjAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmZmYwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTY1ZTU1Zjg4NDRjNDdjMzk3ODg5NmI0MTMyNGM0ZDIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjQwZTg3Nzc4MTQzNDA1MjgzYTk5ZDI1NTNhYzhjM2QgPSAkKCc8ZGl2IGlkPSJodG1sX2Y0MGU4Nzc3ODE0MzQwNTI4M2E5OWQyNTUzYWM4YzNkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EYWxob3VzaWUgVW5pdmVyc2l0eSwgQ2x1c3RlciA1PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xNjVlNTVmODg0NGM0N2MzOTc4ODk2YjQxMzI0YzRkMi5zZXRDb250ZW50KGh0bWxfZjQwZTg3Nzc4MTQzNDA1MjgzYTk5ZDI1NTNhYzhjM2QpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGRjNzI2ZjliZDllNGU0N2EzODg0OTIxMDE0Nzg4ZTYuYmluZFBvcHVwKHBvcHVwXzE2NWU1NWY4ODQ0YzQ3YzM5Nzg4OTZiNDEzMjRjNGQyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzgyMjM5N2ZkMDkyYTRjZmVhMDgxM2Y5MzU3YTcxOGZmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDQuNjcxMjAxMiwtNjMuNjQ1NjA0M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZmZjAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmZmYwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTgzZGE3MWVlNzVkNDllYjlkN2MyNTY4YWM4MzA5ZDIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDMwZWQ3MjY5YjBkNGI2YWJiYTM2OGI4MjliOTIxMjkgPSAkKCc8ZGl2IGlkPSJodG1sXzQzMGVkNzI2OWIwZDRiNmFiYmEzNjhiODI5YjkyMTI5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Nb3VudCBTYWludCBWaW5jZW50IFVuaXZlcnNpdHksIENsdXN0ZXIgNTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTgzZGE3MWVlNzVkNDllYjlkN2MyNTY4YWM4MzA5ZDIuc2V0Q29udGVudChodG1sXzQzMGVkNzI2OWIwZDRiNmFiYmEzNjhiODI5YjkyMTI5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzgyMjM5N2ZkMDkyYTRjZmVhMDgxM2Y5MzU3YTcxOGZmLmJpbmRQb3B1cChwb3B1cF9lODNkYTcxZWU3NWQ0OWViOWQ3YzI1NjhhYzgzMDlkMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yYzhhMTkzYzI3ZDE0MDNkYWM2MDMzOTMzMWVlZGIwMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ0LjY0OTU0NzUsLTYzLjU3NDE1ODM5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjMzMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYzMzAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wNTQ1MmY2OGI3MzE0NzlkYjg4N2IzY2NjZGFmNmI5NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82MGZkMzY5OTI5MDA0MTFmOTY5NzVkMzc0ZWU0YzBmNiA9ICQoJzxkaXYgaWQ9Imh0bWxfNjBmZDM2OTkyOTAwNDExZjk2OTc1ZDM3NGVlNGMwZjYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5vdmEgU2NvdGlhIENvbGxlZ2Ugb2YgQXJ0IGFuZCBEZXNpZ24gVW5pdmVyc2l0eSwgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wNTQ1MmY2OGI3MzE0NzlkYjg4N2IzY2NjZGFmNmI5Ny5zZXRDb250ZW50KGh0bWxfNjBmZDM2OTkyOTAwNDExZjk2OTc1ZDM3NGVlNGMwZjYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmM4YTE5M2MyN2QxNDAzZGFjNjAzMzkzMzFlZWRiMDMuYmluZFBvcHVwKHBvcHVwXzA1NDUyZjY4YjczMTQ3OWRiODg3YjNjY2NkYWY2Yjk3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E4NWIzMzIwNDJmNzRmY2U4MjdkNTFkY2FkZGNjMGIzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDUuNjE3NzM2Njk5OTk5OTksLTYxLjk5NTM5MTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmOTkwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjk5MDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzlkMzFjYmZmM2IyOTQzODM5NWE1NzVjZjAwZjg1MDVhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhkOGZmMDgyNjdmMjQ1ZTk4NjMxY2ZjYWM5NzQ4YzZiID0gJCgnPGRpdiBpZD0iaHRtbF84ZDhmZjA4MjY3ZjI0NWU5ODYzMWNmY2FjOTc0OGM2YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U2FpbnQgRnJhbmNpcyBYYXZpZXIgVW5pdmVyc2l0eSwgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85ZDMxY2JmZjNiMjk0MzgzOTVhNTc1Y2YwMGY4NTA1YS5zZXRDb250ZW50KGh0bWxfOGQ4ZmYwODI2N2YyNDVlOTg2MzFjZmNhYzk3NDhjNmIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTg1YjMzMjA0MmY3NGZjZTgyN2Q1MWRjYWRkY2MwYjMuYmluZFBvcHVwKHBvcHVwXzlkMzFjYmZmM2IyOTQzODM5NWE1NzVjZjAwZjg1MDVhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ViYjM0ZGFlYTFiMzQ2NzNiYmU3ZGI2MGViY2Y1MDNhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDQuNjMxMzMwMSwtNjMuNTgxNDU3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmZmMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZmZjAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zNjI1NDVmNDAwZGE0ZWQ0YjI1OTNmZGFhY2VkZDY0OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hYjhlOWIyYjZkMWI0Yzc3YjhiYWE2MDE2MTBmODNlMSA9ICQoJzxkaXYgaWQ9Imh0bWxfYWI4ZTliMmI2ZDFiNGM3N2I4YmFhNjAxNjEwZjgzZTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNhaW50IE1hcnkgcyBVbml2ZXJzaXR5LCBDbHVzdGVyIDU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM2MjU0NWY0MDBkYTRlZDRiMjU5M2ZkYWFjZWRkNjQ4LnNldENvbnRlbnQoaHRtbF9hYjhlOWIyYjZkMWI0Yzc3YjhiYWE2MDE2MTBmODNlMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lYmIzNGRhZWExYjM0NjczYmJlN2RiNjBlYmNmNTAzYS5iaW5kUG9wdXAocG9wdXBfMzYyNTQ1ZjQwMGRhNGVkNGIyNTkzZmRhYWNlZGQ2NDgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNWZkNTZhNmI1MDU2NDI5Yjg3NjliOGUxYmY5YWMwNjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0NC4zMzI2Nzk4LC02Ni4xMTY4NzUzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmNjMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZjYzAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zZDVjNWZmM2JiNWQ0NmY0YWE3NGQ4MTgzNjRhNDEzOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84YzUzMGEwN2U3ZTI0NWEyYjRlMjEyYzg1Mzc0NTJlNyA9ICQoJzxkaXYgaWQ9Imh0bWxfOGM1MzBhMDdlN2UyNDVhMmI0ZTIxMmM4NTM3NDUyZTciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdMOpIFNhaW50ZS1Bbm5lLCBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzNkNWM1ZmYzYmI1ZDQ2ZjRhYTc0ZDgxODM2NGE0MTM4LnNldENvbnRlbnQoaHRtbF84YzUzMGEwN2U3ZTI0NWEyYjRlMjEyYzg1Mzc0NTJlNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81ZmQ1NmE2YjUwNTY0MjliODc2OWI4ZTFiZjlhYzA2NC5iaW5kUG9wdXAocG9wdXBfM2Q1YzVmZjNiYjVkNDZmNGFhNzRkODE4MzY0YTQxMzgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGU5M2I5MmY0ZTRhNGI2ZmI0OTE4ODNjMGQ4MWY3ODIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0Ni41MDE1Mjg5LC04NC4yODc4Njk0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjk5MDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmY5OTAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83ZWIwNmNhZTM0YTY0Zjk3OTRiZjYzNmJkN2JmNGRlOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kYWJmNTFlYjNjOGQ0YTI3YTM0MDY3Y2VjN2UyNTE4YiA9ICQoJzxkaXYgaWQ9Imh0bWxfZGFiZjUxZWIzYzhkNGEyN2EzNDA2N2NlYzdlMjUxOGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFsZ29tYSBVbml2ZXJzaXR5LCBDbHVzdGVyIDM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzdlYjA2Y2FlMzRhNjRmOTc5NGJmNjM2YmQ3YmY0ZGU4LnNldENvbnRlbnQoaHRtbF9kYWJmNTFlYjNjOGQ0YTI3YTM0MDY3Y2VjN2UyNTE4Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84ZTkzYjkyZjRlNGE0YjZmYjQ5MTg4M2MwZDgxZjc4Mi5iaW5kUG9wdXAocG9wdXBfN2ViMDZjYWUzNGE2NGY5Nzk0YmY2MzZiZDdiZjRkZTgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzdlM2I0ZDhmY2U5NDhjNGJhM2VhNGZjMDRhZmY5ODEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My4xMTc1NzMxLC03OS4yNDc2OTI1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjk5MDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmY5OTAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zZmM4M2Q4YzI1NGE0NTdjYjEyYmIxZTQ4OGFlMWJlNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82NWRlNTk4YjcyYzk0OWU5YmJlOTE4MWE5NDYyYzhmYyA9ICQoJzxkaXYgaWQ9Imh0bWxfNjVkZTU5OGI3MmM5NDllOWJiZTkxODFhOTQ2MmM4ZmMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJyb2NrIFVuaXZlcnNpdHksIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2ZjODNkOGMyNTRhNDU3Y2IxMmJiMWU0ODhhZTFiZTcuc2V0Q29udGVudChodG1sXzY1ZGU1OThiNzJjOTQ5ZTliYmU5MTgxYTk0NjJjOGZjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M3ZTNiNGQ4ZmNlOTQ4YzRiYTNlYTRmYzA0YWZmOTgxLmJpbmRQb3B1cChwb3B1cF8zZmM4M2Q4YzI1NGE0NTdjYjEyYmIxZTQ4OGFlMWJlNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lMTQyMzBlOGMwZmM0MGRhOGEyZWVmYmYzYjZmZTJkMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ1LjM4MzA4MTksLTc1LjY5ODMxMjFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y2ZjU0ZThmMzQ5ZTRhYmViMjdlM2MwMmQ5ODBiNDFlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzU5YTM5MzNiMGNkZjQzZmFiNjMyMWM1OGE0MTkyYzFhID0gJCgnPGRpdiBpZD0iaHRtbF81OWEzOTMzYjBjZGY0M2ZhYjYzMjFjNThhNDE5MmMxYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2FybGV0b24gVW5pdmVyc2l0eSwgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mNmY1NGU4ZjM0OWU0YWJlYjI3ZTNjMDJkOTgwYjQxZS5zZXRDb250ZW50KGh0bWxfNTlhMzkzM2IwY2RmNDNmYWI2MzIxYzU4YTQxOTJjMWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTE0MjMwZThjMGZjNDBkYThhMmVlZmJmM2I2ZmUyZDAuYmluZFBvcHVwKHBvcHVwX2Y2ZjU0ZThmMzQ5ZTRhYmViMjdlM2MwMmQ5ODBiNDFlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M2MzdjMzdjN2M4NjRhYzk4MWM3ZjA0ZmVlYWFlNjgzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDguNDIxMTEwODAwMDAwMDEsLTg5LjI2MDY5OTRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMzMwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjMzMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzFjMTM0MGNjZjg3MzRiOTViMjNlMzEwNDE3OTdmMTUzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZhNGJjZTYxZGM1ZTRiMTNiNTM5MzQxNmY5NjdkZTMxID0gJCgnPGRpdiBpZD0iaHRtbF82YTRiY2U2MWRjNWU0YjEzYjUzOTM0MTZmOTY3ZGUzMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGFrZWhlYWQgVW5pdmVyc2l0eSwgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xYzEzNDBjY2Y4NzM0Yjk1YjIzZTMxMDQxNzk3ZjE1My5zZXRDb250ZW50KGh0bWxfNmE0YmNlNjFkYzVlNGIxM2I1MzkzNDE2Zjk2N2RlMzEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzYzN2MzN2M3Yzg2NGFjOTgxYzdmMDRmZWVhYWU2ODMuYmluZFBvcHVwKHBvcHVwXzFjMTM0MGNjZjg3MzRiOTViMjNlMzEwNDE3OTdmMTUzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZhY2RmNjk3MzEzZDQ1ZjhhZjk3MmE5ZTMxMDg3NzQ5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDYuNDY2NzcwOCwtODAuOTc0MjMzMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYzMzAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMzMwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDJiMDkwM2NhZjczNGNjMmExYWI0NjY1MTAyZTRhZjIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTJhYmQ4ZWQxOTk2NDU3OTkzNDJiYzM5MThjMjkxNmQgPSAkKCc8ZGl2IGlkPSJodG1sX2UyYWJkOGVkMTk5NjQ1Nzk5MzQyYmMzOTE4YzI5MTZkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MYXVyZW50aWFuIFVuaXZlcnNpdHksIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDJiMDkwM2NhZjczNGNjMmExYWI0NjY1MTAyZTRhZjIuc2V0Q29udGVudChodG1sX2UyYWJkOGVkMTk5NjQ1Nzk5MzQyYmMzOTE4YzI5MTZkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZhY2RmNjk3MzEzZDQ1ZjhhZjk3MmE5ZTMxMDg3NzQ5LmJpbmRQb3B1cChwb3B1cF8wMmIwOTAzY2FmNzM0Y2MyYTFhYjQ2NjUxMDJlNGFmMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jZGE3NzBhZTRjZjc0ZTc4OWQwNWU1OGFiOWM3N2NhOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjI2MDg3OSwtNzkuOTE5MjI1NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZmZjAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmZmYwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDU2YTBiZGQxYzRiNDhlMDg3YTFmNzg1YzgzNTgzYmMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTc3N2I3OTUwNzFkNGJjYWFkMTE2MTQyMjExZDIyY2YgPSAkKCc8ZGl2IGlkPSJodG1sXzk3NzdiNzk1MDcxZDRiY2FhZDExNjE0MjIxMWQyMmNmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NY01hc3RlciBVbml2ZXJzaXR5LCBDbHVzdGVyIDU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ1NmEwYmRkMWM0YjQ4ZTA4N2ExZjc4NWM4MzU4M2JjLnNldENvbnRlbnQoaHRtbF85Nzc3Yjc5NTA3MWQ0YmNhYWQxMTYxNDIyMTFkMjJjZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jZGE3NzBhZTRjZjc0ZTc4OWQwNWU1OGFiOWM3N2NhOS5iaW5kUG9wdXAocG9wdXBfNDU2YTBiZGQxYzRiNDhlMDg3YTFmNzg1YzgzNTgzYmMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjVmYTE3ZmZmZGJhNDQ5NTllNDVhZjhjOThmZjBmNDUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0Ni4zNDMyMDk0LC03OS40OTIzMDE3OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmY5OTAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmOTkwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTU3MzZlNTcwNGUwNDQyOGFmMjhjMDZkMjYxNjg4OTcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjUxMzk5MzA0NjYyNDdmZmE2NWJhZjA4OTk1ZWJmMWQgPSAkKCc8ZGl2IGlkPSJodG1sX2Y1MTM5OTMwNDY2MjQ3ZmZhNjViYWYwODk5NWViZjFkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5OaXBpc3NpbmcgVW5pdmVyc2l0eSwgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85NTczNmU1NzA0ZTA0NDI4YWYyOGMwNmQyNjE2ODg5Ny5zZXRDb250ZW50KGh0bWxfZjUxMzk5MzA0NjYyNDdmZmE2NWJhZjA4OTk1ZWJmMWQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjVmYTE3ZmZmZGJhNDQ5NTllNDVhZjhjOThmZjBmNDUuYmluZFBvcHVwKHBvcHVwXzk1NzM2ZTU3MDRlMDQ0MjhhZjI4YzA2ZDI2MTY4ODk3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE0MzVhYmVjY2VkYjQ1Mjg4MTM3YmFjN2ZjN2NhMTU2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUyOTkzNTk5OTk5OTksLTc5LjM5MTIxNjU5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjMzMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYzMzAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iZjRkZTRkOGYyOGI0MzM1OTU3M2VmNjM2NjdmMzJmZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82ZTJmM2MzNTMxYWM0NTcyOTFiNDM4YTc1MDJmNGU0OSA9ICQoJzxkaXYgaWQ9Imh0bWxfNmUyZjNjMzUzMWFjNDU3MjkxYjQzOGE3NTAyZjRlNDkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk9udGFyaW8gQ29sbGVnZSBvZiBBcnQgYW5kIERlc2lnbiBVbml2ZXJzaXR5LCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2JmNGRlNGQ4ZjI4YjQzMzU5NTczZWY2MzY2N2YzMmZmLnNldENvbnRlbnQoaHRtbF82ZTJmM2MzNTMxYWM0NTcyOTFiNDM4YTc1MDJmNGU0OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xNDM1YWJlY2NlZGI0NTI4ODEzN2JhYzdmYzdjYTE1Ni5iaW5kUG9wdXAocG9wdXBfYmY0ZGU0ZDhmMjhiNDMzNTk1NzNlZjYzNjY3ZjMyZmYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzBiYjRhY2JjZDJhNDcwYzk3NzJmZjIzMjQ3OGMwYTIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0NC4yMjUyNzk1LC03Ni40OTUxNDExOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmM1NTg3M2ExOGVhNDY4Yzk2YWM4ZTY0Y2Q0NWQ2NzggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTllYjNhNjUwZjNkNDlmZTk5OWJhNWFkNjY5ZTEyMzkgPSAkKCc8ZGl2IGlkPSJodG1sXzk5ZWIzYTY1MGYzZDQ5ZmU5OTliYTVhZDY2OWUxMjM5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5RdWVlbiBzIFVuaXZlcnNpdHkgYXQgS2luZ3N0b24sIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZmM1NTg3M2ExOGVhNDY4Yzk2YWM4ZTY0Y2Q0NWQ2Nzguc2V0Q29udGVudChodG1sXzk5ZWIzYTY1MGYzZDQ5ZmU5OTliYTVhZDY2OWUxMjM5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzMwYmI0YWNiY2QyYTQ3MGM5NzcyZmYyMzI0NzhjMGEyLmJpbmRQb3B1cChwb3B1cF9mYzU1ODczYTE4ZWE0NjhjOTZhYzhlNjRjZDQ1ZDY3OCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82YTk3Njk3NzAwYWE0M2Q1ODFhMDk2OWI1NDhiNmVjNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ0LjIzMzg4MTIsLTc2LjQ2NzUwNTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzc5MjkwZmRhNDdhMDRiMjVhNmE0MThlYTVkYWVkMTk5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y1MWIzOTk2ZjhjMjRlYTBiOWZkOGIxNjg4NjA4NWI2ID0gJCgnPGRpdiBpZD0iaHRtbF9mNTFiMzk5NmY4YzI0ZWEwYjlmZDhiMTY4ODYwODViNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um95YWwgTWlsaXRhcnkgQ29sbGVnZSBvZiBDYW5hZGEsIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzkyOTBmZGE0N2EwNGIyNWE2YTQxOGVhNWRhZWQxOTkuc2V0Q29udGVudChodG1sX2Y1MWIzOTk2ZjhjMjRlYTBiOWZkOGIxNjg4NjA4NWI2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZhOTc2OTc3MDBhYTQzZDU4MWEwOTY5YjU0OGI2ZWM1LmJpbmRQb3B1cChwb3B1cF83OTI5MGZkYTQ3YTA0YjI1YTZhNDE4ZWE1ZGFlZDE5OSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jY2MxYjdjMDExZTY0ZjczODFkOTk3YzVhOTc1YzgwMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1NzY1ODUsLTc5LjM3ODgwMTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2EwMTUzZThmMTE2ZTRlMmU5NzZlMDNhY2M4NTc0NmMxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2VjMGYwYjI5ZjZhMzQ4Mzg4NDMzNTM0YmUzZjg2MjBjID0gJCgnPGRpdiBpZD0iaHRtbF9lYzBmMGIyOWY2YTM0ODM4ODQzMzUzNGJlM2Y4NjIwYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnllcnNvbiBVbml2ZXJzaXR5LCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2EwMTUzZThmMTE2ZTRlMmU5NzZlMDNhY2M4NTc0NmMxLnNldENvbnRlbnQoaHRtbF9lYzBmMGIyOWY2YTM0ODM4ODQzMzUzNGJlM2Y4NjIwYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jY2MxYjdjMDExZTY0ZjczODFkOTk3YzVhOTc1YzgwMS5iaW5kUG9wdXAocG9wdXBfYTAxNTNlOGYxMTZlNGUyZTk3NmUwM2FjYzg1NzQ2YzEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTllZGVjMTc3OTg3NGM5MDlmOGE5N2Y4OTE2ZTc4ZDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDQ0ODU0LC03OS4zNjkwNjIyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83YjBiOWU0YjI2ZGM0ZWMwYmIxYWVhYzJmOGRjODJkMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83N2U3NGJjYWM4ZGY0Y2M3ODEwYzY4YzJhODk0M2RjYyA9ICQoJzxkaXYgaWQ9Imh0bWxfNzdlNzRiY2FjOGRmNGNjNzgxMGM2OGMyYTg5NDNkY2MiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdMOpIGRlIGwgT250YXJpbyBmcmFuw6dhaXMsIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfN2IwYjllNGIyNmRjNGVjMGJiMWFlYWMyZjhkYzgyZDEuc2V0Q29udGVudChodG1sXzc3ZTc0YmNhYzhkZjRjYzc4MTBjNjhjMmE4OTQzZGNjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk5ZWRlYzE3Nzk4NzRjOTA5ZjhhOTdmODkxNmU3OGQxLmJpbmRQb3B1cChwb3B1cF83YjBiOWU0YjI2ZGM0ZWMwYmIxYWVhYzJmOGRjODJkMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jMGNlYzBkN2Q0MjI0OTljOTZkYTFmNDI5MTQ1MTlhYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjUzMjcyMTcsLTgwLjIyNjE4MDM5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80MmU2ODBiMjRhNmE0NGIzOGNlYmZlYTkzNmFmMzI2OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kMTk2MmExYWVhMDU0MDg4OTcwYjliZmVkZjZlZDllMiA9ICQoJzxkaXYgaWQ9Imh0bWxfZDE5NjJhMWFlYTA1NDA4ODk3MGI5YmZlZGY2ZWQ5ZTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdHkgb2YgR3VlbHBoLCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQyZTY4MGIyNGE2YTQ0YjM4Y2ViZmVhOTM2YWYzMjY4LnNldENvbnRlbnQoaHRtbF9kMTk2MmExYWVhMDU0MDg4OTcwYjliZmVkZjZlZDllMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jMGNlYzBkN2Q0MjI0OTljOTZkYTFmNDI5MTQ1MTlhYS5iaW5kUG9wdXAocG9wdXBfNDJlNjgwYjI0YTZhNDRiMzhjZWJmZWE5MzZhZjMyNjgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmE3M2I0ZDU0YmI0NDU5ZGIxZWE2ZGJhYzdmODBjZjAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My45NDU3NTc5LC03OC44OTYwMDkyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmNjMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZjYzAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83YzkxNmU2ZTJiMjY0MDlhOTJhOGJlOGMxNjNmNDJhMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81ODU1MWYwNDMwZmQ0NjgyOWEwMzdiYmY0MzVkNzA1NiA9ICQoJzxkaXYgaWQ9Imh0bWxfNTg1NTFmMDQzMGZkNDY4MjlhMDM3YmJmNDM1ZDcwNTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk9udGFyaW8gVGVjaCBVbml2ZXJzaXR5LCBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzdjOTE2ZTZlMmIyNjQwOWE5MmE4YmU4YzE2M2Y0MmExLnNldENvbnRlbnQoaHRtbF81ODU1MWYwNDMwZmQ0NjgyOWEwMzdiYmY0MzVkNzA1Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYTczYjRkNTRiYjQ0NTlkYjFlYTZkYmFjN2Y4MGNmMC5iaW5kUG9wdXAocG9wdXBfN2M5MTZlNmUyYjI2NDA5YTkyYThiZThjMTYzZjQyYTEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzgxMjgyMzYyODk5NDY0OTg3NDAzYTA1MDZiNWU2ZjAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0NS40MjMxMDYzOTk5OTk5OSwtNzUuNjgzMTMyODk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMzMwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjMzMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzk5ZDUwZWJiNGRlYTQwMGM4MTk0N2NhZjAwMDQzYjdmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzFjMGZlMjI1YzhiYjQzNThiNjU2MTc3MDZkYTEwNmZhID0gJCgnPGRpdiBpZD0iaHRtbF8xYzBmZTIyNWM4YmI0MzU4YjY1NjE3NzA2ZGExMDZmYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VW5pdmVyc2l0eSBvZiBPdHRhd2EsIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTlkNTBlYmI0ZGVhNDAwYzgxOTQ3Y2FmMDAwNDNiN2Yuc2V0Q29udGVudChodG1sXzFjMGZlMjI1YzhiYjQzNThiNjU2MTc3MDZkYTEwNmZhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzM4MTI4MjM2Mjg5OTQ2NDk4NzQwM2EwNTA2YjVlNmYwLmJpbmRQb3B1cChwb3B1cF85OWQ1MGViYjRkZWE0MDBjODE5NDdjYWYwMDA0M2I3Zik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81MmZmNTgyOGJmMDg0Y2ZjYWZhNGExMjA4NGE0ZmU0YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Mjg5MTcsLTc5LjM5NTY1NjQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83M2EzMmZjMTViODU0MzE5OGY4ZTEwMDczOWQ1ZDcxZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xMzExODAyODJmZmY0MDRjYmFmMzI3ZmEwOTMxNTRjZSA9ICQoJzxkaXYgaWQ9Imh0bWxfMTMxMTgwMjgyZmZmNDA0Y2JhZjMyN2ZhMDkzMTU0Y2UiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdHkgb2YgVG9yb250bywgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83M2EzMmZjMTViODU0MzE5OGY4ZTEwMDczOWQ1ZDcxZS5zZXRDb250ZW50KGh0bWxfMTMxMTgwMjgyZmZmNDA0Y2JhZjMyN2ZhMDkzMTU0Y2UpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTJmZjU4MjhiZjA4NGNmY2FmYTRhMTIwODRhNGZlNGMuYmluZFBvcHVwKHBvcHVwXzczYTMyZmMxNWI4NTQzMTk4ZjhlMTAwNzM5ZDVkNzFlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU1MjMzOWUzYTk2YzRiNjA5N2Y2N2ZkMzdlNzVhOGMwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNDcyMjg1NCwtODAuNTQ0ODU3Nl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYzMzAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMzMwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzA2ODQ1M2YwOWU3NGYwY2E5Yzk4Yjg2NGUzOGVjZTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDcwYjJlOGU2Yjk0NGM0ZTlmNTZiNzE3OWY4YTQ0ZDEgPSAkKCc8ZGl2IGlkPSJodG1sXzA3MGIyZThlNmI5NDRjNGU5ZjU2YjcxNzlmOGE0NGQxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIFdhdGVybG9vLCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMwNjg0NTNmMDllNzRmMGNhOWM5OGI4NjRlMzhlY2U4LnNldENvbnRlbnQoaHRtbF8wNzBiMmU4ZTZiOTQ0YzRlOWY1NmI3MTc5ZjhhNDRkMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81NTIzMzllM2E5NmM0YjYwOTdmNjdmZDM3ZTc1YThjMC5iaW5kUG9wdXAocG9wdXBfMzA2ODQ1M2YwOWU3NGYwY2E5Yzk4Yjg2NGUzOGVjZTgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTdmZTQ1ZTA5MDU1NDVmOWI4NDdiOGQxZWQxMDMxMWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My4wMDk1OTcxLC04MS4yNzM3MzM2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iYmI4NjY5NWM5MmM0YmZmYTgxNzI3OWY4ZTYwOWM5NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84MTQ4MGNlMzk2NDE0ODcyYjZhZjRlN2E5OGMwOWZiNiA9ICQoJzxkaXYgaWQ9Imh0bWxfODE0ODBjZTM5NjQxNDg3MmI2YWY0ZTdhOThjMDlmYjYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdHkgb2YgV2VzdGVybiBPbnRhcmlvLCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2JiYjg2Njk1YzkyYzRiZmZhODE3Mjc5ZjhlNjA5Yzk3LnNldENvbnRlbnQoaHRtbF84MTQ4MGNlMzk2NDE0ODcyYjZhZjRlN2E5OGMwOWZiNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xN2ZlNDVlMDkwNTU0NWY5Yjg0N2I4ZDFlZDEwMzExZC5iaW5kUG9wdXAocG9wdXBfYmJiODY2OTVjOTJjNGJmZmE4MTcyNzlmOGU2MDljOTcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGFlMTYwMDliMjI1NGFjYTgyZGI0OTcwNmIwZTcwOGIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0Mi4zMDQzMTQyLC04My4wNjYwMzg5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZmZjAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmZmYwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWIyZWVlOWZkNGM5NDZkMzllZjU2MjMzZmMzOGM2MzAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDcwOGE1OWVlZTQ5NDM5ZTg1MDc4MDE4N2RkYzdlMDQgPSAkKCc8ZGl2IGlkPSJodG1sXzA3MDhhNTllZWU0OTQzOWU4NTA3ODAxODdkZGM3ZTA0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIFdpbmRzb3IsIENsdXN0ZXIgNTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZWIyZWVlOWZkNGM5NDZkMzllZjU2MjMzZmMzOGM2MzAuc2V0Q29udGVudChodG1sXzA3MDhhNTllZWU0OTQzOWU4NTA3ODAxODdkZGM3ZTA0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBhZTE2MDA5YjIyNTRhY2E4MmRiNDk3MDZiMGU3MDhiLmJpbmRQb3B1cChwb3B1cF9lYjJlZWU5ZmQ0Yzk0NmQzOWVmNTYyMzNmYzM4YzYzMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82N2Q2ZDk4ZjEyMzY0Mjc5YmIyNmMxYzM0OTQ4YmIwNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjQ3Mzk1NjIsLTgwLjUyNzc3NDldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMzMwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjMzMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JiZjBlYzE3MDMxZDRkNjE5MTJmYjkyOGQ1MTQ1NmE3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzgxNDQ5NjQxOTQ1YjRmMzVhM2MwMzY0NWFmZDhlNmQxID0gJCgnPGRpdiBpZD0iaHRtbF84MTQ0OTY0MTk0NWI0ZjM1YTNjMDM2NDVhZmQ4ZTZkMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2lsZnJpZCBMYXVyaWVyIFVuaXZlcnNpdHksIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmJmMGVjMTcwMzFkNGQ2MTkxMmZiOTI4ZDUxNDU2YTcuc2V0Q29udGVudChodG1sXzgxNDQ5NjQxOTQ1YjRmMzVhM2MwMzY0NWFmZDhlNmQxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzY3ZDZkOThmMTIzNjQyNzliYjI2YzFjMzQ5NDhiYjA2LmJpbmRQb3B1cChwb3B1cF9iYmYwZWMxNzAzMWQ0ZDYxOTEyZmI5MjhkNTE0NTZhNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mYTJhZTkyMTEzYWI0NTE1ODQ1ZTdmNTIyMjMzYmM0MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc3MzQ1MzUsLTc5LjUwMTg2ODM5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjMzMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYzMzAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yZmYwM2RiNDgwYTY0NGQ0YTI5N2RhZDJjZGI2NmIzZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81YTA3ZGZjZDI4NjE0M2MzOTY0YmQ4ODQ3NzkzMTE1ZiA9ICQoJzxkaXYgaWQ9Imh0bWxfNWEwN2RmY2QyODYxNDNjMzk2NGJkODg0Nzc5MzExNWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPllvcmsgVW5pdmVyc2l0eSwgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yZmYwM2RiNDgwYTY0NGQ0YTI5N2RhZDJjZGI2NmIzZS5zZXRDb250ZW50KGh0bWxfNWEwN2RmY2QyODYxNDNjMzk2NGJkODg0Nzc5MzExNWYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmEyYWU5MjExM2FiNDUxNTg0NWU3ZjUyMjIzM2JjNDEuYmluZFBvcHVwKHBvcHVwXzJmZjAzZGI0ODBhNjQ0ZDRhMjk3ZGFkMmNkYjY2YjNlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRiZTVjNGM0YzAwMjQ1MGNiNTQ5YmZjMGQ2MGU4OTEyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDYuMjU3NDkyLC02My4xMzc1MDc0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjk5MDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmY5OTAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iMThlNTU5YTliZWQ0OWRkOGJkZTM1YTBkMDMxMDM5MCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hMjg5ZTU2MDQ3MzA0MGNiYTJjOTk5MDYzMTM4OTdkMSA9ICQoJzxkaXYgaWQ9Imh0bWxfYTI4OWU1NjA0NzMwNDBjYmEyYzk5OTA2MzEzODk3ZDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdHkgb2YgUHJpbmNlIEVkd2FyZCBJc2xhbmQsIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjE4ZTU1OWE5YmVkNDlkZDhiZGUzNWEwZDAzMTAzOTAuc2V0Q29udGVudChodG1sX2EyODllNTYwNDczMDQwY2JhMmM5OTkwNjMxMzg5N2QxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzRiZTVjNGM0YzAwMjQ1MGNiNTQ5YmZjMGQ2MGU4OTEyLmJpbmRQb3B1cChwb3B1cF9iMThlNTU5YTliZWQ0OWRkOGJkZTM1YTBkMDMxMDM5MCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81YmM5OTVkODY5MmI0YjI5YTUzZjYxZDgyN2UxODdkOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ1LjM2Mjg1MjgsLTcxLjg0NTY1NjldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmOTkwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjk5MDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzc1NmNhMDliMTdkMDRhYjRhNTdlNzRmZjYzZDQyYzZmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzBhODNmMzAxYTVhMDQzZTY4NjJhOTI5NGQ4MDI1ZGZjID0gJCgnPGRpdiBpZD0iaHRtbF8wYTgzZjMwMWE1YTA0M2U2ODYyYTkyOTRkODAyNWRmYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmlzaG9wIHMgVW5pdmVyc2l0eSwgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83NTZjYTA5YjE3ZDA0YWI0YTU3ZTc0ZmY2M2Q0MmM2Zi5zZXRDb250ZW50KGh0bWxfMGE4M2YzMDFhNWEwNDNlNjg2MmE5Mjk0ZDgwMjVkZmMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNWJjOTk1ZDg2OTJiNGIyOWE1M2Y2MWQ4MjdlMTg3ZDkuYmluZFBvcHVwKHBvcHVwXzc1NmNhMDliMTdkMDRhYjRhNTdlNzRmZjYzZDQyYzZmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBjMThiNGY2M2ZmOTQ3NTc5NGIwYjFiODMwNThiODQ1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDUuNDk0NTY0MywtNzMuNTc3Mzc3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTY3ZjcwY2JiN2U4NDBkYTlhNmNmNGE4ZjAwOGNmZGQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODg2NjVmNTM2ZmY4NDljY2FmMjZhNmQ4ZWYyMDNlZmYgPSAkKCc8ZGl2IGlkPSJodG1sXzg4NjY1ZjUzNmZmODQ5Y2NhZjI2YTZkOGVmMjAzZWZmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Db25jb3JkaWEgVW5pdmVyc2l0eSwgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85NjdmNzBjYmI3ZTg0MGRhOWE2Y2Y0YThmMDA4Y2ZkZC5zZXRDb250ZW50KGh0bWxfODg2NjVmNTM2ZmY4NDljY2FmMjZhNmQ4ZWYyMDNlZmYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGMxOGI0ZjYzZmY5NDc1Nzk0YjBiMWI4MzA1OGI4NDUuYmluZFBvcHVwKHBvcHVwXzk2N2Y3MGNiYjdlODQwZGE5YTZjZjRhOGYwMDhjZmRkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzIzNWM5YmM4NWVhMDRiZTViNjY3MDUwM2IzODNjN2VhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDUuNDk0NTg3NywtNzMuNTYyMjgxNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMmYxYzRlODhjMjlmNDYyMTlhYTZmMTFkOWRhYWVkNjAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODc0YTVmYzk4ODYwNGQ5NTlhNDQ2ZDgzYzAyZTJlNzMgPSAkKCc8ZGl2IGlkPSJodG1sXzg3NGE1ZmM5ODg2MDRkOTU5YTQ0NmQ4M2MwMmUyZTczIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7DiWNvbGUgZGUgdGVjaG5vbG9naWUgc3Vww6lyaWV1cmUsIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMmYxYzRlODhjMjlmNDYyMTlhYTZmMTFkOWRhYWVkNjAuc2V0Q29udGVudChodG1sXzg3NGE1ZmM5ODg2MDRkOTU5YTQ0NmQ4M2MwMmUyZTczKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzIzNWM5YmM4NWVhMDRiZTViNjY3MDUwM2IzODNjN2VhLmJpbmRQb3B1cChwb3B1cF8yZjFjNGU4OGMyOWY0NjIxOWFhNmYxMWQ5ZGFhZWQ2MCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xMzI1NzFmMjRiMjI0NTdmYTBiM2IxYmY4ODRhMjhmYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ2LjgxMzgwMDksLTcxLjIyMjQ3NjldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzVlZjViYjE4MmIxNTQ1MmRiYTdhMGU5ZDhkODdmNTUzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzI3OWMwNGZmOTc1ODQwOWE5ODNkMzJjN2Y2ZDMwN2Q0ID0gJCgnPGRpdiBpZD0iaHRtbF8yNzljMDRmZjk3NTg0MDlhOTgzZDMyYzdmNmQzMDdkNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+w4ljb2xlIG5hdGlvbmFsZSBkIGFkbWluaXN0cmF0aW9uIHB1YmxpcXVlLCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVlZjViYjE4MmIxNTQ1MmRiYTdhMGU5ZDhkODdmNTUzLnNldENvbnRlbnQoaHRtbF8yNzljMDRmZjk3NTg0MDlhOTgzZDMyYzdmNmQzMDdkNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xMzI1NzFmMjRiMjI0NTdmYTBiM2IxYmY4ODRhMjhmYS5iaW5kUG9wdXAocG9wdXBfNWVmNWJiMTgyYjE1NDUyZGJhN2EwZTlkOGQ4N2Y1NTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjlmZmEzNGY4YTUwNGI1YTljNjk1NDUwOWZhN2JmODYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0Ni44MTMwMTU5LC03MS4yMjQwMjQ4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jZTliMDM2MDE3ODE0YzcyOTg2MTNmNjJiZmIyM2IyNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82OTk5ODkzYmE0M2I0Mjc2OWIwZWJkMDUyMDMzNTA3OSA9ICQoJzxkaXYgaWQ9Imh0bWxfNjk5OTg5M2JhNDNiNDI3NjliMGViZDA1MjAzMzUwNzkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkluc3RpdHV0IG5hdGlvbmFsIGRlIGxhIHJlY2hlcmNoZSBzY2llbnRpZmlxdWUsIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2U5YjAzNjAxNzgxNGM3Mjk4NjEzZjYyYmZiMjNiMjQuc2V0Q29udGVudChodG1sXzY5OTk4OTNiYTQzYjQyNzY5YjBlYmQwNTIwMzM1MDc5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzI5ZmZhMzRmOGE1MDRiNWE5YzY5NTQ1MDlmYTdiZjg2LmJpbmRQb3B1cChwb3B1cF9jZTliMDM2MDE3ODE0YzcyOTg2MTNmNjJiZmIyM2IyNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yZTUxZjhjMDExZmQ0Mjk4ODA5MWVlYjY2MWJhNjE4NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ1LjUwNDc4NDY5OTk5OTk5LC03My41NzcxNTExXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jMTUyYWY3ZGZkMGQ0YzljYWY5NzFkYzE2YzZmNzVjMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kOTk3MmZkN2I0MmE0Y2JlYjVlYWNkM2M0YWYxNDJkMCA9ICQoJzxkaXYgaWQ9Imh0bWxfZDk5NzJmZDdiNDJhNGNiZWI1ZWFjZDNjNGFmMTQyZDAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1jR2lsbCBVbml2ZXJzaXR5LCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MxNTJhZjdkZmQwZDRjOWNhZjk3MWRjMTZjNmY3NWMwLnNldENvbnRlbnQoaHRtbF9kOTk3MmZkN2I0MmE0Y2JlYjVlYWNkM2M0YWYxNDJkMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yZTUxZjhjMDExZmQ0Mjk4ODA5MWVlYjY2MWJhNjE4Ny5iaW5kUG9wdXAocG9wdXBfYzE1MmFmN2RmZDBkNGM5Y2FmOTcxZGMxNmM2Zjc1YzApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2RlNjI4YTFiM2JlNDk2Y2EwYTZkNzE5ZjEwMzZiODQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0NS41MjMxMTA0LC03My42MTk2NjA1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iMTVkZDI0OTUwN2I0NTc3OWZlZmJjOTM2MDc2OGEwZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83YTExYjAyMDk3MzY0ZDE2Yjg5MTVjMmUzZjU1NmMxNiA9ICQoJzxkaXYgaWQ9Imh0bWxfN2ExMWIwMjA5NzM2NGQxNmI4OTE1YzJlM2Y1NTZjMTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdMOpIGRlIE1vbnRyw6lhbCwgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iMTVkZDI0OTUwN2I0NTc3OWZlZmJjOTM2MDc2OGEwZS5zZXRDb250ZW50KGh0bWxfN2ExMWIwMjA5NzM2NGQxNmI4OTE1YzJlM2Y1NTZjMTYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2RlNjI4YTFiM2JlNDk2Y2EwYTZkNzE5ZjEwMzZiODQuYmluZFBvcHVwKHBvcHVwX2IxNWRkMjQ5NTA3YjQ1Nzc5ZmVmYmM5MzYwNzY4YTBlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFhODAzNDk5ZTk2ZDRjZjU4ZjIyNzllZTliNWUyOWZiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDUuMzc3OTQzMywtNzEuOTI5Mzg1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjk5MDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmY5OTAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hYWYzNWYxMjk0ZDE0MGU1ODkxYTg2OThjZDBlODM0NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kNWI0YmY2Yzg3N2M0ZDFhYTQ4ZmYyNWVmYTFjMzYxMCA9ICQoJzxkaXYgaWQ9Imh0bWxfZDViNGJmNmM4NzdjNGQxYWE0OGZmMjVlZmExYzM2MTAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdMOpIGRlIFNoZXJicm9va2UsIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYWFmMzVmMTI5NGQxNDBlNTg5MWE4Njk4Y2QwZTgzNDUuc2V0Q29udGVudChodG1sX2Q1YjRiZjZjODc3YzRkMWFhNDhmZjI1ZWZhMWMzNjEwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFhODAzNDk5ZTk2ZDRjZjU4ZjIyNzllZTliNWUyOWZiLmJpbmRQb3B1cChwb3B1cF9hYWYzNWYxMjk0ZDE0MGU1ODkxYTg2OThjZDBlODM0NSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wNzY0ZWU3ODJiMzc0MTQyODQwZGI0Y2EyNjdjMTJiMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ4LjIzMDU3Mzc5OTk5OTk5LC03OS4wMDgyOTA1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjk5MDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmY5OTAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iZmM4ZmIyZWNlYTI0MjZiYjRiYzAwMDU3Y2VkMDY0YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hMDRlNGQ0NDg2N2U0ZDNjYTI3OTdmYzQxYTlhMTM4YiA9ICQoJzxkaXYgaWQ9Imh0bWxfYTA0ZTRkNDQ4NjdlNGQzY2EyNzk3ZmM0MWE5YTEzOGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdMOpIGR1IFF1w6liZWMgZW4gQWJpdGliaS1Uw6ltaXNjYW1pbmd1ZSwgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iZmM4ZmIyZWNlYTI0MjZiYjRiYzAwMDU3Y2VkMDY0Yi5zZXRDb250ZW50KGh0bWxfYTA0ZTRkNDQ4NjdlNGQzY2EyNzk3ZmM0MWE5YTEzOGIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDc2NGVlNzgyYjM3NDE0Mjg0MGRiNGNhMjY3YzEyYjIuYmluZFBvcHVwKHBvcHVwX2JmYzhmYjJlY2VhMjQyNmJiNGJjMDAwNTdjZWQwNjRiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJhZGMwYmVmY2QzNTQxNjhhMGU5Y2E1ODVmZDQwNDM1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDUuNDIyNDY2LC03NS43Mzg3MDE2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjMzMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYzMzAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yNTAzOTMyOTJhNDI0MjAyYmY2YjI2OGJmZmY5NTM1YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85ZDk2NTY1YzAwNzk0MzM2YTNkOGU1YzRlYzdiOTIzOSA9ICQoJzxkaXYgaWQ9Imh0bWxfOWQ5NjU2NWMwMDc5NDMzNmEzZDhlNWM0ZWM3YjkyMzkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdMOpIGR1IFF1w6liZWMgZW4gT3V0YW91YWlzLCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzI1MDM5MzI5MmE0MjQyMDJiZjZiMjY4YmZmZjk1MzVjLnNldENvbnRlbnQoaHRtbF85ZDk2NTY1YzAwNzk0MzM2YTNkOGU1YzRlYzdiOTIzOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYWRjMGJlZmNkMzU0MTY4YTBlOWNhNTg1ZmQ0MDQzNS5iaW5kUG9wdXAocG9wdXBfMjUwMzkzMjkyYTQyNDIwMmJmNmIyNjhiZmZmOTUzNWMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjFmNDZmMzgyOGE0NGJhZGIzZDA2MDRkOTBmNzViM2QgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OC40MTkwMDgsLTcxLjA1MjYyMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZjYzAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmY2MwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzY2ZmE3Y2Y2YjY1NDcyYTgyYjJlODIwZGYzODBiMmIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmNiYmNjYjQ0ZjExNDk5MWFhYjlmZTM0OGEzMTc2MDUgPSAkKCc8ZGl2IGlkPSJodG1sX2JjYmJjY2I0NGYxMTQ5OTFhYWI5ZmUzNDhhMzE3NjA1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXTDqSBkdSBRdcOpYmVjIMOgIENoaWNvdXRpbWksIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzY2ZmE3Y2Y2YjY1NDcyYTgyYjJlODIwZGYzODBiMmIuc2V0Q29udGVudChodG1sX2JjYmJjY2I0NGYxMTQ5OTFhYWI5ZmUzNDhhMzE3NjA1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzIxZjQ2ZjM4MjhhNDRiYWRiM2QwNjA0ZDkwZjc1YjNkLmJpbmRQb3B1cChwb3B1cF83NjZmYTdjZjZiNjU0NzJhODJiMmU4MjBkZjM4MGIyYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xYmNkNDE3YmRiNGY0OWM3YjE4ZTFmMTlkZWMzNmY3ZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ1LjUxMjU5OTUsLTczLjU2MDU5NTQ5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84YzQxNGFmOTFkM2E0NDUyOTc0Nzk4YTUwYjRjYjg3OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81NDk3NzI5NGVhYzE0N2ViOTJmNzEwMjRjM2UwOThmNiA9ICQoJzxkaXYgaWQ9Imh0bWxfNTQ5NzcyOTRlYWMxNDdlYjkyZjcxMDI0YzNlMDk4ZjYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdMOpIGR1IFF1w6liZWMgw6AgTW9udHLDqWFsLCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzhjNDE0YWY5MWQzYTQ0NTI5NzQ3OThhNTBiNGNiODc5LnNldENvbnRlbnQoaHRtbF81NDk3NzI5NGVhYzE0N2ViOTJmNzEwMjRjM2UwOThmNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xYmNkNDE3YmRiNGY0OWM3YjE4ZTFmMTlkZWMzNmY3ZS5iaW5kUG9wdXAocG9wdXBfOGM0MTRhZjkxZDNhNDQ1Mjk3NDc5OGE1MGI0Y2I4NzkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjI1YjhlNjA1YjA1NDQ1Yzg3YmVmM2VhODc0ZjZlNDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OC40NTI1NjAyOTk5OTk5OSwtNjguNTEyMTMxNzk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzc5OWM3YzRmMTA1ZTRlZThhZmY1N2VjZjhkY2Q0ZDBkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2U1ODZjMDk2NTY4YjRjNWU5YmJmYTBlM2UxZDYwOWMxID0gJCgnPGRpdiBpZD0iaHRtbF9lNTg2YzA5NjU2OGI0YzVlOWJiZmEwZTNlMWQ2MDljMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VW5pdmVyc2l0w6kgZHUgUXXDqWJlYyDDoCBSaW1vdXNraSwgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83OTljN2M0ZjEwNWU0ZWU4YWZmNTdlY2Y4ZGNkNGQwZC5zZXRDb250ZW50KGh0bWxfZTU4NmMwOTY1NjhiNGM1ZTliYmZhMGUzZTFkNjA5YzEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjI1YjhlNjA1YjA1NDQ1Yzg3YmVmM2VhODc0ZjZlNDEuYmluZFBvcHVwKHBvcHVwXzc5OWM3YzRmMTA1ZTRlZThhZmY1N2VjZjhkY2Q0ZDBkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzdiNzY4ZTZlYjU1ZDRmZTQ5YWVlZWE1YWM1NmE4ODYzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDYuMzQ3MjAwNiwtNzIuNTc3MDkzNjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMzMwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjMzMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZhNGY5MjliYTcxYjRhZGU5ODA5ZDhmZmM0ZTU1YzQ0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2JlM2Y3MzNjYWFhYzQ1MzU5ZWQ0MzZlOTJiZTM5NDkyID0gJCgnPGRpdiBpZD0iaHRtbF9iZTNmNzMzY2FhYWM0NTM1OWVkNDM2ZTkyYmUzOTQ5MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VW5pdmVyc2l0w6kgZHUgUXXDqWJlYyDDoCBUcm9pcy1SaXZpw6hyZXMsIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmE0ZjkyOWJhNzFiNGFkZTk4MDlkOGZmYzRlNTVjNDQuc2V0Q29udGVudChodG1sX2JlM2Y3MzNjYWFhYzQ1MzU5ZWQ0MzZlOTJiZTM5NDkyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdiNzY4ZTZlYjU1ZDRmZTQ5YWVlZWE1YWM1NmE4ODYzLmJpbmRQb3B1cChwb3B1cF82YTRmOTI5YmE3MWI0YWRlOTgwOWQ4ZmZjNGU1NWM0NCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wYzdkMDU0OWZlNDI0NThlOWViZjdjOTE1NjVjMWE5MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ2Ljc4MTc0NjI5OTk5OTk5LC03MS4yNzQ3NDI0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmZmMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZmZjAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84Y2JmMDg1MDQ1Y2E0YWQwOWQzZmI4ZjUwMmE1ODYyZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lMDQyNjM2MjgxYTk0YjZiOTc3YWMxYzZkYjMxYzVjMyA9ICQoJzxkaXYgaWQ9Imh0bWxfZTA0MjYzNjI4MWE5NGI2Yjk3N2FjMWM2ZGIzMWM1YzMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdMOpIExhdmFsLCBDbHVzdGVyIDU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzhjYmYwODUwNDVjYTRhZDA5ZDNmYjhmNTAyYTU4NjJkLnNldENvbnRlbnQoaHRtbF9lMDQyNjM2MjgxYTk0YjZiOTc3YWMxYzZkYjMxYzVjMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wYzdkMDU0OWZlNDI0NThlOWViZjdjOTE1NjVjMWE5Mi5iaW5kUG9wdXAocG9wdXBfOGNiZjA4NTA0NWNhNGFkMDlkM2ZiOGY1MDJhNTg2MmQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjA2YzgwYjZkYTUxNGNjNmI1OWIyODU1YjJhNjJiNmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MC40MTU0NTQyLC0xMDQuNTg3ODMwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmY5OTAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmOTkwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTNlNjI0NGVjZTAyNDJkZTkwM2FkNGYyZjQxZTliOGIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTNiYzEyNWU1MDdmNGZkOThlYjdlMDdhNjkzYjY1NDEgPSAkKCc8ZGl2IGlkPSJodG1sXzEzYmMxMjVlNTA3ZjRmZDk4ZWI3ZTA3YTY5M2I2NTQxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIFJlZ2luYSwgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85M2U2MjQ0ZWNlMDI0MmRlOTAzYWQ0ZjJmNDFlOWI4Yi5zZXRDb250ZW50KGh0bWxfMTNiYzEyNWU1MDdmNGZkOThlYjdlMDdhNjkzYjY1NDEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjA2YzgwYjZkYTUxNGNjNmI1OWIyODU1YjJhNjJiNmMuYmluZFBvcHVwKHBvcHVwXzkzZTYyNDRlY2UwMjQyZGU5MDNhZDRmMmY0MWU5YjhiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFiYTU0N2YwMTdlOTQ5ZTVhZDI1ZmVhZjBjZGUzNDZlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTIuMTMzNDAwMywtMTA2LjYzMTM1ODJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMzMwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjMzMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzEwY2E1YjZlOWIzMzRhNzFhYTUyNTZjODZkOWNiZmYyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FkZGNiZjMzOWJiNTRiZjViY2Q4ZjYyMTliZDJiMmRkID0gJCgnPGRpdiBpZD0iaHRtbF9hZGRjYmYzMzliYjU0YmY1YmNkOGY2MjE5YmQyYjJkZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VW5pdmVyc2l0eSBvZiBTYXNrYXRjaGV3YW4sIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTBjYTViNmU5YjMzNGE3MWFhNTI1NmM4NmQ5Y2JmZjIuc2V0Q29udGVudChodG1sX2FkZGNiZjMzOWJiNTRiZjViY2Q4ZjYyMTliZDJiMmRkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFiYTU0N2YwMTdlOTQ5ZTVhZDI1ZmVhZjBjZGUzNDZlLmJpbmRQb3B1cChwb3B1cF8xMGNhNWI2ZTliMzM0YTcxYWE1MjU2Yzg2ZDljYmZmMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85NTdjYjkyMWI0NTY0MTJjYjViMTU3MDAwY2U5NzVjNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ5LjI3NzY5NiwtMTIzLjExNTU1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjMzMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYzMzAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iNjE3ZGJhYTM2N2I0MjEyODYxMGY5MjJjZmM3ZWVmNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yYzM0MWYwMTAyOTE0NDI5YmU2OTM4OGVkNmQ4OWNjYiA9ICQoJzxkaXYgaWQ9Imh0bWxfMmMzNDFmMDEwMjkxNDQyOWJlNjkzODhlZDZkODljY2IiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZhaXJsZWlnaCBEaWNraW5zb24gVW5pdmVyc2l0eSAoYnJhbmNoKSwgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iNjE3ZGJhYTM2N2I0MjEyODYxMGY5MjJjZmM3ZWVmNy5zZXRDb250ZW50KGh0bWxfMmMzNDFmMDEwMjkxNDQyOWJlNjkzODhlZDZkODljY2IpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTU3Y2I5MjFiNDU2NDEyY2I1YjE1NzAwMGNlOTc1YzUuYmluZFBvcHVwKHBvcHVwX2I2MTdkYmFhMzY3YjQyMTI4NjEwZjkyMmNmYzdlZWY3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU4NmEwYTI3OGFmZjRlOWY4OGY1OGZhOTBlMWU1MDljID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDkuMjYxNTIzOCwtMTIzLjA0MjA2NjldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmZmYwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmZmMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzAxY2FiNWJkMjE1NDQxMWM5YWFkNDZmMDEwZDRlNzAyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg3ZjlhNzY2ZjU3ZDQyNGFiNTQyMWJhNGI0MGY5NzY5ID0gJCgnPGRpdiBpZD0iaHRtbF84N2Y5YTc2NmY1N2Q0MjRhYjU0MjFiYTRiNDBmOTc2OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TmV3IFlvcmsgSW5zdGl0dXRlIG9mIFRlY2hub2xvZ3kgKGJyYW5jaCksIENsdXN0ZXIgNTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDFjYWI1YmQyMTU0NDExYzlhYWQ0NmYwMTBkNGU3MDIuc2V0Q29udGVudChodG1sXzg3ZjlhNzY2ZjU3ZDQyNGFiNTQyMWJhNGI0MGY5NzY5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU4NmEwYTI3OGFmZjRlOWY4OGY1OGZhOTBlMWU1MDljLmJpbmRQb3B1cChwb3B1cF8wMWNhYjViZDIxNTQ0MTFjOWFhZDQ2ZjAxMGQ0ZTcwMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84NzE3ZWJiODA2MzU0MWExOTdlNjA1ZjhjNmRhNjUwZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ5LjczODE5MTMsLTEyMy4xMDA0MTIxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmZmMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZmZjAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81NDYwMTFlMmY4N2M0MGQzYTM3NDNhZWFhOWZjM2MyOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80YjExZWFhMDhmMDM0ZGIzOWE2Zjc1NDY3OGRkNTRlMyA9ICQoJzxkaXYgaWQ9Imh0bWxfNGIxMWVhYTA4ZjAzNGRiMzlhNmY3NTQ2NzhkZDU0ZTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlF1ZXN0IFVuaXZlcnNpdHksIENsdXN0ZXIgNTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTQ2MDExZTJmODdjNDBkM2EzNzQzYWVhYTlmYzNjMjguc2V0Q29udGVudChodG1sXzRiMTFlYWEwOGYwMzRkYjM5YTZmNzU0Njc4ZGQ1NGUzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg3MTdlYmI4MDYzNTQxYTE5N2U2MDVmOGM2ZGE2NTBmLmJpbmRQb3B1cChwb3B1cF81NDYwMTFlMmY4N2M0MGQzYTM3NDNhZWFhOWZjM2MyOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yOGI4MTI5NGQxYWE0NjhkOGEwNjIyZjlmNDgwODBlYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjEzNjk4MTYsLTc5LjAzNDk5NTVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMzMwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjMzMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q3YWRmMDljNDY4NzQ4YTE4ZWI4MDE2ZGQwZDA4YmJjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzExM2EwMWRjOTVmZjRmYmJiZmFhYTVkMDk4NjI0MzI4ID0gJCgnPGRpdiBpZD0iaHRtbF8xMTNhMDFkYzk1ZmY0ZmJiYmZhYWE1ZDA5ODYyNDMyOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TmlhZ2FyYSBVbml2ZXJzaXR5IChicmFuY2gpLCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q3YWRmMDljNDY4NzQ4YTE4ZWI4MDE2ZGQwZDA4YmJjLnNldENvbnRlbnQoaHRtbF8xMTNhMDFkYzk1ZmY0ZmJiYmZhYWE1ZDA5ODYyNDMyOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yOGI4MTI5NGQxYWE0NjhkOGEwNjIyZjlmNDgwODBlYS5iaW5kUG9wdXAocG9wdXBfZDdhZGYwOWM0Njg3NDhhMThlYjgwMTZkZDBkMDhiYmMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzkxNWU4YzBjMWFhNDlmMGIzODQ4MzdhMTU4MDRkNDYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OS4xNDA2OTU2OTk5OTk5OSwtMTIyLjYwMTk5MTJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmY2MwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmNjMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzEyNTA5MzdmNTJlMjQ4NmJiNDc0N2FkNWIwZTBhNjc0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc2YTI3OTYxZjVkMDQ5YmFhZWEyYzBiMjdkZDYxNjU3ID0gJCgnPGRpdiBpZD0iaHRtbF83NmEyNzk2MWY1ZDA0OWJhYWVhMmMwYjI3ZGQ2MTY1NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VHJpbml0eSBXZXN0ZXJuIFVuaXZlcnNpdHksIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTI1MDkzN2Y1MmUyNDg2YmI0NzQ3YWQ1YjBlMGE2NzQuc2V0Q29udGVudChodG1sXzc2YTI3OTYxZjVkMDQ5YmFhZWEyYzBiMjdkZDYxNjU3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzc5MTVlOGMwYzFhYTQ5ZjBiMzg0ODM3YTE1ODA0ZDQ2LmJpbmRQb3B1cChwb3B1cF8xMjUwOTM3ZjUyZTI0ODZiYjQ3NDdhZDViMGUwYTY3NCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85MTU0NjhhY2FjNzg0NWE2OGU3M2IxYWQyMmJhNWI5OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ5LjI4NDIzNzgsLTEyMy4xMTQ0MjU1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjMzMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYzMzAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lYWFkMmVmMDgxOGE0ZjhjYmQ2YzljZTRkNGZiZmZkMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mYTEzNzZlZjIwZGM0OWMzYjM0NTNkNjZlYmRkOGFjYiA9ICQoJzxkaXYgaWQ9Imh0bWxfZmExMzc2ZWYyMGRjNDljM2IzNDUzZDY2ZWJkZDhhY2IiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdHkgQ2FuYWRhIFdlc3QsIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZWFhZDJlZjA4MThhNGY4Y2JkNmM5Y2U0ZDRmYmZmZDAuc2V0Q29udGVudChodG1sX2ZhMTM3NmVmMjBkYzQ5YzNiMzQ1M2Q2NmViZGQ4YWNiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzkxNTQ2OGFjYWM3ODQ1YTY4ZTczYjFhZDIyYmE1Yjk5LmJpbmRQb3B1cChwb3B1cF9lYWFkMmVmMDgxOGE0ZjhjYmQ2YzljZTRkNGZiZmZkMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84MDUxMzQ3MTI4ZGU0MWFiYTAwZDgwNzg5ZWNmODNiYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ5Ljg5MjkyNjcsLTk3LjE1MTUzNjRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMzMwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjMzMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzFjZTQ2YzAyZGQ4NzRlZTA4ZWE2OWQ5Njc5ZmNjN2VjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzBiMmJhY2Y2MzAwYzQxYzE4MDg0NTExZTdmZTU4NTNjID0gJCgnPGRpdiBpZD0iaHRtbF8wYjJiYWNmNjMwMGM0MWMxODA4NDUxMWU3ZmU1ODUzYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Qm9vdGggVW5pdmVyc2l0eSBDb2xsZWdlLCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFjZTQ2YzAyZGQ4NzRlZTA4ZWE2OWQ5Njc5ZmNjN2VjLnNldENvbnRlbnQoaHRtbF8wYjJiYWNmNjMwMGM0MWMxODA4NDUxMWU3ZmU1ODUzYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84MDUxMzQ3MTI4ZGU0MWFiYTAwZDgwNzg5ZWNmODNiYS5iaW5kUG9wdXAocG9wdXBfMWNlNDZjMDJkZDg3NGVlMDhlYTY5ZDk2NzlmY2M3ZWMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWFiMjdhODdmMjExNDI3MGE1YzY5ODdkMTZmYjAwZTIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OS44NjAwMTM1OTk5OTk5OSwtOTcuMjMyMTc5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmY2NjAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmNjYwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMGU2NjJlN2VmZDI0NGM3NmFjNDY4ZWQ2OTA5ZmFmZTkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDdiMjM3YTNjYWIxNGI4Mjk0OGFkNTk5NGM0NmE3MmIgPSAkKCc8ZGl2IGlkPSJodG1sXzQ3YjIzN2EzY2FiMTRiODI5NDhhZDU5OTRjNDZhNzJiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DYW5hZGlhbiBNZW5ub25pdGUgVW5pdmVyc2l0eSwgQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wZTY2MmU3ZWZkMjQ0Yzc2YWM0NjhlZDY5MDlmYWZlOS5zZXRDb250ZW50KGh0bWxfNDdiMjM3YTNjYWIxNGI4Mjk0OGFkNTk5NGM0NmE3MmIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMWFiMjdhODdmMjExNDI3MGE1YzY5ODdkMTZmYjAwZTIuYmluZFBvcHVwKHBvcHVwXzBlNjYyZTdlZmQyNDRjNzZhYzQ2OGVkNjkwOWZhZmU5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc0ZjhiNWU4NTdhZDQwNzliNGMyYjNlMDMxZmIxMDRjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDUuNzI1MzU3MiwtNjUuNTI0MTI4NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmY5OTAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmOTkwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWFmZTc5MDhlNGJhNGVkZWE3ODQ3NTMzNTJjZGM0ZDYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTFkNzgzNDViMjkyNDM1MTgwYTM3MTlmMWNjMmY3OWIgPSAkKCc8ZGl2IGlkPSJodG1sXzUxZDc4MzQ1YjI5MjQzNTE4MGEzNzE5ZjFjYzJmNzliIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LaW5nc3dvb2QgVW5pdmVyc2l0eSwgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lYWZlNzkwOGU0YmE0ZWRlYTc4NDc1MzM1MmNkYzRkNi5zZXRDb250ZW50KGh0bWxfNTFkNzgzNDViMjkyNDM1MTgwYTM3MTlmMWNjMmY3OWIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzRmOGI1ZTg1N2FkNDA3OWI0YzJiM2UwMzFmYjEwNGMuYmluZFBvcHVwKHBvcHVwX2VhZmU3OTA4ZTRiYTRlZGVhNzg0NzUzMzUyY2RjNGQ2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJhYzYwNjc0MmVlNzRlZTc5YTg2ZTQwMWQwM2YxYmNmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDYuMTM0NjMyNiwtNjQuODYxMjMzNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmY5OTAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmOTkwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzhkYzUxMjBmMDg0NDBkYThkZWIwNjI0ZTYwZWE5NzYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2ZkNjg0ODA2OWRlNDBlNTkzNDQxODViYzFlYmJlMTcgPSAkKCc8ZGl2IGlkPSJodG1sXzdmZDY4NDgwNjlkZTQwZTU5MzQ0MTg1YmMxZWJiZTE3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DcmFuZGFsbCBVbml2ZXJzaXR5LCBDbHVzdGVyIDM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzc4ZGM1MTIwZjA4NDQwZGE4ZGViMDYyNGU2MGVhOTc2LnNldENvbnRlbnQoaHRtbF83ZmQ2ODQ4MDY5ZGU0MGU1OTM0NDE4NWJjMWViYmUxNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYWM2MDY3NDJlZTc0ZWU3OWE4NmU0MDFkMDNmMWJjZi5iaW5kUG9wdXAocG9wdXBfNzhkYzUxMjBmMDg0NDBkYThkZWIwNjI0ZTYwZWE5NzYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDQzZGNmMzMwOTQ0NDI0MTk5NWRmNjExZTk2YjhkOGIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0NS4xOTI5NTU3LC02Ny4yODE5MjAzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjk5MDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmY5OTAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83Mjk5YTU0M2FmYTI0MzNlOGZkNTkzNzk0OTUwNGZhMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83Mzc4ODAzOGM2Zjk0YmYyOGIzYTg1ZDFiNTdkMjMwZSA9ICQoJzxkaXYgaWQ9Imh0bWxfNzM3ODgwMzhjNmY5NGJmMjhiM2E4NWQxYjU3ZDIzMGUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBTdGVwaGVuIHMgVW5pdmVyc2l0eSwgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83Mjk5YTU0M2FmYTI0MzNlOGZkNTkzNzk0OTUwNGZhMy5zZXRDb250ZW50KGh0bWxfNzM3ODgwMzhjNmY5NGJmMjhiM2E4NWQxYjU3ZDIzMGUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDQzZGNmMzMwOTQ0NDI0MTk5NWRmNjExZTk2YjhkOGIuYmluZFBvcHVwKHBvcHVwXzcyOTlhNTQzYWZhMjQzM2U4ZmQ1OTM3OTQ5NTA0ZmEzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzM1MDVlZDI0MjljMTQzYWI4N2Q3ODM1MGU0MTIxMDRkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDUuOTQ1NTcwNCwtNjYuNjQwODI2NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfM2RkNDA3NjhlZTk3NGViNDk4NTRhMjU5ZDQ3NTZkYTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzg2OTlkMmE2OGI1NGE3ZTk5NzE0ZDk1ZTIyMzk4N2MgPSAkKCc8ZGl2IGlkPSJodG1sXzc4Njk5ZDJhNjhiNTRhN2U5OTcxNGQ5NWUyMjM5ODdjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIEZyZWRlcmljdG9uLCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzNkZDQwNzY4ZWU5NzRlYjQ5ODU0YTI1OWQ0NzU2ZGE4LnNldENvbnRlbnQoaHRtbF83ODY5OWQyYTY4YjU0YTdlOTk3MTRkOTVlMjIzOTg3Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zNTA1ZWQyNDI5YzE0M2FiODdkNzgzNTBlNDEyMTA0ZC5iaW5kUG9wdXAocG9wdXBfM2RkNDA3NjhlZTk3NGViNDk4NTRhMjU5ZDQ3NTZkYTgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGU4Y2UyZTQ0NTI5NDY3MTg5YjZmZmI5NDAyNjRmMTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0NC42MjY4MjY0LC02My41ODA1MDMzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmZmMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZmZjAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85OGZjZjJlZTRlOWM0YmI2YTU5MzYzY2EzMGM0ZGI5NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85OThmYjZkZTVmMjU0ZmYwOGE4NDFiNGM5MDcwZTVjZiA9ICQoJzxkaXYgaWQ9Imh0bWxfOTk4ZmI2ZGU1ZjI1NGZmMDhhODQxYjRjOTA3MGU1Y2YiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkF0bGFudGljIFNjaG9vbCBvZiBUaGVvbG9neSwgQ2x1c3RlciA1PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85OGZjZjJlZTRlOWM0YmI2YTU5MzYzY2EzMGM0ZGI5Ni5zZXRDb250ZW50KGh0bWxfOTk4ZmI2ZGU1ZjI1NGZmMDhhODQxYjRjOTA3MGU1Y2YpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGU4Y2UyZTQ0NTI5NDY3MTg5YjZmZmI5NDAyNjRmMTkuYmluZFBvcHVwKHBvcHVwXzk4ZmNmMmVlNGU5YzRiYjZhNTkzNjNjYTMwYzRkYjk2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FmOTY2Y2JkZTIwNDRhM2VhYTc5MjM4YjhmOTBhMjQ0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzk2ODUxMSwtNzkuMzkyMTg2NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZmZjAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmZmYwMCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDAuOCwKICAicmFkaXVzIjogMTIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDEKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjQ4MjJkZmE0OWE1NDhiZjhhYWViYjE5ZDhhMWJlY2UpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmZkYzk0YTQyNGFhNDM2MWJkOWI4MDZlZTY1MjIyMDMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjhkNzJkNjIzOGQ1NGM3MmI2MWVlY2E4OTA2Yjg3MzMgPSAkKCc8ZGl2IGlkPSJodG1sX2Y4ZDcyZDYyMzhkNTRjNzJiNjFlZWNhODkwNmI4NzMzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UeW5kYWxlIFVuaXZlcnNpdHksIENsdXN0ZXIgNTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZmZkYzk0YTQyNGFhNDM2MWJkOWI4MDZlZTY1MjIyMDMuc2V0Q29udGVudChodG1sX2Y4ZDcyZDYyMzhkNTRjNzJiNjFlZWNhODkwNmI4NzMzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FmOTY2Y2JkZTIwNDRhM2VhYTc5MjM4YjhmOTBhMjQ0LmJpbmRQb3B1cChwb3B1cF9mZmRjOTRhNDI0YWE0MzYxYmQ5YjgwNmVlNjUyMjIwMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kNjc2ODhjNjQ2YjQ0NjczYjJiNTE2YTRiNzhjZDUxMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjIwODY3NjksLTc5Ljk0OTE0MDM5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjk5MDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmY5OTAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMC44LAogICJyYWRpdXMiOiAxMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMQp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iNDgyMmRmYTQ5YTU0OGJmOGFhZWJiMTlkOGExYmVjZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zYTk2NTgzYjRiNWU0MWJkYTc4ZjEyZmZjYjAwYTQ3ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83Njg5OGQ0MTNkOTE0OGJmOTMxYTJhNjJiZjlhNGMyMSA9ICQoJzxkaXYgaWQ9Imh0bWxfNzY4OThkNDEzZDkxNDhiZjkzMWEyYTYyYmY5YTRjMjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJlZGVlbWVyIFVuaXZlcnNpdHkgQ29sbGVnZSwgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zYTk2NTgzYjRiNWU0MWJkYTc4ZjEyZmZjYjAwYTQ3ZC5zZXRDb250ZW50KGh0bWxfNzY4OThkNDEzZDkxNDhiZjkzMWEyYTYyYmY5YTRjMjEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDY3Njg4YzY0NmI0NDY3M2IyYjUxNmE0Yjc4Y2Q1MTIuYmluZFBvcHVwKHBvcHVwXzNhOTY1ODNiNGI1ZTQxYmRhNzhmMTJmZmNiMDBhNDdkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2YwMzcwYjg4YzgyMzRjZTg4MTQyYjc0MThkMTRkNjE1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTMuNTI1NDE3OSwtMTEzLjQxNjc1MjRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmY2MwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmNjMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAwLjgsCiAgInJhZGl1cyI6IDEyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAxCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I0ODIyZGZhNDlhNTQ4YmY4YWFlYmIxOWQ4YTFiZWNlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzhiYmZhYTE1ZDU1ZDQ4YjE5NjVmOGJiMjM5OWE2NTIxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E5NTdmY2ZmOGNmZDQyNDg5OTNkNGY3ZjM2MDBmMjgwID0gJCgnPGRpdiBpZD0iaHRtbF9hOTU3ZmNmZjhjZmQ0MjQ4OTkzZDRmN2YzNjAwZjI4MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEtpbmcgcyBVbml2ZXJzaXR5LCBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzhiYmZhYTE1ZDU1ZDQ4YjE5NjVmOGJiMjM5OWE2NTIxLnNldENvbnRlbnQoaHRtbF9hOTU3ZmNmZjhjZmQ0MjQ4OTkzZDRmN2YzNjAwZjI4MCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mMDM3MGI4OGM4MjM0Y2U4ODE0MmI3NDE4ZDE0ZDYxNS5iaW5kUG9wdXAocG9wdXBfOGJiZmFhMTVkNTVkNDhiMTk2NWY4YmIyMzk5YTY1MjEpOwoKICAgICAgICAgICAgCiAgICAgICAgCjwvc2NyaXB0Pg== onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python
univ_map.save("University Food Map.html")
```

Clearly, we cannot observe any clear geographic pattern of the clustering. Almost every province has universities from each label. This is consistent with what we concluded from the province distribution table that no apparent geographic pattern is observed. Therefore, our next step is to go back to our data frames and see which venue category is more common under each group.


```python
# Add labels to the recommendations in counts
df_recomm_count.insert(0,"Label",labels)
```


```python
# Group by labels
df_recomm_countsum = df_recomm_count.groupby(by=['Label'], axis=0).sum()

df_recomm_countsum
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Afghan Restaurant</th>
      <th>American Restaurant</th>
      <th>Asian Restaurant</th>
      <th>BBQ Joint</th>
      <th>Bagel Shop</th>
      <th>Bakery</th>
      <th>Belgian Restaurant</th>
      <th>Bistro</th>
      <th>Brazilian Restaurant</th>
      <th>Breakfast Spot</th>
      <th>...</th>
      <th>Sushi Restaurant</th>
      <th>Taco Place</th>
      <th>Tapas Restaurant</th>
      <th>Thai Restaurant</th>
      <th>Theme Restaurant</th>
      <th>Turkish Restaurant</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Vietnamese Restaurant</th>
      <th>Wings Joint</th>
      <th>nan</th>
    </tr>
    <tr>
      <th>Label</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>...</td>
      <td>8</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>...</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>6</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 68 columns</p>
</div>




```python
# Sort by the count of each category and print out

for label in df_recomm_countsum.index:
    df_recomm_sort = df_recomm_countsum.iloc[label,:].sort_values(ascending=False)
    print("-------- Label {} --------".format(label))
    print(df_recomm_sort.head(10))
```

    -------- Label 0 --------
    Café                             64
    Restaurant                       36
    Bakery                           13
    Gastropub                        12
    Vegetarian / Vegan Restaurant    11
    Pizza Place                      11
    French Restaurant                10
    Italian Restaurant                7
    Japanese Restaurant               6
    Breakfast Spot                    5
    Name: 0, dtype: uint8
    -------- Label 1 --------
    Restaurant                 29
    Sandwich Place             15
    Café                       13
    Pizza Place                10
    Breakfast Spot             10
    Sushi Restaurant            8
    Bakery                      8
    New American Restaurant     7
    Indian Restaurant           7
    Italian Restaurant          6
    Name: 1, dtype: uint8
    -------- Label 2 --------
    nan                            1
    Fast Food Restaurant           0
    Chinese Restaurant             0
    Creperie                       0
    Deli / Bodega                  0
    Diner                          0
    Donut Shop                     0
    Eastern European Restaurant    0
    Fish & Chips Shop              0
    Wings Joint                    0
    Name: 2, dtype: uint8
    -------- Label 3 --------
    Restaurant              33
    Fast Food Restaurant    22
    Pizza Place             16
    Sandwich Place          13
    Diner                    8
    Breakfast Spot           7
    Café                     6
    Wings Joint              5
    Greek Restaurant         4
    Gastropub                3
    Name: 3, dtype: uint8
    -------- Label 4 --------
    Sandwich Place          17
    Fast Food Restaurant    14
    Café                    11
    Pizza Place             10
    Sushi Restaurant         6
    Burger Joint             6
    Steakhouse               5
    American Restaurant      5
    Diner                    4
    Italian Restaurant       3
    Name: 4, dtype: uint8
    -------- Label 5 --------
    Burger Joint           14
    Pizza Place            13
    Sushi Restaurant       13
    Italian Restaurant     11
    Mexican Restaurant      9
    Breakfast Spot          9
    Turkish Restaurant      6
    Japanese Restaurant     6
    Bakery                  6
    Indian Restaurant       5
    Name: 5, dtype: uint8



```python
df_label_count = df_univ.groupby(by=['Label'], axis=0).count()
count_ls = df_label_count['Name'].to_list()

for label, count in zip(df_recomm_countsum.index, count_ls):
    df_recomm_sort = df_recomm_countsum.iloc[label,:].sort_values(ascending=False)
    df_recomm_avgcount = df_recomm_sort/count
    print("-------- Label {} --------".format(label))
    print(df_recomm_avgcount.head(10))
```

    -------- Label 0 --------
    Café                             2.666667
    Restaurant                       1.500000
    Bakery                           0.541667
    Gastropub                        0.500000
    Vegetarian / Vegan Restaurant    0.458333
    Pizza Place                      0.458333
    French Restaurant                0.416667
    Italian Restaurant               0.291667
    Japanese Restaurant              0.250000
    Breakfast Spot                   0.208333
    Name: 0, dtype: float64
    -------- Label 1 --------
    Restaurant                 1.380952
    Sandwich Place             0.714286
    Café                       0.619048
    Pizza Place                0.476190
    Breakfast Spot             0.476190
    Sushi Restaurant           0.380952
    Bakery                     0.380952
    New American Restaurant    0.333333
    Indian Restaurant          0.333333
    Italian Restaurant         0.285714
    Name: 1, dtype: float64
    -------- Label 2 --------
    nan                            1.0
    Fast Food Restaurant           0.0
    Chinese Restaurant             0.0
    Creperie                       0.0
    Deli / Bodega                  0.0
    Diner                          0.0
    Donut Shop                     0.0
    Eastern European Restaurant    0.0
    Fish & Chips Shop              0.0
    Wings Joint                    0.0
    Name: 2, dtype: float64
    -------- Label 3 --------
    Restaurant              2.0625
    Fast Food Restaurant    1.3750
    Pizza Place             1.0000
    Sandwich Place          0.8125
    Diner                   0.5000
    Breakfast Spot          0.4375
    Café                    0.3750
    Wings Joint             0.3125
    Greek Restaurant        0.2500
    Gastropub               0.1875
    Name: 3, dtype: float64
    -------- Label 4 --------
    Sandwich Place          1.416667
    Fast Food Restaurant    1.166667
    Café                    0.916667
    Pizza Place             0.833333
    Sushi Restaurant        0.500000
    Burger Joint            0.500000
    Steakhouse              0.416667
    American Restaurant     0.416667
    Diner                   0.333333
    Italian Restaurant      0.250000
    Name: 4, dtype: float64
    -------- Label 5 --------
    Burger Joint           0.8750
    Pizza Place            0.8125
    Sushi Restaurant       0.8125
    Italian Restaurant     0.6875
    Mexican Restaurant     0.5625
    Breakfast Spot         0.5625
    Turkish Restaurant     0.3750
    Japanese Restaurant    0.3750
    Bakery                 0.3750
    Indian Restaurant      0.3125
    Name: 5, dtype: float64


## PART 5 DISCUSSION

We summarize the features of each group as follows.

* Label 0

Café dominates the list of the top 10 venue categories of this group. There are 64 cafés recommended in total and an average of 2.67 cafes near each university in this group. These two numbers are much higher than those of the rest of the groups. So the percentage of café must be an important factor that K-Means used when deciding which university should be assigned to cluster 0.

This group is also the only one that has Vegan Restaurant on the list. Good news for vegetarians. In addition, no fast-food was recommended. So basically this is a very healthy group.

The foreign restaurants recommended and are on the list include French Restaurant, Japanese Restaurant and Italian Restaurant. 

The universities in this group include University of British Columbia, McGill University, Université de Montréal and other 21 universities.

* Label 1

Restaurant is the top one on the list with an absolute advantage of 29 in total. Since we can infer nothing from it, we skip this category. Followed Restaurant are Sandwich Place and Café. Except that, the numbers of the rest of the categories in the list are quite even . 

The foreign restaurants recommended and are on the list include Sushi Restaurant, New American Restaurant, Indian Restaurant and Italian Restaurant. 

The universities in this group include University of Manitoba, University of Waterloo, York University and other 18 universities.

* Label 2

Only Canadian Mennonite University was classified to this group with no recommendations available.

* Label 3

The top one on the list of this group is still Restaurant, which tells us nothing. Other than that, this group got plenty of fast-food restaurants and nearly no foreign restaurants recommended. There are on average 1.4 fast-food restaurants, 0.8 sandwich place and 0.3 wings joint. In total, there are about 40 venues in these kinds being recommended. 

Greek Restaurant is the only foreign restaurants recommended and are on the list.

The universities in this group include University of Calgary, Université de Sherbrooke, and 14 other universities.

* Label 4

Only this group, group 2 and group 5 that Restaurant is not on the list of their top 10 common categories.

Similar to group 3, the universities in this group got a lot of fast-food recommendations. The difference is that Fast-Food Restaurant is the most common one in group 3 while in group 4 is Sandwich Place. There are on average 1.4 sandwich place, 1.2 fast-food restaurants and 0.5 Burger Joint. In total, there are about 37 venues in these kinds being recommended.

The foreign restaurants recommended and are on the list include Sushi Restaurant, American Restaurant and Italian Restaurant. 

The universities in this group include University of Alberta, Simon Fraser University and 10 other universities.

* Label 5

What is special about this group is that it has the most diverse food supplier recommendations. The foreign restaurants recommended and are on the list include Sushi Restaurant, Italian Restaurant, Mexican Restaurant, Turkish Restaurant, Japanese Restaurant and Indian Restaurant, six in total.

Another feature of this group is that no one category can get an average number over one. Some actually there is not a venue category that can dominate the list of this group.

The universities in this group include Dalhousie University, McMaster University and 14 other universities.


## PART 6 CONCLUSION

Overall, it seems that finding delicious foods around the universities does not constitute a serious problem for students there. The surrounding of almost every university offers enough options to be considered. 

Based on the result of our clustering, we could also see that the most common venue categories of each group are quite different. In general, venues that are classified into restaurants, fast-food restaurants, pizza places and cafés are most easily to be found nearby.  

In addition, there are many foreign restaurants in the neighbourhood. Therefore, international students and domestic students who would like to have something different can all find good places to go.

There are some categories we did not mention in the last section like gastropub, bakery and breakfast spots are also great options. 

So in summary, we hope this project could help you understand what the food map is like around the universities, and may inspire you on some similar or any other interesting ideas.

One kind warning is that the features of the groups are summarized based on groups in aggregate. Depending on the performance of the clustering algorithm, the individual university may not completely share the features of its group.
