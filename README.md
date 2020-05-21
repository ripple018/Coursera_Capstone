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

As discussed in the Data section, we need the tables on a Wikipedia page. The function of Pandas read_html makes this process quite easy. There are 14 tables on that page, and what we need is the first 11 tables, 10 for public universities and one for private universities.

We will not perform separate operations on public universities and private universities, so it is better to combine these two tables before cleaning. It is clear that the two tables have different formats. The one with public universities has multiple indexes, so the names of the columns are not complete. We refer to the private universities’ table to adjust the format of the first table.

Furthermore, we will not use all the columns in the table. We drop the unnecessary columns and leave Name, City and Province as a result. A complete list of universities is as follows. It contains 91 records in total. Below display the first 5 rows for illustration.

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

At this stage, we are ready to retrieve the coordinates of the universities. We will send requests to the address where the copy of the previous Google Geocoding API is stored, sending the API key and the addresses of the universities as parameters. 

In this process, we got the warning that we failed to retrieve the coordinate of Trent University. So we remove any records that contain at least a NA value. The result after this step is as follows. Now we have only 90 records in total.

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

#### 3.1.3. Recommendations of Food around the Universities

After obtaining the coordinates of the universities, we use Foursquare’s API to get the recommendations in the food section.

Below is the result we got. There are 856 records in total, with the venue category we care most in the last column of the table.

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

The final step of our data preprocessing is to calculate the percentages we needed for K-Means to process.

We build two tables via calculations done on the table we got in the last step, one displays the counts of the recommended venues and one displays the percentages. Other information will all be dropped in this process but only be available in the last table for reference only.

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

# We set the number of groups to be 6, and set the random seed to be 0 to avoid the result changes everytime we rerun the program
n_clusters = 6

km = KMeans(n_clusters = n_clusters, init='k-means++', random_state = 0)
```

The result of the model is as follows. Here we only display the first 5 universities to illustrate. The labels returned by the model are in the last column.

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

## Pending Maps

Clearly, we cannot observe any clear geographic pattern of the clustering. Almost every province has universities from each label. This is consistent with what we concluded from the province distribution table that no apparent geographic pattern is observed. Therefore, our next step is to go back to our data frames and see which venue category is more common under each group.

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
