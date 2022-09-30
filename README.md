# Modelling Airbnb's Property Listing Dataset

Creating a multimodal deep learning structure that processes text, images and tabular data to predict the type of property from an Airbnb listing.


## Data preparation



> A Tableau dashboard showing initial insights into the data.

The 'structured' data is downloaded as a `.csv` file from an AWS S3 bucket containing the following information:
- `ID`: Unique identifier for the listing
- `Category`: The category of the listing
- `Title`: The title of the listing
- `Description`: The description of the listing
- `Amenities`: The available amenities of the listing
- `Location`: The location of the listing
- `Guests`: The number of guests that can be accommodated in the listing
- `Beds`: The number of available beds in the listing
- `Bathrooms`: The number of bathrooms in the listing
- `Price_Night`: The price per night of the listing
- `Cleanliness_rate`: The cleanliness rating of the listing
- `Accuracy_rate`: How accurate the description of the listing is, as reported by previous guests
- `Location_rate`: The rating of the location of the listing
- `Check-in_rate`: The rating of check-in process given by the host
- `Value_rate`: The rating of value given by the host
- `Amenities_count`: The number of amenities in the listing
- `URL`: The URL of the listing
- `Bedrooms`: The number of bedrooms in the listing

The file `tabular_data.py` contains the function `clean_data()` which will take a DataFrame as an input, and return a cleaned DataFrame. This is saved in the `dataframes` directory as `cleaned_dataset.csv`. A Tableau dashboard is created to take an overview at the data provided.

The function `create_numerical_dataset()` will drop all columns containing non-numerical data and `load_airbnb()` will return the tuple `(features, labels)` ready to train a regression model in the next section.


## Creating a regression model




> A comparison of the RMSE of each model calculated from the validation set.


## Creating a classification model


```python
label_encoder = LabelEncoder().fit(y)
label_encoded_y = label_encoder.transform(y)
```

> Encoding the label space using `LabelEncoder`.


> A comparison of the accuracy score of each model calculated from the validation set.


## Creating an FeedForward Artificial Neural Network



### Creating the DataLoader



### Creating the network architecture


### Instantiating the model



### Setting model parameters



### Training the model



> The training loop.

### Tuning the parameters

