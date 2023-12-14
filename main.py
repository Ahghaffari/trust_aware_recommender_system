from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd
import numpy as np
from custom_rs import CollabTrustRecSys
import random

max_neighbour_users = 40
model_type_selection = "1"

def collaborative_filtering(train_data, test_data, trust_data, k, min_k, model_type):
    sim_options = {'name': 'pearson', 'user_based': True}
    model = CollabTrustRecSys(k=k, model_type=model_type, sim_options=sim_options, min_k=min_k)
    model.fit(train_data, trust_data=trust_data, nu_u=k)
    predictions = model.test(test_data)
    
    return predictions


def train_test_split_each_user(data, train_size_percentage=0.8, random_state=42):
    random.seed(random_state)
    temp = 1
    last_data = []
    selected_elements_train = []
    selected_elements_test = []
    for indx, row in data.iterrows():
        if temp == row["user_id"] and indx<=(len(data)-1):
            last_data.append(row)
        else:
            num_elements_to_select = round(train_size_percentage * len(last_data))

            random_indexes = random.sample(range(len(last_data)), num_elements_to_select)

            for i, item in enumerate(last_data):
                if i in random_indexes:
                    selected_elements_train.append(item) 
                else:
                    selected_elements_test.append(item)
            last_data = []
            temp = row["user_id"]

    return selected_elements_train, selected_elements_test


# Load ratings data
ratings_data = pd.read_csv("ratings.txt", sep=" ", header=None, names=["user_id", "item_id", "rating"])

# Load trust data
trust_data = pd.read_csv("trust.txt", sep=" ", header=None, names=["trustor", "trustee", "trust_value"])

Number_of_users = 1508

# # Use Surprise library to load and preprocess the dataset
# reader = Reader(rating_scale=(0.5, 4.0))
# data = Dataset.load_from_df(ratings_data, reader)

# # Split the data into training and test sets using Surprise
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_data, test_data = train_test_split_each_user(ratings_data, train_size_percentage=0.8, random_state=42)
train_data_df = pd.concat(train_data, axis=1).T
test_data_df = pd.concat(test_data, axis=1).T

reader = Reader(rating_scale=(0.5, 4.0))
train_data = Dataset.load_from_df(train_data_df, reader).build_full_trainset()
test_data = Dataset.load_from_df(test_data_df, reader).build_full_trainset().build_testset()

predictions = collaborative_filtering(train_data=train_data, test_data=test_data, trust_data=trust_data, k=max_neighbour_users, min_k=1, model_type=model_type_selection)

rating_impossible = 0
for i in predictions[:]:
    if not i.details["was_impossible"]:
        rating_impossible += 1

# Calculate MAE, RMSE, and RC metrics
mae = accuracy.mae(predictions)
rmse = accuracy.rmse(predictions)
rate_coverage = rating_impossible / len(predictions)

# Display the evaluation metrics
print(f"\nmodel  = {model_type_selection} Evaluation metrics with K={max_neighbour_users}:")
print("MAE:", mae)
print("RMSE:", rmse)
print("Rate Coverage:", rate_coverage)
print(100*"#")

