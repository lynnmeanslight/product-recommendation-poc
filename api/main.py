import os
from pinecone import Pinecone
import torch
import time
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import zipfile
import implicit
from implicit import evaluation
from tqdm.auto import tqdm
from pinecone import ServerlessSpec
from typing import Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
# Add CORSMiddleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define environment variables for Pinecone
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY') or 'your_pine_cone_api_key_here'
PINECONE_CLOUD = os.environ.get('PINECONE_CLOUD') or 'aws'
PINECONE_REGION = os.environ.get('PINECONE_REGION') or 'us-east-1'

# Define index name
INDEX_NAME = 'product-recommender'

# Define batch size for upserting vectors
BATCH_SIZE = 100

# Load data from zip files
files = [
    'instacart-market-basket-analysis.zip',
    'order_products__train.csv.zip',
    'order_products__prior.csv.zip',
    'products.csv.zip',
    'orders.csv.zip'
]

for filename in files:
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('./')

# Load data into pandas DataFrames
order_products_train = pd.read_csv('order_products__train.csv')
order_products_prior = pd.read_csv('order_products__prior.csv')
products = pd.read_csv('products.csv')
orders = pd.read_csv('orders.csv')

# Combine order products data
order_products = pd.concat([order_products_train, order_products_prior])

# Merge orders and order products data
customer_order_products = pd.merge(orders, order_products, how='inner', on='order_id')

# Create a table with "confidences"
data = customer_order_products.groupby(['user_id', 'product_id'])[['order_id']].count().reset_index()
data.columns = ["user_id", "product_id", "total_orders"]
data.product_id = data.product_id.astype('int64')

# Create a lookup frame for product names
products_lookup = products[['product_id', 'product_name']].drop_duplicates()
products_lookup['product_id'] = products_lookup.product_id.astype('int64')

# Add some test data
data_new = pd.DataFrame([[data.user_id.max() + 1, 22802, 97],
                         [data.user_id.max() + 2, 26834, 89],
                         [data.user_id.max() + 2, 12590, 77]
                        ], columns=['user_id', 'product_id', 'total_orders'])

data = pd.concat([data, data_new]).reset_index(drop=True)

# Create mappings for users and items
users = list(np.sort(data.user_id.unique()))
items = list(np.sort(products.product_id.unique()))
purchases = list(data.total_orders)

index_to_user = pd.Series(users)
user_to_index = pd.Series(data=index_to_user.index + 1, index=index_to_user.values)

index_to_item = pd.Series(items)
item_to_index = pd.Series(data=index_to_item.index, index=index_to_item.values)

# Create sparse matrices for user-product interactions
products_rows = data.product_id.astype(int)
users_cols = data.user_id.astype(int)

sparse_product_user = sparse.csr_matrix((purchases, (products_rows, users_cols)), shape=(len(items) + 1, len(users) + 1))
sparse_product_user.data = np.nan_to_num(sparse_product_user.data, copy=False)

sparse_user_product = sparse.csr_matrix((purchases, (users_cols, products_rows)), shape=(len(users) + 1, len(items) + 1))
sparse_user_product.data = np.nan_to_num(sparse_user_product.data, copy=False)

# Split data into train and test sets
train_set, test_set = evaluation.train_test_split(sparse_user_product, train_percentage=0.9)

# Initialize and train the ALS model
model = implicit.als.AlternatingLeastSquares(factors=100,
                                             regularization=0.05,
                                             iterations=50,
                                             num_threads=1)

alpha_val = 15
train_set = (train_set * alpha_val).astype('double')
model.fit(train_set, show_progress=True)

test_set = (test_set * alpha_val).astype('double')
evaluation.ranking_metrics_at_k(model, train_set, test_set, K=100,
                         show_progress=True, num_threads=1)

# Initialize Pinecone connection
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define serverless index specification
spec = ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)

# Create Pinecone index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        INDEX_NAME,
        dimension=100,
        metric='cosine',
        spec=spec
    )
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)

# Connect to the index
index = pc.Index(INDEX_NAME)

# Prepare data for upserting into Pinecone
all_items_titles = [{'title': title} for title in products_lookup['product_name']]
all_items_ids = [str(product_id) for product_id in products_lookup['product_id']]

items_factors = model.item_factors
device = "cuda" if torch.cuda.is_available() else "cpu"
item_embeddings = items_factors[1:].to_numpy().tolist() if device == "cuda" else items_factors[1:].tolist()

items_to_insert = list(zip(all_items_ids, item_embeddings, all_items_titles))

# Upsert items into Pinecone index
print('Index statistics before upsert:', index.describe_index_stats())

for i in tqdm(range(0, len(items_to_insert), BATCH_SIZE)):
    index.upsert(vectors=items_to_insert[i:i+BATCH_SIZE], async_req=True)

print('Index statistics after upsert:', index.describe_index_stats())

# Helper function to get past purchases
def products_bought_by_user_in_the_past(user_id: int, top: int = 10):
    selected = data[data.user_id == user_id].sort_values(by=['total_orders'], ascending=False)
    selected['product_name'] = selected['product_id'].map(products_lookup.set_index('product_id')['product_name'])
    selected = selected[['product_id', 'product_name', 'total_orders']].reset_index(drop=True)
    if selected.shape[0] < top:
        return selected
    return selected[:top]

# Define FastAPI endpoint for retrieving recommendations
@app.get("/users/{user_id}")
def get_recommendations(user_id: int):
    user_factors = model.user_factors[user_to_index[[user_id]]]
    user_embeddings = user_factors.to_numpy()[0].tolist() if device == "cuda" else user_factors[0].tolist()

    start_time = time.process_time()
    query_results = index.query(vector=user_embeddings, top_k=10, include_metadata=True)
    retrieving_time = time.process_time() - start_time

    response_data = [
        {
            'id': match['id'],
            'name': match["metadata"]['title'],
            'scores': match["score"]
        } for match in query_results["matches"]
    ]

    final_response = {
        'retrieving_time': retrieving_time,
        'body': response_data,
        'user_id': user_id
    }

    return final_response

# Define FastAPI endpoint for getting past purchases
@app.get("/users/{user_id}/past_purchases")
def get_past_purchases(user_id: int):
    past_purchases = products_bought_by_user_in_the_past(user_id, top=15)
    return past_purchases.to_dict(orient='records')

# # Run the FastAPI server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
