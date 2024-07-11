import os
import time
import pandas as pd
import zipfile
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib

# --- Configuration ---
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY') or 'your_pine_cone_api_key_here'
PINECONE_CLOUD = os.environ.get('PINECONE_CLOUD') or 'aws'
PINECONE_REGION = os.environ.get('PINECONE_REGION') or 'us-east-1'
INDEX_NAME = 'product-recommender'
BATCH_SIZE = 100
MODEL_FILE = 'recommendations.pkl'

model=""
index=""
data = ""
products_lookup="" 
user_to_index=""

# --- Data Loading ---
def load_data():
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

    order_products_train = pd.read_csv('order_products__train.csv')
    order_products_prior = pd.read_csv('order_products__prior.csv')
    products = pd.read_csv('products.csv')
    orders = pd.read_csv('orders.csv')

    order_products = pd.concat([order_products_train, order_products_prior])
    customer_order_products = pd.merge(orders, order_products, how='inner', on='order_id')

    data = customer_order_products.groupby(['user_id', 'product_id'])[['order_id']].count().reset_index()
    data.columns = ["user_id", "product_id", "total_orders"]
    data.product_id = data.product_id.astype('int64')

    products_lookup = products[['product_id', 'product_name']].drop_duplicates()
    products_lookup['product_id'] = products_lookup.product_id.astype('int64')

    data_new = pd.DataFrame([[data.user_id.max() + 1, 22802, 97],
                             [data.user_id.max() + 2, 26834, 89],
                             [data.user_id.max() + 2, 12590, 77]
                            ], columns=['user_id', 'product_id', 'total_orders'])

    data = pd.concat([data, data_new]).reset_index(drop=True)

    users = list(np.sort(data.user_id.unique()))

    index_to_user = pd.Series(users)
    user_to_index = pd.Series(data=index_to_user.index + 1, index=index_to_user.values)

    return data, products_lookup, user_to_index

# --- Model Loading ---
def load_model():
    return joblib.load(MODEL_FILE)

# --- Pinecone Setup ---
def setup_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            INDEX_NAME,
            dimension=100,
            metric='cosine',
            spec=spec
        )
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)

    return pc.Index(INDEX_NAME)


# --- Past Purchases ---
def products_bought_by_user_in_the_past(user_id: int, data, products_lookup, top: int = 10):
    selected = data[data.user_id == user_id].sort_values(by=['total_orders'], ascending=False)
    selected['product_name'] = selected['product_id'].map(products_lookup.set_index('product_id')['product_name'])
    selected = selected[['product_id', 'product_name', 'total_orders']].reset_index(drop=True)
    if selected.shape[0] < top:
        return selected
    return selected[:top]

# --- FastAPI Setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- API Endpoints ---
@app.get("/users/{user_id}")
def get_recommendations(user_id: int):
    user_factors = model.user_factors[user_to_index[[user_id]]]
    user_embeddings = user_factors[0].tolist()

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

@app.get("/users/{user_id}/past_purchases")
def get_past_purchases(user_id: int):
    past_purchases = products_bought_by_user_in_the_past(user_id, data, products_lookup, top=15)
    return past_purchases.to_dict(orient='records')

# --- Main Execution ---
if __name__ == "__main__":
    data, products_lookup, user_to_index = load_data()
    model = load_model()
    index = setup_pinecone()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
