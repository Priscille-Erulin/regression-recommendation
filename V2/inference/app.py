import json
import pandas as pd
import streamlit as st
import redis
import requests
import asyncio

import inference_v2 as inf

REDIS_CLIENT = redis.Redis(host='localhost', port=6379)

def get_live_campaigns():
    url = 'https://api.appchoose.io/app/sales/live'
    response = requests.get(url)
    sales = response.json()['sales']
    return pd.DataFrame(sales)

campaigns = get_live_campaigns()

# def get_predictions(user_id: str, live_campaigns: list[str]):
#     url = 
#     request_body = {
#     "sale_ids": live_campaigns
#     }
#     response = requests.post(f'{url}/api/v1/sales', json=request_body)

async def get_user_recommendations(user_id: str, live_campaigns: list[str]):
    # log.info('Started personalised sale feature flow')
    # redis_key = f'reco:sales:{user_id}'
    # reco = json.loads(REDIS_CLIENT.get(redis_key))
    # return reco

    engine = inf.PersonalisedEngine(
    redis_host='localhost',
    redis_port=6379)
    return await engine.recommend(uid=user_id, ongoing=live_campaigns)

def make_campaign_card(sale: dict):
    st.image(resize_image(sale['image_header']), width=200)
    category = sale['braze_category']
    st.write(f"{sale['name']} - {category}")

def resize_image(url):
    return url.replace("[W]x[H]", "")

### STREAMLIT APP ###

st.title('Recommendation')

user_id = st.text_input('Enter user id', value="gMjTJgn7LFeWPMbn2m9u3yBzFTY2")

if user_id:
    live_sale_ids = campaigns['key'].to_list()
    predictions = asyncio.run(get_user_recommendations(user_id, live_sale_ids))
    top = pd.DataFrame(predictions.top, columns=["key"])
    bottom = pd.DataFrame(predictions.bottom, columns=["key"])
    top = top.merge(campaigns, how="inner", on="key")
    bottom = bottom.merge(campaigns, how="inner", on="key")

    st.write('<span style="font-size:35px">Nouvelles ventes</span>', unsafe_allow_html=True)
    for sale in top.to_dict(orient='records'):
        make_campaign_card(sale)
    st.write('<span style="font-size:35px">Toujours Disponible</span>', unsafe_allow_html=True)
    for sale in bottom.to_dict(orient='records'):
        make_campaign_card(sale)
