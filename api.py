import src.reqhandler_v2 as rh
from src.utils import s3_load_object
import os
from obscene_words_filter import conf

modelversion = 'nps_v1/'

s3_load_object(f'kiosk/models/{modelversion}bert.pt', './model/bert.pt')
s3_load_object(f'kiosk/models/{modelversion}bert2.pt', './model/bert2.pt')

s3_load_object(f'kiosk/models/{modelversion}coupons.xlsx', './obtain/coupons.xlsx')

s3_load_object(f'kiosk/models/{modelversion}t4m_2022_04_20__14_31.xlsx', './scrub/t4m_2022_04_20__14_31.xlsx')
s3_load_object(f'kiosk/models/{modelversion}t4m_2022_05_05__21_59.xlsx', './scrub/t4m_2022_05_05__21_59.xlsx')

ANALYTICS_DB_URL = os.getenv('POSTGRES_CONN')


import socket
import sys
import json_logging
from fastapi import FastAPI, Response, Header, status, Request
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
from fastapi.responses import JSONResponse
app = FastAPI(root_path=os.getenv("ROOT_PATH"))

@app.post('/predict')
async def get_result(request: Request):
    data_for_analysis = await request.json()
    if data_for_analysis == {'run':'run'}:
        print('renew classification!')
        try:
            res = rh.classify_new_comments(ANALYTICS_DB_URL)
            return JSONResponse(status_code=200, content={"error": "no errors"})
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
    else:
        return JSONResponse(status_code=400, content={"error": 'awaiting command'})

if __name__ == "__main__":

    print('started ok!')
    # res = rh.classify_new_comments(ANALYTICS_DB_URL)
    uvicorn.run(app, host="0.0.0.0", port=5000)
