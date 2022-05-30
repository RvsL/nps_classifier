# nps_negative_classification

docker-compose repo to assign class to recent negative app comments, and put result in a table

you execute

docker-compose build
docker-compose up

then you get a service running on port 5000. And you can access it from a jupyter notebook like this:

```
import numpy as np
import pandas as pd
import requests
import datetime
import json

headers = {'Content-Type': 'application/json'}
url = 'http://127.0.0.1:5000/predict'
test = {'run':'run'}

request = requests.post(url=url, headers=headers, json=test)

print(request.content)
```
