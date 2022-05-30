import os, boto3
from sqlalchemy import create_engine
import pandas as pd
import numpy as np


def db_get_con_sqlalchemy(db, echo=False, database='BurgerKing_Installation_Final'):
    if db == 'analytics':
        try:
            cn = sa.create_engine(ANALYTICS_DB_URL, echo=echo)
        except Exception as e:
            logger.exception(f'Error connecting to app database: {e}')
        return cn
    if db == 'app':
        try:
            cn = sa.create_engine(DB_133_URL, echo=echo)
        except Exception as e:
            logger.exception(f'Error connecting to app database: {e}')
        return cn
    if db in ('db21', 'db17'):
        if db == 'db17':
            server = 'Msk-v-db17'
        else:
            server = DB_21_HOST_PORT
        database = database
        driver = DB_21_DRIVE
        trusted_con = 'no'

        if trusted_con == 'yes':
            con_string = f"mssql+pyodbc://{server}/{database}?driver={'+'.join(driver.split(' '))};"
        else:
            uid = DB_21_USER_NAME
            pwd = DB_21_PASSWORD
            con_string = f"mssql+pyodbc://{uid}:{pwd}@{server}/{database}?driver={'+'.join(driver.split(' '))}"

        try:
            cn = sa.create_engine(con_string, fast_executemany=True, echo=echo)
        except Exception as e:
            logger.exception(f'Error connecting to database {database}: {e}')
        return cn

def execute_query(query, filename, conn):

    try:
        engine = create_engine(conn)

        with engine.connect() as connection:
            result = pd.read_sql(query, connection)

        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(result, f)
    except:
        try:
            with open(f'{filename}.pkl', 'rb') as f:
                result = pickle.load(f)
        except:
            if conn.startswith('postgresql'):
                message = 'postgresql database connection problem'
            elif conn.startswith('mssql'):
                message = 'mssql database connection problem'
            else:
                message = 'database connection problem'
            raise Exception(message)

    return result


def s3_load_object(path, saveas):

    S3_ENDPOINT = os.getenv('S3_ENDPOINT')
    S3_KEYID = os.getenv('S3_KEYID')
    S3_ACCESSKEY = os.getenv('S3_ACCESSKEY')
    BUCKET_NAME = 'recsys-data_s3multipartuploads'

    kwargs = dict(
        service_name='s3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_KEYID,
        aws_secret_access_key=S3_ACCESSKEY
    )

    s3 = boto3.client(**kwargs)

    s3 = boto3.resource(**kwargs)
    s3.Bucket(BUCKET_NAME).download_file(path, saveas)
