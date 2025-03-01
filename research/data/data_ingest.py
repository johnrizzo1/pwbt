import os
import csv
import time
from io import StringIO

from dotenv import load_dotenv

import pwb_toolbox.datasets as pwb_ds

import pandas as pd
import numpy as np

from sqlalchemy import create_engine

load_dotenv()

def psql_insert_copy(table, conn, keys, data_iter): #mehod
    """
    Execute SQL statement inserting data

    Parameters
    ----------
    table : pandas.io.sql.SQLTable
    conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
    keys : list of str
        Column names
    data_iter : Iterable that iterates the values to be inserted
    """
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = '{}.{}'.format(table.schema, table.name)
        else:
            table_name = table.name

        sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)

indices = { 
    "All-Daily-News": "daily_news",
    "Bonds-Daily-Price": "bonds_daily_price",
    "Commodities-Daily-Price": "commodities_daily_price",
    "Cryptocurrencies-Daily-Price": "cryptocurrencies_daily_price",
    "ETFs-Daily-Price": "etfs_daily_price",
    "Forex-Daily-Price": "forex_daily_price",
    "Indices-Daily-Price": "indices_daily_price",
    "Stocks-1min-Price": "stocks_1m_price",
    "Stocks-Daily-Price": "stocks_daily_price",
    "Stocks-Quarterly-BalanceSheet": "stocks_quarterly_balancesheet",
    "Stocks-Quarterly-CashFlow": "stocks_quarterly_cashflow",
    "Stocks-Quarterly-IncomeStatement": "stocks_quarterly_incomestatement"
}

CONNECTION_STRING = f'postgresql://{os.environ["PG_USER"]}:{os.environ["PG_PASS"]}@{os.environ["PG_URL"]}/{os.environ["PG_DB"]}'
engine = create_engine(CONNECTION_STRING)

start_total_time = time.time() # get start time before insert
print(f'Starting Data Load {start_total_time}')

for key in indices.keys():
    print(f'Loading {key} dataset')
    start_time = time.time() # get start time before insert
    df = pwb_ds.load_dataset(key) #, extend=True)
    end_time = time.time() # get end time after insert
    total_time = end_time - start_time # calculate the time
    print(f"Dataset Load time: {total_time} seconds") # print time


    start_time = time.time() # get start time before insert
    df.to_sql(
        name=indices[key],
        con=engine,
        if_exists="replace",
        index=False,
        method=psql_insert_copy
    )

    end_time = time.time() # get end time after insert
    total_time = end_time - start_time # calculate the time
    print(f"Insert time: {total_time} seconds") # print time


end_total_time = time.time() # get start time before insert
total_time = end_total_time - start_total_time # calculate the time
print(f"Total time: {total_time} seconds") # print time