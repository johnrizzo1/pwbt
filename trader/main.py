from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

from dotenv import load_dotenv

import pwb_toolbox.datasets as pwb_ds
import sqlalchemy as sa
import backtrader as bt

load_dotenv()

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    
    DB_URI = f'postgresql://{os.environ["PG_USER"]}:{os.environ["PG_PASS"]}@{os.environ["PG_URL"]}/{os.environ["PG_DB"]}'

    df = pwb_ds.load_dataset("Stocks-Daily-Price")

    engine = sa.create_engine(DB_URI)
    cerebro.broker.setcash(100000.0)
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue()}')
    cerebro.run()
    print(f'Final Portfolio Value: {cerebro.broker.getvalue()}')