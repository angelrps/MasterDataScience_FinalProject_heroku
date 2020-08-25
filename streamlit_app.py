
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
pd.options.display.max_columns = None
pd.options.display.max_rows = None
import altair as alt

#############################   DEFINE FUNCTIONS START   #############################

def get_LocationIDs():
    # 1. Import Location and Borough columns form NY TAXI ZONES dataset
    dfzones = pd.read_csv('NY_taxi_zones.csv', sep=',',
                          usecols=['LocationID', 'borough'])

    # 2. Filter Manhattan zones
    dfzones = dfzones[dfzones['borough']=='Manhattan']\
                    .drop(['borough'], axis=1)\
                    .sort_values(by='LocationID')\
                    .drop_duplicates('LocationID').reset_index(drop=True)    
    return dfzones

# RUN FUNCTIONS
df_zones = get_LocationIDs()
st.write(df_zones)
