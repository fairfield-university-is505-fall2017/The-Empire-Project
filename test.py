import pandas as pd


def add_columns(dataframe,newcolname,names):
    dataframe[newcolname]=0
    for i in names:
        dataframe[newcolname]=dataframe[newcolname]+dataframe[i]
    return dataframe    
    