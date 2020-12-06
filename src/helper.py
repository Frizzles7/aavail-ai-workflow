#!/usr/bin/env python
"""
collection of helper functions
"""

import os
import sys
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd


def fetch_data(data_dir):
    """
    load json files from data_dir directory into a dataframe
    """

    ## test input
    if not os.path.isdir(data_dir):
        raise Exception("specified data directory does not exist")
    if not len(os.listdir(data_dir)) > 0:
        raise Exception("specified data directory does not contain any files")

    datafiles = sorted([f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))])
    
    # correct list of column names
    correct_cols = sorted(['country', 'customer_id', 'invoice', 'price',
                           'stream_id', 'times_viewed', 'year', 'month',
                           'day'])

    # create empty dataframe for loading data
    df = pd.DataFrame()
    
    # read in data
    for file in datafiles:
        filename = os.path.join(data_dir, file)
        file_df = pd.read_json(filename,
                               dtype={'country': str, 'customer_id': str, 'invoice': str, 
                                      'price': str, 'stream_id': str, 'times_viewed': str, 
                                      'year': str, 'month': str, 'day': str})
        
        # correct inconsistent column names
        cols = file_df.columns.to_list()
        if "total_price" in cols:
            file_df.rename(columns={"total_price": "price"}, inplace=True)
        if "StreamID" in cols:
            file_df.rename(columns={"StreamID": "stream_id"}, inplace=True)
        if "TimesViewed" in cols:
            file_df.rename(columns={"TimesViewed": "times_viewed"}, inplace=True)
        if sorted(file_df.columns.to_list()) != correct_cols:
            print("column names do not match")
        
        df = df.append(file_df)

    # remove alpha characters from invoice column
    df["invoice"] = df["invoice"].str.replace(r"[a-zA-Z]", "")
    
    # convert (customer_id, invoice, price, times_viewed, year, month, day) columns to numeric
    df[["customer_id", "invoice", "price", "times_viewed", "year", 
        "month", "day"]] = df[["customer_id", "invoice", "price", 
                               "times_viewed", "year", "month", 
                               "day"]].apply(pd.to_numeric, errors="coerce")

    # remove the negative values from price
    df = df[df["price"] >= 0]

    # create a datetime field with the date
    df["date"] = pd.to_datetime(df["year"]*10000 + df["month"]*100 + 
                                df["day"], format='%Y%m%d')

    # sort by date and reset the index
    df.sort_values(by="date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return(df)


def convert_to_ts(raw_df, country=None):
    """
    convert raw_df from output of fetch_data to a time series 
    dataframe by aggregating over each day
    """
    
    # filter by country if specified
    if country:
        if country not in set(raw_df["country"].unique()):
            raise Excpetion("country not found")
        df = raw_df[raw_df["country"] == country]
    else:
        df = raw_df

    ## use a date range to ensure all days are accounted for in the data
    invoice_dates = df['date'].values
    df_dates = df['date'].values.astype('datetime64[D]')
    
    start_month = '{}-{}'.format(df['year'].values[0],str(df['month'].values[0]).zfill(2))
    stop_month = '{}-{}'.format(df['year'].values[-1],str(df['month'].values[-1]).zfill(2))
    days = np.arange(start_month,stop_month,dtype='datetime64[D]')
    
    purchases = np.array([np.where(df_dates==day)[0].size for day in days])
    invoices = [np.unique(df[df_dates==day]['invoice'].values).size for day in days]
    streams = [np.unique(df[df_dates==day]['stream_id'].values).size for day in days]
    views =  [df[df_dates==day]['times_viewed'].values.sum() for day in days]
    revenue = [df[df_dates==day]['price'].values.sum() for day in days]
    year_month = ["-".join(re.split("-",str(day))[:2]) for day in days]

    df_time = pd.DataFrame({'date': days,
                            'purchases': purchases,
                            'unique_invoices': invoices,
                            'unique_streams': streams,
                            'total_views': views,
                            'year_month': year_month,
                            'revenue': revenue})
    
    return(df_time)


def fetch_ts(data_dir, clean=False):
    """
    function to read in new data and use csv to load quickly
    use clean=True when you want to re-create the files
    """

    ts_data_dir = os.path.join(data_dir, "ts-data")
    
    if clean:
        shutil.rmtree(ts_data_dir)
    if not os.path.exists(ts_data_dir):
        os.mkdir(ts_data_dir)

    # if files have already been processed load them        
    if len(os.listdir(ts_data_dir)) > 0:
        print("... loading ts data from files")
        return({re.sub("\.csv","",cf)[3:]:pd.read_csv(os.path.join(ts_data_dir,cf)) for cf in os.listdir(ts_data_dir)})

    # get original data
    print("... processing data for loading")
    df = fetch_data(data_dir)

    # find the top ten countries (wrt revenue)
    table = pd.pivot_table(df,index='country',values="price",aggfunc='sum')
    table.columns = ['total_revenue']
    table.sort_values(by='total_revenue',inplace=True,ascending=False)
    top_ten_countries =  np.array(list(table.index))[:10]

    file_list = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if re.search("\.json",f)]
    countries = [os.path.join(data_dir,"ts-"+re.sub("\s+","_",c.lower()) + ".csv") for c in top_ten_countries]

    # load the data
    dfs = {}
    dfs['all'] = convert_to_ts(df)
    for country in top_ten_countries:
        country_id = re.sub("\s+","_",country.lower())
        file_name = os.path.join(data_dir,"ts-"+ country_id + ".csv")
        dfs[country_id] = convert_to_ts(df, country=country)

    # save the data as csvs    
    for key, item in dfs.items():
        item.to_csv(os.path.join(ts_data_dir,"ts-"+key+".csv"),index=False)
        
    return(dfs)


def engineer_features(df,training=True):
    """
    for any given day the target becomes the sum of the next days revenue
    for that day we engineer several features that help predict the summed revenue
    
    the 'training' flag will trim data that should not be used for training
    when set to false all data will be returned
    """

    # extract dates
    dates = df['date'].values.copy()
    dates = dates.astype('datetime64[D]')

    # engineer some features
    eng_features = defaultdict(list)
    previous =[7, 14, 21, 28, 35, 42]
    y = np.zeros(dates.size)
    for d,day in enumerate(dates):

        # use windows in time back from a specific date
        for num in previous:
            current = np.datetime64(day, 'D') 
            prev = current - np.timedelta64(num, 'D')
            mask = np.in1d(dates, np.arange(prev,current,dtype='datetime64[D]'))
            eng_features["previous_{}".format(num)].append(df[mask]['revenue'].sum())

        # get the target revenue
        plus_30 = current + np.timedelta64(30,'D')
        mask = np.in1d(dates, np.arange(current,plus_30,dtype='datetime64[D]'))
        y[d] = df[mask]['revenue'].sum()

        # attempt to capture monthly trend with previous years data (if present)
        start_date = current - np.timedelta64(365,'D')
        stop_date = plus_30 - np.timedelta64(365,'D')
        mask = np.in1d(dates, np.arange(start_date,stop_date,dtype='datetime64[D]'))
        eng_features['previous_year'].append(df[mask]['revenue'].sum())

        # add some non-revenue features
        minus_30 = current - np.timedelta64(30,'D')
        mask = np.in1d(dates, np.arange(minus_30,current,dtype='datetime64[D]'))
        eng_features['recent_invoices'].append(df[mask]['unique_invoices'].mean())
        eng_features['recent_views'].append(df[mask]['total_views'].mean())
        eng_features['recent_streams'].append(df[mask]['unique_streams'].mean())

    X = pd.DataFrame(eng_features)
    # combine features into df and remove rows with all zeros
    X.fillna(0,inplace=True)
    mask = X.sum(axis=1)>0
    X = X[mask]
    y = y[mask]
    dates = dates[mask]
    X.reset_index(drop=True, inplace=True)

    if training == True:
        # remove the last 30 days (because the target is not reliable)
        mask = np.arange(X.shape[0]) < np.arange(X.shape[0])[-30]
        X = X[mask]
        y = y[mask]
        dates = dates[mask]
        X.reset_index(drop=True, inplace=True)
    
    return(X,y,dates)


if __name__ == "__main__":

    run_start = time.time() 
    data_dir = os.path.join(".","data","cs-train")
    print("...fetching data")

    ts_all = fetch_ts(data_dir, clean=False)

    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("load time:", "%d:%02d:%02d"%(h, m, s))

    for key,item in ts_all.items():
        print(key,item.shape)

