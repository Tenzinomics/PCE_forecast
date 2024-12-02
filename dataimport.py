
import pyeviews as evp # for eviews (ev)
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def data_main(data_code,startyr):
    #using eviews to call in data from macrobond.
    #The reason for using eviews is that they have a prebuilt function that we can use and dont have to recreate it on python
    eviewsapp = evp.GetEViewsApp(instance='new',showwindow=True)
    evp.Run('WFCREATE monthly '+str(startyr)+' '+str(dt.date.today().year), app=eviewsapp)
    evp.Run("   smpl @all  ", app=eviewsapp)
    for x in data_code["series"]:
        evp.Run("   copy(c=l) mb::"+x, app=eviewsapp)
        evp.Run(x+".ipolate "+x+"i", app=eviewsapp)

        #Series Transformation
        transform = data_code[data_code["series"]==x]["transformation"].values[0]

        #%yoy chng
        if transform == 1:
            evp.Run("series _trans_yoy_"+x+" = @pcy("+x+"i)", app=eviewsapp)

        #12mnth value change
        if transform == 2:
            evp.Run("series _trans_12mchng_"+x+" = "+x+"i -"+x+"i(-12)", app=eviewsapp)
        
        #No transformation or levels
        if transform == 0:
            evp.Run("series _trans_none_"+x+" = "+x+"i", app=eviewsapp)
        
        #This condition is for the input variable #at the moment just applied the yoy change. 
        if transform == 999:
            evp.Run("series _output_12mchng_"+x+" = "+x+"i -"+x+"i(-12)", app=eviewsapp)

    df_ev = evp.GetWFAsPython(app=eviewsapp)
    eviewsapp.Hide()
    eviewsapp = None
    evp.Cleanup()

    #This is to run PCA
    filtered_columns_pca = [col for col in df_ev.columns if '_TRANS' in col]
    filtered_df_pca = df_ev[filtered_columns_pca] #this is a database for the 


    #This is for the regression
    filtered_columns_reg = [col for col in df_ev.columns if '_OUTPUT' in col]
    # Create a new DataFrame with only those columns
    filtered_df_reg = df_ev[filtered_columns_reg]

    return filtered_df_pca, filtered_df_reg, df_ev
