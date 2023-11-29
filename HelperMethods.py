
'''
File: HelperMethods.py
Author: Divya Veerapaneni MS4, Ong Lab
Description: helper functions designed to be imported for preprocessing datasets
'''

#import statements
import os
import pandas as pd
import numpy as np
import csv
from datetime import datetime 
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
from scipy.stats import f_oneway
import datetime
import warnings
import statistics
warnings.filterwarnings("ignore")
import plotly.express as px
import plotly.graph_objects as go
import pdb


#method to add column for difference between 2 desired column values
def add_diff_cols(dataset, col1, col2, new_col):
  # select obs. if left & right exist. If any is missing, diff cols show Nan
    dataset.loc[(~dataset[col1].isnull()) & (~dataset[col2].isnull()), new_col] = abs(dataset[col1] - dataset[col2])
    return dataset

#pre_process dataset to filter only desired columns, replace empty space 
# with nan, and convert number values into numerics
def clean_tbi_dataframe(df):
    cleaned_df = df.replace('  ', np.nan)
    cleaned_df['latr'] = cleaned_df['latr'].replace('i', '999')
    cleaned_df['latl'] = cleaned_df['latl'].replace('i', '999')
    numeric_cols = [ 'npil', 'sizel', 'minl', '%l', 'cvl', 'mcvl', 'dvl', 'latl','npir', 'sizer', 'minr', '%r', 'cvr', 'mcvr', 'dvr','latr'] 
    cleaned_df.loc[:,numeric_cols] = cleaned_df.loc[:,numeric_cols].apply(pd.to_numeric)
    cleaned_df.date = pd.to_datetime(cleaned_df.date)

    #columns for difference for npi and size
    cleaned_tbi_df = add_diff_cols(cleaned_df, 'npil', 'npir', 'npi_diff') # add NPI_DIFFERENCES
    cleaned_tbi_df = add_diff_cols(cleaned_tbi_df, 'sizel', 'sizer', 'size_diff') # add SIZE_DIFFERENCES

    #column for min for npi from both eyes
    cleaned_tbi_df['lower_npi'] = cleaned_tbi_df[['npil', 'npir']].min(axis=1) # add LOWER_NPI between LEFT-RIGHT, nan excluded, 0 included

    #column for averages for pupil metrics from both eyes
    cleaned_tbi_df['average_npi'] = cleaned_tbi_df[['npil', 'npir']].mean(axis=1) # add AVERAGE_NPI between LEFT-RIGHT, nan excluded, 0 included
    cleaned_tbi_df['average_mcv'] = cleaned_tbi_df[['mcvl', 'mcvr']].mean(axis=1) # add AVERAGE_MCV between LEFT-RIGHT, nan excluded, 0 included
    cleaned_tbi_df['average_cv'] = cleaned_tbi_df[['cvl', 'cvr']].mean(axis=1) # add AVERAGE_CV between LEFT-RIGHT, nan excluded, 0 included
    cleaned_tbi_df['average_dv'] = cleaned_tbi_df[['dvl', 'dvr']].mean(axis=1) # add AVERAGE_DV between LEFT-RIGHT, nan excluded, 0 included
    cleaned_tbi_df['average_min'] = cleaned_tbi_df[['minl', 'minr']].mean(axis=1) # add AVERAGE_MIN between LEFT-RIGHT, nan excluded, 0 included
    cleaned_tbi_df['average_%'] = cleaned_tbi_df[['%l', '%r']].mean(axis=1) # add AVERAGE_% between LEFT-RIGHT, nan excluded, 0 included
    cleaned_tbi_df['average_size'] = cleaned_tbi_df[['sizel', 'sizer']].mean(axis=1) # add AVERAGE_SIZE between LEFT-RIGHT, nan excluded, 0 included
    cleaned_tbi_df['average_latency'] = cleaned_tbi_df[['latl', 'latr']].mean(axis=1) # add AVERAGE_LAT between LEFT-RIGHT, nan excluded, 0 included
    cleaned_tbi_df.sort_values(by=['mrn', 'date'], ascending=True, inplace=True)
    return cleaned_tbi_df

# helper method to define time zero which is either presentation date or admit date
#if first obs occurs prior to admission date, time zero= presentation date
def set_time_zero(mrn, obs_date, admit_df):
  admission_date = admit_df[admit_df.mrn == mrn].ADMIT_DT.iloc[0]
  presentation_date = admit_df[admit_df.mrn == mrn].PRES_DT.iloc[0]
  if pd.isna(presentation_date) or pd.isna(admission_date):
    print(mrn, ' is not here')
  if obs_date < admission_date:
    return presentation_date
  else:
    return admission_date

# helper method to compute time in hrs from time zero
def compute_time_interval_from_time_zero(time_zero, obs_time):
  time_interval = obs_time - time_zero
  time_interval_in_hrs = round(time_interval.total_seconds() / (60 *60),3)
  return time_interval_in_hrs

# helper method to create dataframe for observations only within first x hrs from admission
def helper_create_first_x_hours_df(input_df, mrn, admit_df, hour_threshold):
  mrn_df = input_df[input_df.mrn == mrn]
  time_zero = set_time_zero(mrn, mrn_df.date.min(), admit_df)
  upper_limit_date =  time_zero + datetime.timedelta(hours=hour_threshold)
  subset_mrn_df = mrn_df[mrn_df.date <= upper_limit_date]
  subset_mrn_df['time'] = [compute_time_interval_from_time_zero(time_zero, obs_datetime) for obs_datetime in subset_mrn_df.date]
  subset_mrn_df['obs_number'] = range(len(subset_mrn_df))
  subset_mrn_df['time_zero'] = time_zero
  subset_mrn_df['total_obs'] = len(subset_mrn_df)
  if len(subset_mrn_df) == 0:
    print(mrn)
  return subset_mrn_df

#creates dateframe with only observations from the first x hrs from admission for each patient
def create_first_x_hours_df(input_df, admit_df, hour_threshold):
    dfs = [helper_create_first_x_hours_df(input_df, mrn, admit_df, hour_threshold) for mrn in input_df.mrn.unique()]
    final_df = pd.concat(dfs)
    return final_df

# for each mrn dataframe, creates a col_start and col_end to do a start stop column
def helper_create_start_end_column_version(df, mrn, col):
    input_df = df[df.mrn == mrn]
    start_column_list = []
    end_column_list = []
    for i in range(len(input_df)):
        if i == 0:
            start_column_list.append(0)
            end_column_list.append(input_df[col].iloc[i])  
        elif i < len(input_df):
            start_column_list.append(input_df[col].iloc[i-1]) 
            end_column_list.append(input_df[col].iloc[i])

    input_df[col + '_start'] = start_column_list
    input_df[col + '_end'] = end_column_list
    return input_df

# creates a col_start and col_end to do a start stop column
def create_start_end_column_version(input_df, col):
    dfs = [helper_create_start_end_column_version(input_df, mrn, col) for mrn in input_df.mrn.unique()]
    final_df = pd.concat(dfs)
    return final_df

#ABNORMAL PUPIL PHENOTYPES

#separate conditions
def has_lower_npi(row): #NPi min<3
  return row['lower_npi']<3

def has_npi_diff(row): #unilateral NPi diff>=0.7
  return row['npi_diff']>=0.7

def has_size_diff(row): #unilateral Size diff>=1
  return row['size_diff']>=1

def has_uni_lower_npi(row): #exclusively unilateral NPi min<3
  return has_lower_npi(row) & (max(row['npil'], row['npir'])>3)

#STAGE DEFINITIONS
def meets_stage1u_criteria(row): #isolated unilateral NPi min<3
  return has_uni_lower_npi(row) #and not has_npi_diff(row) and not has_size_diff(row)

def meets_stage2u_criteria(row): #concurrent unilateral NPi min<3, NPi diff>=0.7
  return has_uni_lower_npi(row) and has_npi_diff(row) #and not has_size_diff(row)

def meets_stage3u_criteria(row): #concurrent unilateral NPi min<3, NPi diff>=0.7, size diff>=1
  return has_uni_lower_npi(row) and has_npi_diff(row) and has_size_diff(row)

# def meets_stage3u_criteria_v2(row): #concurrent unilateral NPi min between 0 and 3, NPi diff>=0.7, size diff>=1
#   return has_uni_lower_npi(row) and row['lower_npi'] != 0 and has_npi_diff(row) and has_size_diff(row) 

def meets_stage4u_criteria(row):  #concurrent unilateral NPi min 0, NPi diff>=0.7, size diff>=1
  return has_uni_lower_npi(row) and row['lower_npi'] == 0 and has_npi_diff(row) and has_size_diff(row)

def meets_stage1b_criteria(row): #both eyes NPi<3 with neither eye 0
  return bool(row['npil']<3 and row['npir']<3 and row['lower_npi']!=0)

def meets_stage2b_criteria(row): #both eyes NPi<3 with exactly 1 eye zero
  return bool(max(row['npil'], row['npir'])< 3 and max(row['npil'], row['npir'])!= 0 and row['lower_npi'] == 0)

def meets_stage3b_criteria(row): #both eyes NPi is zero
  return bool(row['npil']==0 and row['npir']==0)

# sums of pupil abns
def has_any_stage(row):
   return bool(has_uni_abn(row) or has_bi_stage(row))

def has_uni_abn(row):
   return bool(has_uni_lower_npi(row) or has_npi_diff(row) or has_size_diff(row))

def has_bi_stage(row):
   return bool(row['npil']<3 and row['npir']<3)

# exploratory analysis
def has_npi_diff_size_diff(row): # npi diff, size diff, but NPi normal
   return has_npi_diff(row) and has_size_diff(row) and not has_lower_npi(row)

def has_npi_size_diff(row): #size diff, NPi<3, diff NPi normal
   return has_lower_npi(row) and has_size_diff(row) and not has_npi_diff(row)

# compute incidence row by row
def compute_incidence(row):
  row['any_incidence'] = has_any_stage(row) 

  #separate conditions
  row['poor_npi_incidence'] = has_lower_npi(row)
  row['npi_diff_incidence'] = has_npi_diff(row)
  row['size_diff_incidence'] = has_size_diff(row)

  row['uni_any_incidence'] = has_uni_abn(row)

  row['stage1u_incidence'] = meets_stage1u_criteria(row)
  row['stage2u_incidence'] = meets_stage2u_criteria(row)
  row['stage3u_incidence'] = meets_stage3u_criteria(row)
  #row['stage3u_v2_incidence'] = meets_stage3u_criteria_v2(row)
  row['stage4u_incidence'] = meets_stage4u_criteria(row)
  
  row['bi_incidence'] = has_bi_stage(row)

  row['stage1b_incidence'] = meets_stage1b_criteria(row)
  row['stage2b_incidence'] = meets_stage2b_criteria(row)
  row['stage3b_incidence'] = meets_stage3b_criteria(row)

  #exploratory analysis
  row['npi_diff_size_diff_incidence'] = has_npi_diff_size_diff(row)
  row['size_diff_poor_npi_incidence'] = has_npi_size_diff(row)
  return row

# helper method to compute frequency/burden per patient 
def helper_compute_burden(mrn_df):
  list_cols = ['mrn', 'any_burden', 'poor_npi_burden', 'npi_diff_burden' ,'size_diff_burden', 'uni_any_burden',\
               'stage1u_burden', 'stage2u_burden', 'stage3u_burden', 'stage4u_burden',\
                'bi_burden', 'stage1b_burden', 'stage2b_burden', 'stage3b_burden', \
                  'npi_diff_size_diff_burden', 'size_diff_poor_npi_burden']
  
  mrn_df = mrn_df.apply(compute_incidence, axis=1)
  output_df = pd.DataFrame(columns=list_cols)
  output_df['mrn'] = mrn_df.mrn.mode()

  for col in list_cols[1:]:
    unique_outcomes = mrn_df[col.replace('burden', 'incidence')].unique()
    if len(unique_outcomes) == 1 and unique_outcomes[0] == False: # no burdens present in column
      output_df[col] = 0
    else:
      output_df[col] = mrn_df[col.replace('burden', 'incidence')].value_counts()[True] / len(mrn_df) * 100
  return output_df

# method to compute frequency/burden
def compute_burden(input_df):
    output_df = [helper_compute_burden(input_df[input_df.mrn == mrn]) for mrn in input_df.mrn.unique()]
    final_df = pd.concat(output_df)
    final_df = final_df.set_index('mrn')
    return final_df

#helper method to organize race subcategories so only white, black, hispanic, other, unk exist
def combine_race(row):
  race = row['RACE']
  #White is reference race, so if any White, would be considered White
  if 'White' in row['RACE']:
    race = 'White'
  #combine all small race categories into 'Other'
  if row['RACE'] in ['Asian', 'Filipino', 'American Indian / Native American']:
    race = 'Other'
  #combine 'Unknown' and 'Declined / Not Available'
  if row['RACE'] == 'Declined / Not Available':
    race = 'Unknown'
  trimmed_race = race.split(' ')
  return trimmed_race[0]

#helper method to combine race and ethnicity appropriately
def clean_demographics(df):
  df['RACE'] = df.apply(combine_race, axis=1)
  var = 'RACE'
  cat_list='var'+'_'+var
  cat_list = pd.get_dummies(df[var], prefix=var)
  df=df.join(cat_list)
  df= df.drop(columns=['ETHNICITY'])
  return df

#helper method to categorize discharge disposition appropriately
def clean_discharge_disposition(df):
  discharge_disposition_dict = {'Home or Self Care':'Home', 
                              'Skilled Nursing Facility':'Rehab Facility',
                                'Rehab Facility': 'Rehab Facility',
                                'Home-Health Care Svc' :'Home',
                                'Long Term Care': 'Long Term Care',
                                'Deceased': 'Deceased',
                                'Hospice/Medical Facility' : 'Hospice',
                                'Against Medical Advice':  'Against Medical Advice',
                                'Hospice/Home': 'Hospice'}
  df = df.replace({"DISCH_DISP_NM":discharge_disposition_dict})
  return df