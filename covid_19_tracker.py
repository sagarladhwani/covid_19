# %%
"""
# Documentation
"""

# %%
"""
### Code Details
"""

# %%
"""
Author            : Sagar Ladhwani (LinkedIn: https://www.linkedin.com/in/sagar-ladhwani-713b96112/) <br>
Created On        : 31st May 2021 <br>
Last Modifiied On : 22nd June 2021 <br>
Code Descr        : A detailed tracker of India's Covid-19 Pandemic & Vaccination Drive
"""

# %%
"""
### Importing Required Libraries
"""

# %%
import pandas as pd
import numpy as np
import warnings
import os
import datetime as dt
from datetime import datetime 
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from matplotlib.dates import DateFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

# %%
## Ignoring Warnings
warnings.filterwarnings('ignore')

# %%
## Getting terminal width for better printing later on
width = os.get_terminal_size().columns

# %%
"""
# 1. Covid Cases Data
"""

# %%
"""
## Data Exploration
"""

# %%
## Read the cases data from Covid API
cases = pd.read_csv("https://api.covid19india.org/csv/latest/states.csv")
print(cases.shape)
cases.head()

# %%
## Drop unnecessary columns and Unassigned Cases
cases.drop(['Other','Tested'],axis=1,inplace=True)
cases = cases[cases['State']!='State Unassigned']
cases.shape

# %%
## Create required columns
cases['Active'] = cases.Confirmed - cases.Recovered - cases.Deceased
cases['Recovered/Deceased'] = cases.Recovered + cases.Deceased
cases.head()

# %%
## Change type of Date column
cases['Date'] = pd.to_datetime(cases['Date'])
cases.dtypes

# %%
## Check for null values (if any)
cases.isna().sum()

# %%
## Create a list of all states removing India from the list
states = list(np.sort(cases['State'].unique()))
states.remove('India')
print(states)

# %%
"""
## a. Trend of Confirmed, Recovered/Deceased and Active Cases over time
"""

# %%
## Defining starting and end points for analysis
start_date = datetime(2021, 1, 1)
end_date = datetime.today() + dt.timedelta(days=1)

# %%
## Define a function to plot time-series cases trend
def plot_conf_rec_act(state_name):
    
    ## Filter for State and Analysis Time Period
    state_df = cases[cases['State']==state_name]
    state_df = state_df[(state_df['Date']>=start_date) & (state_df['Date']<=end_date)]
    state_df['Mon-Year'] = state_df.Date.dt.strftime("%b-%Y")
    state_df.reset_index(drop=True,inplace=True)
    
    ## Creating a separate Dataframe for plotting text at monthly interval and last data point
    first_dates_df = state_df.loc[state_df.groupby(['Mon-Year'])['Date'].idxmin()]
    first_dates_df = first_dates_df.append(state_df.loc[state_df.index[-1]])
    
    ## Creating plots
    fig, ax = plt.subplots(figsize=(15,6))

    ## Plots for Confirmed, Recovered/Deceased and Active cases with markers at monthly interval and last data point
    markers_on = list(first_dates_df.index) + [state_df.index[-1]]
    ax.plot(state_df.Date,state_df.Confirmed,marker='o',markevery=markers_on,label='Confirmed')
    ax.plot(state_df.Date,state_df['Recovered/Deceased'],marker='^',markevery=markers_on,label='Recovered/Deceased')
    ax.plot(state_df.Date,state_df.Active,marker='D',markevery=markers_on,label='Active')
    ax.legend(loc='upper right',bbox_to_anchor=(1.225, 1.005))

    ## Formatting X-axis for appropriate Date format, Major and Minor Ticks along with axis limits
    date_form = DateFormatter("%b-%y")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=start_date.weekday()))
    ax.set_xlim([start_date - dt.timedelta(5), end_date])
    ax.set_ylim(top=state_df.Confirmed.max()*1.2)

    ## Using a loop to plot text for all the points of each of the 3 trend lines
    for i in range(len(first_dates_df)):
        ax.text(mdates.date2num(first_dates_df.Date.iloc[i])-2,1.1*first_dates_df.Confirmed.iloc[i],'{:,}'.format(first_dates_df.Confirmed.iloc[i]))
        ax.text(mdates.date2num(first_dates_df.Date.iloc[i])-2,0.8*first_dates_df['Recovered/Deceased'].iloc[i],'{:,}'.format(first_dates_df['Recovered/Deceased'].iloc[i]))
        ax.text(mdates.date2num(first_dates_df.Date.iloc[i])-2,first_dates_df.Active.iloc[i] + 0.05*first_dates_df.Confirmed.iloc[i],'{:,}'.format(first_dates_df.Active.iloc[i]))
    
    ## Adding labels
    plt.xlabel('Month-Year')
    plt.ylabel('Count')
    plt.title('Total Covid-19 Cases in '+state_name)
    plt.legend(facecolor='White')
    plt.show()

# %%
## Setting Style and Context for Seaborn plots along with colors to be used
sns.set_style('darkgrid')
sns.set_context('notebook',font_scale=1.2)
colors = sns.color_palette('muted')

# %%
"""
### Visualizations
"""

# %%
## Plotting time-series of Confirmed, Recovered/Deceased and Active cases for India as a whole
plot_conf_rec_act('India')

# %%
## Plotting time-series of Confirmed, Recovered/Deceased and Active cases for every State and UT
for state_name in states:
    plot_conf_rec_act(state_name)

# %%
"""
## b. Top 3 states with the highest number of Active cases at any point of time
"""

# %%
## Loading a copy of original data and subsetting data for choosen time period
state_df = cases.copy()
#state_df = state_df[(state_df['Date']>=start_date) & (state_df['Date']<=end_date)]
state_df['Year-Mon'] = state_df.Date.dt.strftime("%Y-%m")
state_df.head()

# %%
## Given this is a time-series data picking up the last observation of the month as the latest snapshots
state_df['max_date'] = state_df.groupby(['Year-Mon','State'])['Date'].transform('max')
grouped_df = state_df[state_df['Date']==state_df['max_date']]

## Calculaing percentages to identify the most critical states
grouped_df['dec_per'] = round(grouped_df['Deceased']/grouped_df['Confirmed']*100,2)
grouped_df['act_per'] = round(grouped_df['Active']/grouped_df['Confirmed']*100,2)
grouped_df['rec_per'] = round(grouped_df['Recovered']/grouped_df['Confirmed']*100,2)
grouped_df.head()

# %%
## Subset for active cases and states only
states_only_grouped_df = grouped_df[grouped_df['State']!='India']
active_grouped_df = states_only_grouped_df[['Year-Mon','State','Active','act_per']]
active_grouped_df.head()

# %%
## Defining a function that returns the highest 3 values along with their indices 
## It will be used to find Top 3 states with the highest percent of active cases for each Month-Year time period

def max_finder(ser):
    
    ## Creating a blank dataframe where observatons will be appended
    temp_df = pd.DataFrame(columns=['nth_largest','value','indexes'])
    
    ## Find Top 3 values along with their indices and append to above dataframe
    try:
        nth_largest = 1
        value = ser.max()
        indexes = ', '.join(list(ser[ser==value].index))
        temp_df.loc[len(temp_df)] = [nth_largest,value,indexes]
    except:
        pass
    
    try:
        nth_largest = 2
        value = sorted(ser.unique())[-2]
        indexes = ', '.join(list(ser[ser==value].index))
        temp_df.loc[len(temp_df)] = [nth_largest,value,indexes]
    except:
        pass

    try:
        nth_largest = 3
        value = sorted(ser.unique())[-3]
        indexes = ', '.join(list(ser[ser==value].index))
        temp_df.loc[len(temp_df)] = [nth_largest,value,indexes]
    except:
        pass
    
    ## Return dataframe with ranks, values and indices
    return temp_df

# %%
## Create an empty dataframe that will contain time-period (Month-Year), Top 3 states by active cases based on absolute numbers 
## and percentage values
summary_df = pd.DataFrame()

## Run the loop on a Time-period based Grouped object for individual states
for name, df in active_grouped_df.groupby('Year-Mon'):
    
    ## Set index as State so that our max_finder function that returns index of max values can provide state names
    df.set_index('State',inplace=True)
    
    ## Add suffix distinguishing top 3 by absolute numbers and percent values and concat them with their resp. time-period
    df_1 = max_finder(df['Active']).add_suffix('_by_cases')
    df_2 = max_finder(df['act_per']).add_suffix('_by_percent')
    final_df = pd.concat([df_1,df_2],axis=1)
    final_df['Year-Mon'] = name
    
    ## Append observations for each individual time period to summary dataframe
    summary_df = summary_df.append(final_df)
    
## Check the final dataframe
summary_df.head()

# %%
## Creating summary columns for both absolute numbers and percent values
summary_df['active_cases'] = summary_df['indexes_by_cases'].apply(str) + ' : ' + summary_df['value_by_cases'].apply(str)
summary_df['active_percent'] = summary_df['indexes_by_percent'].apply(str) + ' : ' + summary_df['value_by_percent'].apply(str) + '%'
summary_df.head()

# %%
## Picking final columns, pivoting to bring to required format and changing names
active_cases_top = summary_df[['Year-Mon', 'nth_largest_by_cases', 'active_cases']]
active_cases_top = active_cases_top.pivot_table(index=['Year-Mon'],columns='nth_largest_by_cases',values='active_cases',aggfunc='first')
active_cases_top.columns.name = None
active_cases_top.index.name = None
active_cases_top = active_cases_top.add_prefix('No_').add_suffix('_by_cases')
active_cases_top.head()

# %%
## Picking final columns, pivoting to bring to required format and changing names
active_percent_top = summary_df[['Year-Mon', 'nth_largest_by_percent', 'active_percent']]
active_percent_top = active_percent_top.pivot_table(index=['Year-Mon'],columns='nth_largest_by_percent',values='active_percent',aggfunc='first')
active_percent_top.columns.name = None
active_percent_top.index.name = None
active_percent_top = active_percent_top.add_prefix('No_').add_suffix('_by_percent')
active_percent_top.head()

# %%
"""
### Visualizations
"""

# %%
## Displaying top 3 states at every point of time based on absolute numbers of active cases
fig = go.Figure(data=[go.Table(header=dict(values=['Year-Mon']+list(active_cases_top.columns), align='left'),
                 cells=dict(values=[active_cases_top.index, active_cases_top.iloc[:,0], active_cases_top.iloc[:,1], active_cases_top.iloc[:,2]],
                           align='left'))
                     ])
fig.update_layout(height=len(active_cases_top)*30, margin=dict(r=20, l=3, t=5, b=0))
fig.show()

# %%
## Displaying top 3 states at every point of time based on percent values of active cases
fig = go.Figure(data=[go.Table(header=dict(values=['Year-Mon']+list(active_percent_top.columns), align='left'),
                 cells=dict(values=[active_percent_top.index, active_percent_top.iloc[:,0], active_percent_top.iloc[:,1], active_percent_top.iloc[:,2]],
                           align='left'))
                     ])
fig.update_layout(height=len(active_cases_top)*30, margin=dict(r=20, l=3, t=3, b=0))
fig.show()

# %%
"""
## c. Trend of Recovered and Death Percentage over time
"""

# %%
## Subset for Recovered and Deceased case columns only
data_india = grouped_df[grouped_df['State']=='India'][['Year-Mon','dec_per','rec_per']]
data_india.head()

# %%
## Define a function to plot trend of Recovered and Death Percentages over time
def dec_rec_plotter(state):
    
    ## Filter for State and required columns
    data = grouped_df[grouped_df['State']==state][['Year-Mon','dec_per','rec_per']]
    
    ## Creating plots
    fig, ax = plt.subplots(1,2,figsize=(15,3))

    ## Bar plot for Deceased cases for individual states along with a line plot showing national average
    ax[0].plot(data_india['Year-Mon'],data_india['dec_per'],color=colors[3],linewidth=2,label='India')
    ax[0].bar(data['Year-Mon'],data['dec_per'],color=colors[0],width=0.5)
    ax[0].legend(loc=1,facecolor='White')
    
    ## Formatting ticks and adding labels
    ax[0].tick_params(rotation=90)
    ax[0].set_xlabel('Year-Mon',labelpad=25)
    ax[0].set_ylabel('Percent',labelpad=15)
    ax[0].set_title('Trend of Deceased People over Time')

    ## Bar plots for Recovered cases for individual states along with a line plot showing national average
    ax[1].plot(data_india['Year-Mon'],data_india['rec_per'],color=colors[4],linewidth=2,label='India')
    ax[1].bar(data['Year-Mon'],data['rec_per'],color=colors[1])
    ax[1].legend(loc=1,facecolor='White')
    
    ## Formatting ticks and adding labels
    ax[1].tick_params(rotation=90)
    ax[1].set_xlabel('Year-Mon',labelpad=25)
    ax[1].set_ylabel('Percent',labelpad=15)
    ax[1].set_title('Trend of Recovered People over Time')
    ax[1].set_ylim(top=120)

    ## Adding figure title
    fig.suptitle('Percent wise trend of Deceased and Recovered cases over time in '+state+' compared to India',y=1.2,fontsize=20)
    plt.show()

# %%
"""
### Visualizations
"""

# %%
## As a sample, test case numbers for Maharashtra and comparing them to National Average over time
dec_rec_plotter('Maharashtra')

# %%
## Plotting trend of Recovered and Death Percentages over time for every State and UT
for state in states:
    print('\n')
    dec_rec_plotter(state)

# %%
"""
# 2. Covid Vaccinations Data
"""

# %%
"""
## Data Exporation
"""

# %%
## Read the vaccinations data from Covid API
vac = pd.read_csv("http://api.covid19india.org/csv/latest/cowin_vaccine_data_statewise.csv")
vac.head()

# %%
## Keeping ony granular detail columns which can be used to generate aggrgated ones whenever needed
vac = vac[['Updated On', 'State', 'Total Covaxin Administered', 'Total CoviShield Administered', 
           'Total Sputnik V Administered', 'First Dose Administered', 'Second Dose Administered', 
           '18-45 years (Age)', '45-60 years (Age)', '60+ years (Age)']]
vac.head()

# %%
## Change type of Date column
vac['Updated On'] = pd.to_datetime(vac['Updated On'],format='%d/%m/%Y')
vac.dtypes

# %%
## Drop rows with na threshold of 5 and check for NA values
vac.dropna(thresh=5,inplace=True)
vac.isna().sum()

# %%
"""
## Data Processing
##### Since provided data is a time series and we want to analyze the daily numbers, we need to nullify the cummulative values  
"""

# %%
## Defining a function that takes a cumulative series and returns the original one
def nullify_cummulative(ser):
    
    ## The first element will remain the same in the original list as well
    orig = [ser[0]]
    
    ## For every subsequeent entry, substract the previous one to get original value and append it to original list
    for i in range(1,len(ser)):
        orig.append(ser[i]-ser[i-1])
        
    ## Return the original series
    return orig

# %%
## Creating an empty dataframe that will contain the original values (not cummulated) for all metrics 
mod_df = pd.DataFrame()

## Run the loop on a State based Grouped object and sort each of them by date inside the group
for name, df in vac.groupby('State',as_index=False):
    df.sort_values('Updated On',inplace=True)
    df.reset_index(drop=True,inplace=True)
    
    ## We'll fill inital null values of cumulative series with 0
    ## And if there are any null value amidst the cummulative series, we'll fill those by its previous observation
    for col in df.columns[2:]:
        df.loc[:df[col].first_valid_index()-1,col] = 0
        df[col].fillna(method='ffill',inplace=True)  
    
    ## Now, we don't have any null values thus we can perform our cummulative nullifying action
    orig_values = df.iloc[:,2:].apply(nullify_cummulative)
    orig_values = orig_values.applymap(int)

    ## Combine the numeric columns with Date and State to get the final modified dataframe
    vaccine_orig = pd.concat([df.iloc[:,:2],orig_values],axis=1)
    mod_df = mod_df.append(vaccine_orig)   

## Check the final modified dataframe
mod_df.reset_index(drop=True,inplace=True)
mod_df.head()

# %%
## Check the modified dataframe for any null values
mod_df.isna().sum()

# %%
mod_df = mod_df[mod_df['Updated On'] < dt.datetime.today()-dt.timedelta(days=2)].shape

# %%
"""
## a. Daily Vaccination jabs by Drug Brands
"""

# %%
## Define a function to plot daily vaccination jabs by Drug Brands
def daily_vaccination_plotter_by_drug_brand(state):
    
    ## Filter for State and find the most latest observation and print the summary numbers of vaccines administered till date 
    df = vac[vac['State']==state]
    last_entry = df[df['Updated On'] == df['Updated On'].max()]
    
    str_to_print = '        '+state+' as of '+last_entry['Updated On'].iloc[0].strftime("%d-%b-%Y")+': '+\
        ' CoviShield '+'{:,}'.format(int(last_entry['Total CoviShield Administered'].iloc[0]))+'; '+\
        ' Covaxin '+'{:,}'.format(int(last_entry['Total Covaxin Administered'].iloc[0]))+'; '+\
        ' Sputnik '+'{:,}'.format(int(last_entry['Total Sputnik V Administered'].iloc[0]))+'; '+\
        ' Total '+'{:,}'.format(int(last_entry['Total CoviShield Administered'].iloc[0] +\
                                     last_entry['Total Covaxin Administered'].iloc[0] +\
                                     last_entry['Total Sputnik V Administered'].iloc[0]))
    print(str_to_print.center(width))
    
    ## Filter for State and summarize daily numbers on weekly level
    state_df = mod_df[mod_df['State']==state]
    data = state_df.resample('W',on='Updated On').sum()
    data.reset_index(inplace=True)

    ## Creating plots
    fig, ax = plt.subplots(figsize=(15,5))

    ## Plots for Daily Administered Doses of CoviShield and Covaxin over time
    ax.bar(mdates.date2num(data['Updated On'])-0.75,data['Total CoviShield Administered'],label='Covishield',width=1.5)
    ax.bar(mdates.date2num(data['Updated On'])+0.75,data['Total Covaxin Administered'],label='Covaxin',width=1.5)
    plt.legend(loc=1,facecolor='White')

    ## Formatting X-axis for appropriate Date format, Major and Minor Ticks along with axis limits
    date_form = DateFormatter("%d-%b-%y")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1,byweekday=(2)))
    _, y_top = ax.get_ylim()
    ax.set_ylim(top=max(data['Total CoviShield Administered'])*1.25)
    
    ## Using a loop to plot text for all the points of each of the 2 bar lines
    for i in range(len(data)):
        ax.text(mdates.date2num(data['Updated On'].iloc[i])-1.3,  y_top/40 + data['Total CoviShield Administered'].iloc[i], '{:,}'.format(data['Total CoviShield Administered'].iloc[i]), rotation=90, size = 9)
        ax.text(mdates.date2num(data['Updated On'].iloc[i])+0.45, y_top/40 + data['Total Covaxin Administered'].iloc[i],       '{:,}'.format(data['Total Covaxin Administered'].iloc[i]),    rotation=90, size = 9)

    ## Adding labels
    fig.autofmt_xdate()
    plt.xlabel('Timeline',labelpad=10)
    plt.ylabel('Number of Vaccines',labelpad=10)
    plt.title('Daily Vaccinations by drug brand in '+state)
    plt.show()

# %%
## Plotting daily vaccination jabs by Drug Brands for India as a whole
daily_vaccination_plotter_by_drug_brand('India')

# %%
## Plotting daily vaccination jabs by Drug Brands for every State and UT
for state in states:
    daily_vaccination_plotter_by_drug_brand(state)

# %%
"""
## b. Daily Vaccination jabs by Dose Number
"""

# %%
## Define a function to plot daily vaccination jabs by Dose number
def daily_vaccination_plotter_by_dose_number(state):
    
    ## Filter for State and find the most latest observation and print the summary numbers of vaccines till date
    df = vac[vac['State']==state]
    last_entry = df[df['Updated On'] == df['Updated On'].max()]
    str_to_print = '       '+state+' as of '+last_entry['Updated On'].iloc[0].strftime("%d-%b-%Y")+': '+\
        ' First-Dose '+'{:,}'.format(int(last_entry['First Dose Administered'].iloc[0]))+'; '+\
        ' Second-Dose '+'{:,}'.format(int(last_entry['Second Dose Administered'].iloc[0]))+'; '+\
        ' Total Doses '+'{:,}'.format(int(last_entry['First Dose Administered'].iloc[0] +\
                                         last_entry['Second Dose Administered'].iloc[0]))
    print(str_to_print.center(width))
    
    ## Filter for State and summarize daily numbers on weekly level
    state_df = mod_df[mod_df['State']==state]
    data = state_df.resample('W',on='Updated On').sum()
    data.reset_index(inplace=True)
    
    ## Creating plots
    fig, ax = plt.subplots(figsize=(15,5))

    ## Plots for Daily Administered First and Second Doses over time
    ax.bar(mdates.date2num(data['Updated On'])-0.75,data['First Dose Administered'],label='First Dose',width=1.5)
    ax.bar(mdates.date2num(data['Updated On'])+0.75,data['Second Dose Administered'],label='Second Dose',width=1.5)
    plt.legend(loc=1,facecolor='White')

    ## Formatting X-axis for appropriate Date format, Major and Minor Ticks along with axis limits
    date_form = DateFormatter("%d-%b-%y")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1,byweekday=(2)))
    ax.set_ylim(top=max(data['First Dose Administered'])*1.25)
    _, y_top = ax.get_ylim()

    ## Using a loop to plot text for all the points of both the trend lines
    for i in range(len(data)):
        ax.text(mdates.date2num(data['Updated On'].iloc[i])-1.3,  y_top/40 + data['First Dose Administered'].iloc[i], '{:,}'.format(data['First Dose Administered'].iloc[i]), rotation=90, size = 9)
        ax.text(mdates.date2num(data['Updated On'].iloc[i])+0.45, y_top/40 + data['Second Dose Administered'].iloc[i],   '{:,}'.format(data['Second Dose Administered'].iloc[i]),    rotation=90, size = 9)

    ## Adding labels
    fig.autofmt_xdate()
    plt.xlabel('Timeline',labelpad=10)
    plt.ylabel('Number of Vaccines',labelpad=10)
    plt.title('Daily Vaccinations by dose number in '+state)
    plt.show()

# %%
## Plotting daily vaccination jabs by Dose number for India as a whole
daily_vaccination_plotter_by_dose_number('India')

# %%
## Plotting daily vaccination jabs by Dose number for every State and UT
for state in states:
    daily_vaccination_plotter_by_dose_number(state)

# %%
"""
## c. Daily Vaccinated Individuals by Age Group
"""

# %%
## Subset the data only for the time post which age data is available in the dataframe
age_df = mod_df[mod_df['Updated On']>'2021-03-15']
age_df = age_df[['Updated On', 'State','18-45 years (Age)', '45-60 years (Age)', '60+ years (Age)']]
age_df['Total Individuals Vaccinated'] = age_df[['18-45 years (Age)', '45-60 years (Age)', '60+ years (Age)']].sum(axis=1)
age_df.head()

# %%
## Define a function to plot number of daily vaccinated individuals by age group
def individual_age_plotter(state):
    
    ## Filter for State and find the most latest observation and print the summary numbers of vaccines till date
    df = vac[vac['State']==state]
    last_entry = df[df['Updated On'] == df['Updated On'].max()]
    str_to_print = '       '+state+' as of '+last_entry['Updated On'].iloc[0].strftime("%d-%b-%Y")+': '+\
        ' 18-45 yrs: '+'{:,}'.format(int(last_entry['18-45 years (Age)'].iloc[0]))+'; '+\
        ' 45-60 yrs: '+'{:,}'.format(int(last_entry['45-60 years (Age)'].iloc[0]))+'; '+\
        ' 60+ yrs: '+'{:,}'.format(int(last_entry['60+ years (Age)'].iloc[0]))+'; '+\
        ' Total: '+'{:,}'.format(int(last_entry['18-45 years (Age)'].iloc[0] +\
                                    last_entry['45-60 years (Age)'].iloc[0] +\
                                    last_entry['60+ years (Age)'].iloc[0]))
    print(str_to_print.center(width))
    
    ## Filter for State and summarize daily numbers on weekly level
    state_df = age_df[age_df['State']==state]
    data = state_df.resample('W',on='Updated On').sum()
    data.reset_index(inplace=True)
    
    ## On any given day, calculate how much percentage of vacciated individuals belonged to each category  
    data['18-45 pct'] = round(data['18-45 years (Age)']/data['Total Individuals Vaccinated']*100,2)
    data['45-60 pct'] = round(data['45-60 years (Age)']/data['Total Individuals Vaccinated']*100,2)
    data['60+ pct'] = round(data['60+ years (Age)']/data['Total Individuals Vaccinated']*100,2)

    ## Create a new columns including both absolute numbers and percent to display as text on labels
    data['18-45'] = data['18-45 years (Age)'].apply(lambda x: '{:,}'.format(x)) + ' (' + data['18-45 pct'].apply(str) + '%)'
    data['45-60'] = data['45-60 years (Age)'].apply(lambda x: '{:,}'.format(x)) + ' (' + data['45-60 pct'].apply(str) + '%)'
    data['60+'] = data['60+ years (Age)'].apply(lambda x: '{:,}'.format(x)) + ' (' + data['60+ pct'].apply(str) + '%)'
    
    ## Creating Plots
    fig, ax = plt.subplots(figsize=(15,5))

    ## Plots for Daily Administered Doses of CoviShield and Covaxin over time
    ax.bar(mdates.date2num(data['Updated On'])-1.5,data['18-45 years (Age)'],label='18-45',width=1.5)
    ax.bar(mdates.date2num(data['Updated On'])+0,data['45-60 years (Age)'],label='45-60',width=1.5)
    ax.bar(mdates.date2num(data['Updated On'])+1.5,data['60+ years (Age)'],label='60+',width=1.5)
    plt.legend(loc=1,facecolor='White')

    ## Formatting X-axis for appropriate Date format, Major and Minor Ticks along with axis limits
    date_form = DateFormatter("%d-%b-%y")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1,byweekday=(3)))
    _, y_top = ax.get_ylim()

    ## Using a loop to plot text for all the points of each of the 3 trend lines
    for i in range(len(data)):
            ax.text(mdates.date2num(data['Updated On'].iloc[i])-1.7, y_top/40 + data['18-45 years (Age)'].iloc[i], data['18-45'].iloc[i], rotation=90, size = 9)
            ax.text(mdates.date2num(data['Updated On'].iloc[i])-0.2, y_top/40 + data['45-60 years (Age)'].iloc[i], data['45-60'].iloc[i], rotation=90, size = 9)
            ax.text(mdates.date2num(data['Updated On'].iloc[i])+1.3, y_top/40 + data['60+ years (Age)'].iloc[i],data['60+'].iloc[i],   rotation=90, size = 9)

    ## Adding Labels
    ax.set_ylim(top=data[['18-45 years (Age)','45-60 years (Age)', '60+ years (Age)']].max().max()*1.6)
    fig.autofmt_xdate()
    plt.xlabel('Timeline',labelpad=10)
    plt.ylabel('Number of Individuals Vaccinated',labelpad=10)
    plt.title('Vaccinated Individuals Age wise composition of '+state)
    plt.show()

# %%
## Plotting number of daily vaccinated individuals by age group for India as a whole
individual_age_plotter('India')

# %%
## Plotting number of daily vaccinated individuals by age group for every state and UT
for state in states:
    individual_age_plotter(state)

# %%
