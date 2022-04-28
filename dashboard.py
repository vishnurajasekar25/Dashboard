#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:01:34 2022

@author: vishnu
"""
 #import libraries
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime 
 
#function definitions

def audience_simple(country):
    """Show top represented countries"""
    if country == 'US':
        return 'USA'
    elif country == 'IN':
        return 'India'
    else:
        return 'Other'
    
def style_negative(v, props=''):
    """ Style negative values in dataframe"""
    try: 
        return props if v < 0 else None
    except:
        pass
    
def style_positive(v, props=''):
    """Style positive values in dataframe"""
    try: 
        return props if v > 0 else None
    except:
        pass    
    
#loading the data
@st.cache
def load_data():
    df_agg = pd.read_csv('Aggregated_Metrics_By_Video.csv.xls').iloc[1:,:]
    df_agg_country_subs = pd.read_csv('Aggregated_Metrics_By_Country_And_Subscriber_Status.csv')
    df_comments = pd.read_csv('All_Comments_Final.csv')
    df_time = pd.read_csv('Video_Performance_Over_Time.csv')
    
    #engineer the data
    df_agg.columns =['Video','Video title','Video publish time','Comments added', 'Shares','Dislikes','Likes','Subscribers lost','Subscribers gained','RPM(USD)','CPM(USD)','Average % viewed','Average view duration','View','Watch time(hours)','Subscribers','Your estimated revenue(USD)','Impressions','Impressions ctr %']
    df_agg['Video publish time'] = pd.to_datetime(df_agg['Video publish time'])
    df_agg['Average view duration'] = df_agg['Average view duration'].apply(lambda x:datetime.strptime(x,'%H:%M:%S'))
    df_agg['Avg_duration_sec'] = df_agg['Average view duration'].apply(lambda x: x.second + x.minute*60 + x.hour*3600) 
    df_time['Date'] = pd.to_datetime(df_time['Date']) 
    return df_agg, df_agg_country_subs, df_comments, df_time


 #dataframes from the function
df_agg, df_agg_country_subs, df_comments, df_time = load_data()

#engineering the data
df_agg_copy = df_agg.copy()
metric_date_12mo = df_agg_copy['Video publish time'].max() - pd.DateOffset(months =12)
median_agg =df_agg_copy[df_agg_copy['Video publish time'] >= metric_date_12mo].median()
numeric_cols = np.array((df_agg_copy.dtypes == 'float64') | (df_agg_copy.dtypes == 'int64'))
                        
#building the dashboard
#st.set_page_config(page_title = "dashboard",  page_icon = ":bar_chart:", layout = "wide")
add_sidebar = st.sidebar.selectbox('Vidoe analysis', ('View metrics','Growth of the channel'))
if add_sidebar == 'View metrics':
    Views_by_Country = df_agg_country_subs[['Country Code', 'Views']].copy().sort_values(by='Views',ascending= False)
    x = Views_by_Country.groupby('Country Code').sum().sort_values(by='Views',ascending= False).head(10) 
    st.write(x.T)    
    df = df_agg_country_subs[['Video Title', 'Views', 'Average Watch Time', 'Country Code']].copy().sort_values(by='Views',ascending= False)
    US_df = df[df['Country Code']=='US']
    IN_df = df[df['Country Code']=='IN']
    GB_df = df[df['Country Code']=='GB']
    audience = pd.concat([US_df, IN_df, GB_df], axis=0, sort=False).sort_values(by='Views',ascending= False)
    plt.title("Views count from top 3 countries")
    plt.xlabel('Views')
    sns.barplot(x=audience['Country Code'] , y= audience['Views']);
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot() 
    
    #fig=px.pie(sub,values='Counts',names="Subscribed",title='Number of Subscribers',
          #color_discrete_sequence=["maroon","lightblue"])

#fig.update_xaxes(showgrid=False)

#fig.update_yaxes(showgrid=False, categoryorder='total ascending', ticksuffix=' ', showline=False)

#fig.update_traces(hovertemplate=None, marker=dict(line=dict(width=0)))

#fig.update_layout(margin=dict(t=80, b=0, l=70, r=40),hovermode="y unified",
      #            xaxis_title=' ', yaxis_title=" ", height=300,plot_bgcolor='#333', paper_bgcolor='#333',
#title_font=dict(size=25, color='#8a8d93', family="Lato, sans-serif"),
# font=dict(color='#8a8d93'),
                  #legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=0.5),
                  #hoverlabel=dict(bgcolor="black", font_size=13, font_family="Lato, sans-serif"))
                 
    
if add_sidebar == 'Growth of the channel':
    #Tried to make a pie chart but for some reason doesn't work. Let me know if there is a way
    #plt.pie(df_agg['Your estimated revenue(USD)'], labels= df_agg['Video title'], radius = 2, autopct='%1.1f%%', shadow=True, pctdistance =0.9)
    #fig.savefig('Estimated earning of top videos.png');
    #st.pyplot()
    subscribers = st.sidebar.multiselect("Is a Subscriber?",
                                         options = df_agg_country_subs["Is Subscribed"].unique(),
                                         default = df_agg_country_subs["Is Subscribed"].unique()
                                         )
    #Treid to input the dataframe query to filter out the unnecessary info according to the user
    # df_agg_country_subs_selection = df_agg_country_subs.query(
       # "Is Subscribed == @Is Subscribed")
    st.title(":bar_chart: Channel viewers")
    st.markdown("##")
    st.dataframe(df_agg_country_subs)
    left_column, mid_column, right_column = st.columns(3)
    with left_column:
        videos = tuple(df_agg['Video title'])
        video_select = st.selectbox('Pick a Video:', videos)
        agg_filtered = df_agg[df_agg['Video title'] == video_select]
        agg_sub_filtered = df_agg_country_subs[df_agg_country_subs['Video Title'] == video_select]
        agg_sub_filtered['Country'] = agg_sub_filtered['Country Code'].apply(audience_simple)
        agg_sub_filtered.sort_values('Is Subscribed', inplace= True) 
        fig = px.bar(agg_sub_filtered, x ='Views', y='Is Subscribed', color ='Country', orientation ='h')
        

        #Tried to make a real time tracking of how the current video did in comparison to the other videos
        # df_time_diff = pd.merge(df_time, df_agg.loc[:,['Video','Video publish time']], left_on ='External Video ID', right_on = 'Video')
        # df_time_diff['days_pub'] = (df_time_diff['Date'] - df_time_diff['Video publish time']).dt.days
        # date_12mo = df_agg['Video publish time'].max() - pd.DateOffset(months =12)
        # df_time_diff_yr = df_time_diff[df_time_diff['Video publish time'] >= date_12mo]
        # df_time_diff_yr = df_time_diff_yr.loc[:,~df_time_diff_yr.columns.duplicated()]
        # views_days = pd.pivot_table(df_time_diff_yr,index= 'days_pub',values ='Views', aggfunc = [np.mean,np.median,lambda x: np.percentile(x, 80),lambda x: np.percentile(x, 20)]).reset_index()
        # views_days.columns = ['days_pub','mean_views','median_views','80pct_views','20pct_views']
        # views_days = views_days[views_days['days_pub'].between(0,30)]
        # views_cumulative = views_days.loc[:,['days_pub','median_views','80pct_views','20pct_views']] 
        # views_cumulative.loc[:,['median_views','80pct_views','20pct_views']] = views_cumulative.loc[:,['median_views','80pct_views','20pct_views']].cumsum()
        # agg_time_filtered = df_time_diff[df_time_diff['Video Title'] == video_select]
        # first_30 = agg_time_filtered[agg_time_filtered['days_published'].between(0,30)]
        # first_30 = first_30.sort_values('days_published')
        # fig2 = go.Figure()
        # fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['20pct_views'],
        #                 mode='lines',
        #                 name='20th percentile', line=dict(color='purple', dash ='dash')))
        # fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['median_views'],
        #                     mode='lines',
        #                     name='50th percentile', line=dict(color='black', dash ='dash')))
        # fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['80pct_views'],
        #                     mode='lines', 
        #                     name='80th percentile', line=dict(color='royalblue', dash ='dash')))
        # fig2.add_trace(go.Scatter(x=first_30['days_published'], y=first_30['Views'].cumsum(),
        #                     mode='lines', 
        #                     name='Current Video' ,line=dict(color='firebrick',width=8)))
            
        # fig2.update_layout(title='View comparison first 30 days',
        #                 xaxis_title='Days Since Published',
        #                 yaxis_title='Cumulative views')
        
        # st.plotly_chart(fig2)
        
            
        with mid_column:    
            st.plotly_chart(fig)
            
        #Tried to show the subscriber increase or decrease to the video displayed in %
            df_agg_copy['Publish_date'] = df_agg_copy['Video publish time'].apply(lambda x: x.date())
            df_agg_copy_final = df_agg_copy.loc[:,['Subscribers']]
            
            df_agg_numeric_lst = df_agg_copy_final.median().index.tolist()
            df_to_pct = {}
            for i in df_agg_numeric_lst:
                df_to_pct[i] = '{:.1%}'.format
            
            st.dataframe(df_agg_copy_final.style.applymap(style_negative, props='color:red;').applymap(style_positive, props='color:green;').format(df_to_pct))
            
            
    
    #Hiding the streamlit displays and styling
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style,unsafe_allow_html=True)

