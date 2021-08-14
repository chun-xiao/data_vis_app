"""
Created on Fri Aug 13 21:48:35 2021

@author: Chun
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import base64
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Donut/Burger/Pancake Prediction App",layout='wide')


def plot_1year_prediction(plot_df, list_of_food, increase_rate, training_years, show_y):
    how_many_years_to_predict = 2
    annotate = 1
    fig, axes  = plt.subplots(1, 3, figsize= (20, 5), sharex = True, sharey = False)
    for i in range(3):
        food_type = list_of_food[i]

        if show_y == "total count":
            feature_interested = food_type+'_count'            
        elif show_y == "count per person":
            feature_interested = food_type+'_count_per_FTE'
            
        this_ax = axes[i] #TODO
        plot_df[feature_interested].head(11).plot(style='.-', ax=this_ax)
        if increase_rate >0:
            plot_df[feature_interested].iloc[10:10+how_many_years_to_predict].plot(style='.-', color = 'green', ms = 10, ax=this_ax)
        else:
            plot_df[feature_interested].iloc[10:10+how_many_years_to_predict].plot(style='.-', color = 'firebrick', ms = 10, ax=this_ax)

        pred_2021 = plot_df[feature_interested].iloc[10]
        count_2020 = plot_df[feature_interested].iloc[9]
        count_change_rate_prior_year = round((pred_2021 - count_2020)*100/count_2020, 2)
        if count_change_rate_prior_year > 0:
            count_change_rate_prior_year = '+'+str(count_change_rate_prior_year)

        if show_y == "total count":
            this_ax.set_ylabel("# total count")
            if annotate:
                this_ax.annotate('year: 2021\ntotal count: {:.0f}\nchange rate: {}%'.format(pred_2021, count_change_rate_prior_year),
                         xy = (2021, pred_2021) , textcoords='offset points', 
                         xytext=(-40,-60), # distance from text to points (x,y)
                         ha='center',
                         bbox=dict(boxstyle="round", alpha=0.1),
                         arrowprops=dict(arrowstyle= "fancy",fc="0.6", ec="none",
                          connectionstyle="angle3,angleA=0,angleB=-90"))
        elif show_y == "count per person":
            this_ax.set_ylabel("# count per person")
            this_ax.set_ylim(0,7)
            if annotate:
                if food_type == "DONUT":
                    this_ax.annotate('year: 2021\navg count: {: .2f}\nchange rate: {}%'.format(pred_2021, count_change_rate_prior_year),
                             xy = (2021, pred_2021) , textcoords='offset points', 
                             xytext=(-40,-60), # distance from text to points (x,y)
                             ha='center',
                             bbox=dict(boxstyle="round", alpha=0.1),
                             arrowprops=dict(arrowstyle= "fancy",fc="0.6", ec="none",
                              connectionstyle="angle3,angleA=0,angleB=-90"))
                else:
                    this_ax.annotate('year: 2021\navg count: {: .2f}\nchange rate: {}%'.format(pred_2021, count_change_rate_prior_year),
                             xy = (2021, pred_2021) , textcoords='offset points', 
                             xytext=(-40,40), # distance from text to points (x,y)
                             ha='center',
                             bbox=dict(boxstyle="round", alpha=0.1),
                             arrowprops=dict(arrowstyle= "fancy",fc="0.6", ec="none",
                              connectionstyle="angle3,angleA=0,angleB=-90"))
        this_ax.set_xlabel("")

        this_ax.set_title(food_type)
    return fig

st.title(":chart_with_upwards_trend: Food Production Prediction")

from_year_to_predict = 2021

food_list = ['DONUT', 'BURGER', 'PANCAKE']
VSP_honorary = {'DONUT': 0.75, 'BURGER': 0.89, 'PANCAKE': 0.92}
food_list.sort()



upload_file = st.file_uploader('', type="csv", accept_multiple_files=False)

if upload_file:
    df_FTE = pd.read_csv(upload_file, index_col = 'year')
    
    st.subheader("Raw data exploration: shape {}".format(df_FTE.shape))

    st.write("---")


    st.title(":cookie: Burger, Donut, Pancake")
    
    st.sidebar.title(":pencil: Settings")
    increase_rate_input = st.sidebar.number_input("Assume: number of staff yearly increase rate (%) in next 5 years:", -10, 10, 0, 1)
    increase_rate_input = increase_rate_input*0.01
    consider_VSP_honorary = st.sidebar.checkbox("Consider 2020 retains with 20% decay", 1)
    
    
    features_selected = st.sidebar.selectbox("Factors related to staff number", ["total FTE", "total & re FTE", "three FTE"], 0)
    
    years_for_training = st.sidebar.number_input("Training years", 2, 7, 7, 1)
    
    plot_radio = st.sidebar.radio("Display total counts or counts per FTE: ", ["total count", "count per person"])
    
    st.sidebar.write("---")
    st.sidebar.write("@2021 Aug by Chun")
    
    
    ##  --- prepare data
    all_deps = pd.DataFrame()
    for food_type in food_list:
        if features_selected == "total FTE":
            features = [food_type+'_total_FTE']
        elif features_selected == "total & re FTE":
            features = [food_type+'_re_FTE', food_type+'_total_FTE']
        elif features_selected == "three FTE":
            features =  [food_type+'_re_FTE', food_type+'_total_FTE', food_type+'_both_FTE']
        target = food_type+'_counts_mix' 
    
        one_food_type = df_FTE[[food_type+'_re_FTE', food_type+'_both_FTE', food_type+'_total_FTE']].T
        
        
        ###------ add the 78% of VSP have honorary appointments. for 2021 FTE
        # 20% VSP retains decay 
        if consider_VSP_honorary:
            one_food_type_original_2021 = one_food_type[from_year_to_predict].tolist()
            one_food_type_original_2021 = [x*(1+increase_rate_input) for x in one_food_type_original_2021]
    
            true_FTE_from_yr_to_predict = one_food_type[from_year_to_predict].copy()
            if food_type == "BURGER":
                VSP_2021_number = (one_food_type[from_year_to_predict-1] - true_FTE_from_yr_to_predict - 24.51) * VSP_honorary[food_type]
            else:
                VSP_2021_number = (one_food_type[from_year_to_predict-1] - true_FTE_from_yr_to_predict) * VSP_honorary[food_type]
            VSP_2021 = (true_FTE_from_yr_to_predict + VSP_2021_number).tolist()[:3]
            one_food_type[from_year_to_predict].iloc[:3] = VSP_2021
    
            for i in range(4):
                # 20% decay            
                one_food_type[from_year_to_predict+i+1] = (true_FTE_from_yr_to_predict+ VSP_2021_number*(0.8-i*0.2))*(1+increase_rate_input)
                
        else:
            for i in range(4):
                one_food_type[from_year_to_predict+i+1] = one_food_type[from_year_to_predict+i]*(1+increase_rate_input)
        
          
        
        #### ------------ the next 5 years prediction ----------
        
        
        one_food_type_y = df_FTE[target].head(-1).tolist()
        for i in [-5, -4, -3, -2,-1]:
        #### ----------------------------------------------------
        
        #### ------------------------------------------------
            one_food_type_X = one_food_type.T[features].head(i).tail(years_for_training)
            one_food_type_y_slice = one_food_type_y[(-1)*years_for_training:]
            reg = LinearRegression(normalize= True).fit(one_food_type_X, one_food_type_y_slice)
            if i != -1:
                X_test = one_food_type.T[features].iloc[i:i+1]
            else:
                X_test = one_food_type.T[features].iloc[i:]
            y_pred = reg.predict(X_test)
        
            one_food_type_y.append(y_pred.tolist()[0])
        
        one_food_type_T = one_food_type.T
        one_food_type_T[food_type+'_count'] = one_food_type_y
        one_food_type_T[food_type+'_count_per_FTE'] = (one_food_type_T[food_type+'_count'] / one_food_type_T[food_type+'_total_FTE']).round(2)
        one_food_type_T[food_type+'_count_change_rate_yearly'] = one_food_type_T[food_type+'_count'].pct_change()
        one_food_type_T[food_type+'_count_per_FTE_change_rate_yearly'] = one_food_type_T[food_type+'_count_per_FTE'].pct_change()
    
        one_food_type = one_food_type_T.T
        
        all_deps = all_deps.append(one_food_type)
        
    
    
    
    
    ### plot all deps ---------------
    plot_df = all_deps.T
    
    st.subheader("Factors selected: {}, assume yearly staff number increase: {}, trained by historical {} years".format(features_selected, increase_rate_input, years_for_training))
    fig = plot_1year_prediction(plot_df, food_list, increase_rate_input, years_for_training, plot_radio)
    st.pyplot(fig)
    

    
    download_clo1, download_clo2, download_clo3, download_clo4, col5  = st.beta_columns([1,3,3,5,8])
    download_clo1.write(":floppy_disk:")
    csv_pred = all_deps.to_csv(index=True)
    b64_pred = base64.b64encode(csv_pred.encode()).decode()  # some strings
    save_filename = "Product_pred_{}_VSP_{}_years_{}_FTEincrease.csv".format(consider_VSP_honorary, years_for_training, increase_rate_input)
    link_pred= f'<a href="data:file/csv;base64,{b64_pred}" download="{save_filename}">Save the full table</a>'
    download_clo2.markdown(link_pred, unsafe_allow_html=True)
    
    count_cols = [x for x in plot_df.columns if (('change' not in x) and (x.endswith('_count')))]
    temp = plot_df[count_cols].T.copy()
    temp = temp.round()
    csv_pred_count = temp.to_csv(index=True)
    b64_pred_count = base64.b64encode(csv_pred_count.encode()).decode()  # some strings
    save_count_filename = "Product_pred_count_{}_VSP_{}_years_{}_FTEincrease.csv".format(consider_VSP_honorary, years_for_training, increase_rate_input)
    link_count_pred= f'<a href="data:file/csv;base64,{b64_pred_count}" download="{save_count_filename}">Save the count table</a>'
    download_clo3.markdown(link_count_pred, unsafe_allow_html=True)
    
    
    count_perFTE_cols = [x for x in plot_df.columns if (('change' not in x) and ('_count_per_FTE' in x))]
    temp = plot_df[count_perFTE_cols].T.copy()
    temp = temp.round(2)
    csv_pred_countperFTE = temp.to_csv(index=True)
    b64_pred_countperFTE = base64.b64encode(csv_pred_countperFTE.encode()).decode()  # some strings
    save_countperFTE_filename = "Product_pred_countperFTE_{}_VSP_{}_years_{}_FTEincrease.csv".format(consider_VSP_honorary, years_for_training, increase_rate_input)
    link_countperFTE_pred= f'<a href="data:file/csv;base64,{b64_pred_countperFTE}" download="{save_countperFTE_filename}">Save the count per FTE table</a>'
    download_clo4.markdown(link_countperFTE_pred, unsafe_allow_html=True)
    
    st.write("---")
    
    st.subheader("Food sector data")
    cols_after_2014 = [x for x in all_deps.columns if x>2014 ]
    for i in range(len(food_list)):
        st.table(all_deps[cols_after_2014].iloc[i*7: (i+1)*7])
    
