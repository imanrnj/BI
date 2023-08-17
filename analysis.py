import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import plotly.express as px
import functions
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def basicDfForAnalysis(factor_df,factorItems_df,foods_df,foodTypes_df,refreshLoad=False):
    '''
    This function generates a dataframe containing basic data for analysising.

    Parameters:

    foods_df: A dataframe containing information about different foods.
    factorItems_df: A dataframe representing different factor items.
    factors_df: A dataframe containing factors categorized by their IDs.
    
    Returns:

    basicDfForAnalysis_df: A dataframe containing all basic data.
    basicDfForAnalysis.csv will be written in project folder 
    '''
    if not refreshLoad:
        if 'basicDfForAnalysis.csv' in os.listdir():
            functions.LOG("Analysis - reading basicDfForAnalysis.csv.")
            return pd.read_csv('basicDfForAnalysis.csv',index_col=0)
        else:
            functions.LOG("Analysis - basicDfForAnalysis.csv not found.")
        
    functions.LOG("Analysis - generating main dataframe.")
    results=pd.DataFrame()
    for ind in tqdm(factor_df.index):
        date = factor_df['date'][ind]
        factor_id= factor_df['id'][ind]
        dayname= factor_df['day_name'][ind]
        year=date[0:4]
        month=date[5:7]
        day=date[8:10]
        hour=factor_df['time'][ind][0:2]
        for index, row in factorItems_df.loc[factorItems_df['factor_id'] == factor_id].iterrows():      
            amount=row['amount']
            Foodname=row['name']
            fact_Sale=0
            for index2, row2 in foods_df.loc[foods_df['id'] == row['food_id']].iterrows():
                # Sale =row2['price']-row2['real_price']-row2['packing_price']
                # fact_Sale+=Sale*amount
                for index3 , row3 in foodTypes_df.loc[foodTypes_df['id']==  row2['food_type_id']].iterrows():
                    fact_Sale=0
                    #for index4 , row4 in foods_df.loc[foods_df['food_type_id'] == row3['id']].iterrows():
                    # Sale =row2['price']-row2['real_price']-row2['packing_price']
                    Sale =row2['price']
                    fact_Sale+=Sale*amount
                    foodTypes = row3['name']
                results=results.append({
                    'year':year,
                    'month':month,
                    'day':day,
                    'dayname':dayname,
                    'hour':hour,
                    'Food_Types':foodTypes,
                    'Food':Foodname,
                    'Food_id':row['food_id'],
                    'amount':amount,
                    'Sale':fact_Sale,}, ignore_index = True)
                    
    functions.LOG("Analysis - writing basicDfForAnalysis.csv.")
    results.to_csv( 'basicDfForAnalysis.csv', encoding='utf-8')
    
    return results            
                


def filter_df(filter_string,inputType,dfLists):
    
    
    functions.LOG("Analysis - filtering <filter_string> in dfLists inputType  ")
    filtered_list=[]
    for df in dfLists:
        
        filtered_inputTypeAggregateColumnbyGroupBy= df[df[inputType] == filter_string]
        
        filtered_list.append(filtered_inputTypeAggregateColumnbyGroupBy)
        
    return filtered_list  
    
    

def df_statistics(filtered_grouped_df,metric):
    
    '''
        This function generates a dataframe containing statistics of filtered_grouped_df .

    Parameters:

    filtered_grouped_df:filtered grouped by yearmonth, dayname or hour.
    metric: list of 1 str for making statistics
    Returns:

    df_statistics:filtered_grouped_df with statistics columns. These statistics are calculated from previous rows.
    
    '''
    first_quarters_column= []
    second_quarters_column= []
    third_quarters_column =[]
    maxs_column=[]
    stds_column=[]
    means_column=[]
    # metric=','.join(metrics)
    # print(metric)
        
    for i in range(len(filtered_grouped_df)):

        des = filtered_grouped_df[metric[0]][0:i].describe()
        first_quarter_column = des['25%']
        first_quarters_column.append(first_quarter_column)


        second_quarter_column= des['50%']
        second_quarters_column.append(second_quarter_column)

        third_quarter_column=des['75%']
        third_quarters_column.append(third_quarter_column)

        max_column = des['max']
        maxs_column.append(max_column)

        std_column = des['std']
        stds_column.append(std_column)
        
        means_column.append(des['mean'])


    filtered_grouped_df['first_quarters'] = first_quarters_column
    filtered_grouped_df['second_quarters'] = second_quarters_column
    filtered_grouped_df['third_quarters'] = third_quarters_column
    filtered_grouped_df['maxs'] = maxs_column
    filtered_grouped_df['stds'] = stds_column
    filtered_grouped_df['means'] = means_column
    filtered_grouped_df['means-2.5std'] =filtered_grouped_df['means']-2.5*filtered_grouped_df['stds']
    filtered_grouped_df['means+2.5std'] =filtered_grouped_df['means']+2.5*filtered_grouped_df['stds']



    # filtered_grouped_df['date']=filtered_grouped_df['year'].astype(str) +"-"+ filtered_grouped_df["month"]
    df_statistics=filtered_grouped_df.drop(['means'], axis=1)

    return(df_statistics)


def inputTypeAggregateColumnbyGroupBy(basicDfForAnalysis,inputType, aggregateColumn, groupBy):
    '''
        This function generates a dataframe containing analysis for all inputTypes(food or category) on aggregatecolumn  by groupBy.

        Parameters:

        basicDfForAnalysis: A dataframe containing basic data for analysising.
        inputType:
        aggregateColumn: 
        groupBy: A list

        Returns:

        newfunction_df: A dataframe containing aggregatecolumn Sum, aggregatecolumn Count, aggregatecolumn Mean, aggregatecolumn Sum%, aggregatecolumn count%, aggregatecolumn/Mean, Means, aggregatecolumn/Means for all inputTypes in each groupBy
        newfunction.csv will be written in project folder
    '''
    groupBy_Type =groupBy.copy()
    groupBy_Type.append(inputType)
    functions.LOG("Analysis - inputTypeAggregateColumnbyGroupBy.")
    functions.LOG("Analysis - "+inputType+" "+aggregateColumn+" by "+", ".join(groupBy))
    # گروه بندی بر اساس سال و ماه و غذا - میزان فروش و تعداد فروش هر غذا در ماه و سال به دست می اید
    results_grouped1=basicDfForAnalysis.groupby(groupBy_Type)
    results_grouped1=results_grouped1[aggregateColumn].agg(['sum','count']).reset_index()


    #گروه بندی بر اساس ماه و سال تا میزان فروش و تعداد فروش کل رستوران بر اساس ماه و سال به دست اید
    results_grouped2=basicDfForAnalysis.groupby(inputType)
    results_grouped2=results_grouped2[aggregateColumn].agg(['sum','count','mean']).reset_index()
    

    # ترکیب دو جدول بالا برای اینکه تجمیع شوند
    new_df = pd.merge(results_grouped1, results_grouped2,  how='inner', left_on=inputType, right_on = inputType)

    # ایجاد، محاسبه ستون های جدید
    new_df[aggregateColumn +'Sum%']=100*new_df['sum_x']/new_df['sum_y']
    new_df[aggregateColumn+'count%']=100*new_df['count_x']/new_df['count_y']
    new_df['Sum/Mean']= new_df['sum_x']/new_df['mean']
    
    
    functions.LOG("Analysis - preparing result dataframe.")
    # حذف ستون های اضافه و تغییر نام ستون ها
    inputTypeAggregateColumnbyGroupBy_df=new_df.drop(['sum_y', 'count_y'], axis=1).rename(columns={"sum_x": aggregateColumn+"Sum", "count_x": "Count","mean":aggregateColumn+ "Mean"})
    
    functions.LOG("Analysis - writing "+inputType+aggregateColumn+"by"+", ".join(groupBy)+".csv.")
    inputTypeAggregateColumnbyGroupBy_df.to_csv(inputType+aggregateColumn+"by"+", ".join(groupBy)+".csv", encoding='utf-8')
    
    return inputTypeAggregateColumnbyGroupBy_df

def lineChart(input_df,inputType,X,Y):
    X_len=len(X)
    X=X[-1]
    title = Y+" by "+X
    temp_df=input_df.copy()
    if X=='month' and X_len>1:
        temp_df['date']=temp_df['year'].astype(str) +"*"+ temp_df['month'].astype(str)
        X='date'
    if X=='day' and X_len>1:
        temp_df['date']=temp_df['year'].astype(str) +"*"+ temp_df['month'].astype(str)+"*"+ temp_df['day'].astype(str)
        X='date'
        
    
    fig = px.line(temp_df, x = X, y = Y,
              title = title,
              color=inputType)
    # fig.add_shape(type='line',
    #           x0=0,
    #           y0=temp_df['SaleMean'][0:1],
    #           x1=3000,
    #           y1=temp_df['SaleMean'][0:1],
    #           line=dict(color='Red'))
    fig.show()


def plotchart(newfunction_df,inputType, aggregateColumn,groupBy):
    
    if len(groupBy)>1 :
            
        newfunction_df['date']=newfunction_df[groupBy[0]].astype(str) +"-"+ newfunction_df[groupBy[1]]
        fig = px.line(newfunction_df,x=newfunction_df['date'],y=aggregateColumn ,title = newfunction_df[inputType][0:1] )
       
    else:
        fig = px.line(newfunction_df  ,x=groupBy,y=aggregateColumn, title = 'filtered for cake and dessert')
     
    fig.show()
    
def regressionpred(filtered_grouped_df,metric):
    
    '''
        This function generates a dataframe containing statistics of filtered_grouped_df .

    Parameters:

    filtered_grouped_df:filtered grouped by yearmonth, dayname or hour.
    metric: list of 1 str: please choose metric=['Sale Sum'] or metric=['Count']
    Returns:

    df_statistics:filtered_grouped_df with statistics columns. These statistics are calculated from previous rows.
    
    '''
    functions.LOG("regressionpred.")

    filtered_grouped_df=filtered_grouped_df.reset_index()
    filtered_grouped_df['month']=filtered_grouped_df['month'].astype(float) 

    realSaleSum=[filtered_grouped_df.loc[0,metric[0]]]
    realCount=[filtered_grouped_df.loc[0,metric[1]]]
    Month=[int((filtered_grouped_df.loc[0,['month']]).values)]


    for i in range(len(filtered_grouped_df)):

        x_train=filtered_grouped_df[['month']][0:i].values
        y_train1=filtered_grouped_df[metric[0]][0:i].values  
        y_train2=filtered_grouped_df[metric[1]][0:i].values  
        
        if i==0:
                predictionsSaleSum=['']
                predictionsCount=['']
                continue
        
        if i==len(filtered_grouped_df):
                break

        model1 = LinearRegression()
        model1.fit(x_train, y_train1)
        #print(i)
        x_pred=(filtered_grouped_df.loc[i,['month']]).values
        y_pred1= model1.predict([x_pred])
        #print(x_pred)
        predictionsSaleSum.append(float(y_pred1))
        realSaleSum.append(float((filtered_grouped_df.loc[i,[metric[0]]]).values))

        model2 = LinearRegression()
        model2.fit(x_train, y_train2)
        #print(i)
        x_pred=(filtered_grouped_df.loc[i,['month']]).values
        y_pred2= model2.predict([x_pred])

        predictionsCount.append(float(y_pred2))
        realCount.append(float((filtered_grouped_df.loc[i,[metric[1]]]).values))

        Month.append(int(x_pred))

    Df={"Month":Month,"Sale Sum":realSaleSum,"Prediction Sale Sum":predictionsSaleSum,"Count":realCount,"Prediction Count":predictionsCount}
    Regression_df=pd.DataFrame(Df)
    functions.LOG("Analysis - writing Regression.csv.")
    Regression_df.to_csv('Regression_df.csv', encoding='utf-8')
    return Regression_df





    
#-------------------------------------------------------------------------------------------
#-------------------------------- OLD Functions --------------------------------------------
#---------------------------------------------- --------------------------------------------

def foodSalesbyYearMonth(basicDfForAnalysis):
    '''
    This function generates a dataframe containing analysis for all foods in each month of years.

    Parameters:

    basicDfForAnalysis: A dataframe containing basic data for analysising.

    
    Returns:

    foodSalesbyYearMonth_df: A dataframe containing Sale Sum, Sale Count, Sale Mean, Sale Sum%, Sale count%, Sum/Mean, Means, Sum/Means for all foods in each month of years.
    foodSalesbyYearMonth.csv will be written in project folder
    '''
    

    functions.LOG("Analysis - foodSalesbyYearMonth.")
    # گروه بندی بر اساس سال و ماه و غذا - میزان فروش و تعداد فروش هر غذا در ماه و سال به دست می اید
    results_grouped1=basicDfForAnalysis.groupby(['year','month','Food'])
    results_grouped1=results_grouped1['Sale'].agg(['sum','count']).reset_index()


    #گروه بندی بر اساس ماه و سال تا میزان فروش و تعداد فروش کل رستوران بر اساس ماه و سال به دست اید
    results_grouped2=basicDfForAnalysis.groupby(['year','month'])
    results_grouped2=results_grouped2['Sale'].agg(['sum','count','mean']).reset_index()
    

    # ترکیب دو جدول بالا برای اینکه تجمیع شوند
    new_df = pd.merge(results_grouped1, results_grouped2,  how='inner', left_on=['year','month'], right_on = ['year','month'])

    # ایجاد، محاسبه ستون های جدید
    new_df['Sale Sum%']=100*new_df['sum_x']/new_df['sum_y']
    new_df['Sale count%']=100*new_df['count_x']/new_df['count_y']
    new_df['Sum/Mean']= new_df['sum_x']/new_df['mean']
    
    
    functions.LOG("Analysis - preparing result dataframe.")
    # حذف ستون های اضافه و تغییر نام ستون ها
    foodSalesbyYearMonth_df=new_df.drop(['sum_y', 'count_y'], axis=1).rename(columns={"sum_x": "Sale Sum", "count_x": "Count","mean":"Sale Mean"})
    
    functions.LOG("Analysis - writing foodSalesbyYearMonth.csv.")
    foodSalesbyYearMonth_df.to_csv('foodSalesbyYearMonth.csv', encoding='utf-8')
    
    return foodSalesbyYearMonth_df

def foodSalesbyYearMonthDay(basicDfForAnalysis):
    '''
    This function generates a dataframe containing analysis for all foods in each day of years.

    Parameters:

    basicDfForAnalysis: A dataframe containing basic data for analysising.

    
    Returns:

    foodSalesbyYearMonth_df: A dataframe containing Sale Sum, Sale Count, Sale Mean, Sale Sum%, Sale count%, Sum/Mean, Means, Sum/Means for all foods in each day of years.
    foodSalesbyYearMonth.csv will be written in project folder
    '''
    

    functions.LOG("Analysis - foodSalesbyYearMonthDay.")
    # گروه بندی بر اساس سال و ماه و غذا - میزان فروش و تعداد فروش هر غذا در ماه و سال به دست می اید
    results_grouped1=basicDfForAnalysis.groupby(['year','month','day','Food'])
    results_grouped1=results_grouped1['Sale'].agg(['sum','count']).reset_index()


    #گروه بندی بر اساس ماه و سال تا میزان فروش و تعداد فروش کل رستوران بر اساس ماه و سال به دست اید
    results_grouped2=basicDfForAnalysis.groupby(['year','month','day'])
    results_grouped2=results_grouped2['Sale'].agg(['sum','count','mean']).reset_index()

    results_grouped3=basicDfForAnalysis.groupby(['year','month','day'])
    results_grouped3=results_grouped3['dayname'].agg(['first']).reset_index()

    # ترکیب دو جدول بالا برای اینکه تجمیع شوند
    new_df = pd.merge(results_grouped1, results_grouped2,  how='inner', left_on=['year','month','day'], right_on = ['year','month','day'])
    new_df = pd.merge(new_df, results_grouped3,  how='inner', left_on=['year','month','day'], right_on = ['year','month','day'])

    # ایجاد، محاسبه ستون های جدید
    new_df['Sale Sum%']=100*new_df['sum_x']/new_df['sum_y']
    new_df['Sale count%']=100*new_df['count_x']/new_df['count_y']
    new_df['Sum/Mean']= new_df['sum_x']/new_df['mean']
    
    
    functions.LOG("Analysis - preparing result dataframe.")
    # حذف ستون های اضافه و تغییر نام ستون ها
    foodSalesbyYearMonthDay_df=new_df.drop(['sum_y', 'count_y'], axis=1).rename(columns={"sum_x": "Sale Sum", "count_x": "Count","mean":"Sale Mean","first":"DayName"})
    
    functions.LOG("Analysis - writing foodSalesbyYearMonthDay.csv.")
    foodSalesbyYearMonthDay_df.to_csv('foodSalesbyYearMonthDay.csv', encoding='utf-8')
    
    return foodSalesbyYearMonthDay_df

def foodSalesbyHour(basicDfForAnalysis):
    '''
    This function generates a dataframe containing analysis for all foods in each hour of day.

    Parameters:

    basicDfForAnalysis: A dataframe containing basic data for analysising.

    
    Returns:

    foodSalesbyHour_df: A dataframe containing Sale Sum, Sale Count, Sale Mean, Sale Sum%, Sale count%, Sum/Mean, Means, Sum/Means for all foods in each hour of day.
    foodSalesbyHour.csv will be written in project folder
    '''
    

    functions.LOG("Analysis - foodSalesbyHour.")
    # گروه بندی بر اساس سال و ماه و غذا - میزان فروش و تعداد فروش هر غذا در ماه و سال به دست می اید
    results_grouped1=basicDfForAnalysis.groupby(['hour','Food'])
    results_grouped1=results_grouped1['Sale'].agg(['sum','count']).reset_index()


    #گروه بندی بر اساس ماه و سال تا میزان فروش و تعداد فروش کل رستوران بر اساس ماه و سال به دست اید
    results_grouped2=basicDfForAnalysis.groupby(['hour'])
    results_grouped2=results_grouped2['Sale'].agg(['sum','count','mean']).reset_index()
    

    # ترکیب دو جدول بالا برای اینکه تجمیع شوند
    new_df = pd.merge(results_grouped1, results_grouped2,  how='inner', left_on=['hour'], right_on = ['hour'])

    # ایجاد، محاسبه ستون های جدید
    new_df['Sale Sum%']=100*new_df['sum_x']/new_df['sum_y']
    new_df['Sale count%']=100*new_df['count_x']/new_df['count_y']
    new_df['Sum/Mean']= new_df['sum_x']/new_df['mean']
    
    
    functions.LOG("Analysis - preparing result dataframe.")
    # حذف ستون های اضافه و تغییر نام ستون ها
    foodSalesbyHour_df=new_df.drop(['sum_y', 'count_y'], axis=1).rename(columns={"sum_x": "Sale Sum", "count_x": "Count","mean":"Sale Mean"})
    
    functions.LOG("Analysis - writing foodSalesbyHour.csv.")
    foodSalesbyHour_df.to_csv('foodSalesbyHour.csv', encoding='utf-8')
    
    return foodSalesbyHour_df

def foodSalesbyDayname(basicDfForAnalysis):
    '''
    This function generates a dataframe containing analysis for all foods in each day of week.

    Parameters:

    basicDfForAnalysis: A dataframe containing basic data for analysising.

    
    Returns:

    foodSalesbyHour_df: A dataframe containing Sale Sum, Sale Count, Sale Mean, Sale Sum%, Sale count%, Sum/Mean, Means, Sum/Means for all foods in each day of week.
    foodSalesbyHour.csv will be written in project folder
    '''
    

    functions.LOG("Analysis - foodSalesbyDayname.")
    # گروه بندی بر اساس سال و ماه و غذا - میزان فروش و تعداد فروش هر غذا در ماه و سال به دست می اید
    results_grouped1=basicDfForAnalysis.groupby(['dayname','Food'])
    results_grouped1=results_grouped1['Sale'].agg(['sum','count']).reset_index()


    #گروه بندی بر اساس ماه و سال تا میزان فروش و تعداد فروش کل رستوران بر اساس ماه و سال به دست اید
    results_grouped2=basicDfForAnalysis.groupby(['dayname'])
    results_grouped2=results_grouped2['Sale'].agg(['sum','count','mean']).reset_index()
    

    # ترکیب دو جدول بالا برای اینکه تجمیع شوند
    new_df = pd.merge(results_grouped1, results_grouped2,  how='inner', left_on=['dayname'], right_on = ['dayname'])

    # ایجاد، محاسبه ستون های جدید
    new_df['Sale Sum%']=100*new_df['sum_x']/new_df['sum_y']
    new_df['Sale count%']=100*new_df['count_x']/new_df['count_y']
    new_df['Sum/Mean']= new_df['sum_x']/new_df['mean']
    
    
    functions.LOG("Analysis - preparing result dataframe.")
    # حذف ستون های اضافه و تغییر نام ستون ها
    foodSalesbyDayname_df=new_df.drop(['sum_y', 'count_y'], axis=1).rename(columns={"sum_x": "Sale Sum", "count_x": "Count","mean":"Sale Mean"})
    
    functions.LOG("Analysis - writingfoodSalesbyDayname.csv.")
    foodSalesbyDayname_df.to_csv('foodSalesbyDayname.csv', encoding='utf-8')
    
    return foodSalesbyDayname_df

def categorySalesbyYearMonth(basicDfForAnalysis):
    '''
    This function generates a dataframe containing analysis for all categories in each month of years.

    Parameters:

    foods_df: A dataframe containing information about different foods.
    foodTypes_df: A dataframe containing information about different categories.
    factorItems_df: A dataframe representing different factor items.
    factors_df: A dataframe containing factors categorized by their IDs.

    
    Returns:

    categorySalesbyYearMonth_df: A dataframe containing Sale Sum, Sale Count, Sale Mean, Sale Sum%, Sale count%, Sum/Mean, Means, Sum/Means for all categories in each month of years.
    categorySalesbyYearMonth.csv will be written in project folder
    '''
    
    functions.LOG("Analysis - categorySalesbyYearMonth")
    # گروه بندی بر اساس سال و ماه و غذا - میزان فروش و تعداد فروش هر غذا در ماه و سال به دست می اید
    results_grouped1=basicDfForAnalysis.groupby(['year','month','Food_Types'])
    results_grouped1=results_grouped1['Sale'].agg(['sum','count']).reset_index()


    #گروه بندی بر اساس ماه و سال تا میزان فروش و تعداد فروش کل رستوران بر اساس ماه و سال به دست اید
    results_grouped2=basicDfForAnalysis.groupby(['year','month'])
    results_grouped2=results_grouped2['Sale'].agg(['sum','count','mean']).reset_index()
    

    # ترکیب دو جدول بالا برای اینکه تجمیع شوند
    new_df = pd.merge(results_grouped1, results_grouped2,  how='inner', left_on=['year','month'], right_on = ['year','month'])

    # ایجاد، محاسبه ستون های جدید
    new_df['Sale Sum%']=100*new_df['sum_x']/new_df['sum_y']
    new_df['Sale count%']=100*new_df['count_x']/new_df['count_y']
    new_df['Sum/Mean']= new_df['sum_x']/new_df['mean']
    
    
    functions.LOG("Analysis - preparing result dataframe.")
    # حذف ستون های اضافه و تغییر نام ستون ها
    categorySalesbyYearMonth_df=new_df.drop(['sum_y', 'count_y'], axis=1).rename(columns={"sum_x": "Sale Sum", "count_x": "Count","mean":"Sale Mean"})
    
    functions.LOG("Analysis - writing categorySalesbyYearMonth.csv.")
    categorySalesbyYearMonth_df.to_csv('categorySalesbyYearMonth.csv', encoding='utf-8')
    
    return categorySalesbyYearMonth_df


def categorySalesbyYearMonthDay(basicDfForAnalysis):
    '''
    This function generates a dataframe containing analysis for all categories in each day of years.

    Parameters:

    basicDfForAnalysis: A dataframe containing basic data for analysising.

    
    Returns:

    categorySalesbyYearMonth_df: A dataframe containing Sale Sum, Sale Count, Sale Mean, Sale Sum%, Sale count%, Sum/Mean, Means, Sum/Means for all food typess in each day of years.
    categorySalesbyYearMonth.csv will be written in project folder
    '''
    

    functions.LOG("Analysis - categorySalesbyYearMonthDay.")
    # گروه بندی بر اساس سال و ماه و غذا - میزان فروش و تعداد فروش هر غذا در ماه و سال به دست می اید
    results_grouped1=basicDfForAnalysis.groupby(['year','month','day','Food_Types'])
    results_grouped1=results_grouped1['Sale'].agg(['sum','count']).reset_index()


    #گروه بندی بر اساس ماه و سال تا میزان فروش و تعداد فروش کل رستوران بر اساس ماه و سال به دست اید
    results_grouped2=basicDfForAnalysis.groupby(['year','month','day'])
    results_grouped2=results_grouped2['Sale'].agg(['sum','count','mean']).reset_index()

    results_grouped3=basicDfForAnalysis.groupby(['year','month','day'])
    results_grouped3=results_grouped3['dayname'].agg(['first']).reset_index()

    # ترکیب دو جدول بالا برای اینکه تجمیع شوند
    new_df = pd.merge(results_grouped1, results_grouped2,  how='inner', left_on=['year','month','day'], right_on = ['year','month','day'])
    new_df = pd.merge(new_df, results_grouped3,  how='inner', left_on=['year','month','day'], right_on = ['year','month','day'])

    # ایجاد، محاسبه ستون های جدید
    new_df['Sale Sum%']=100*new_df['sum_x']/new_df['sum_y']
    new_df['Sale count%']=100*new_df['count_x']/new_df['count_y']
    new_df['Sum/Mean']= new_df['sum_x']/new_df['mean']
    
    
    functions.LOG("Analysis - preparing result dataframe.")
    # حذف ستون های اضافه و تغییر نام ستون ها
    categorySalesbyYearMonthDay_df=new_df.drop(['sum_y', 'count_y'], axis=1).rename(columns={"sum_x": "Sale Sum", "count_x": "Count","mean":"Sale Mean","first":"DayName"})
    
    functions.LOG("Analysis - writing categorySalesbyYearMonthDay.csv.")
    categorySalesbyYearMonthDay_df.to_csv('categorySalesbyYearMonthDay.csv', encoding='utf-8')
    
    return categorySalesbyYearMonthDay_df



def categorySalesbyHour(basicDfForAnalysis):
    '''
    This function generates a dataframe containing analysis for all categories in each hour of day.

    Parameters:

    basicDfForAnalysis: A dataframe containing basic data for analysising.

    
    Returns:

    categorySalesbyHour_df: A dataframe containing Sale Sum, Sale Count, Sale Mean, Sale Sum%, Sale count%, Sum/Mean, Means, Sum/Means for all categories in each hour of day.
    categorySalesbyHour.csv will be written in project folder
    '''
    

    functions.LOG("Analysis - categorySalesbyHour.")
    # گروه بندی بر اساس سال و ماه و غذا - میزان فروش و تعداد فروش هر غذا در ماه و سال به دست می اید
    results_grouped1=basicDfForAnalysis.groupby(['hour','Food_Types'])
    results_grouped1=results_grouped1['Sale'].agg(['sum','count']).reset_index()


    #گروه بندی بر اساس ماه و سال تا میزان فروش و تعداد فروش کل رستوران بر اساس ماه و سال به دست اید
    results_grouped2=basicDfForAnalysis.groupby(['hour'])
    results_grouped2=results_grouped2['Sale'].agg(['sum','count','mean']).reset_index()
    

    # ترکیب دو جدول بالا برای اینکه تجمیع شوند
    new_df = pd.merge(results_grouped1, results_grouped2,  how='inner', left_on=['hour'], right_on = ['hour'])

    # ایجاد، محاسبه ستون های جدید
    new_df['Sale Sum%']=100*new_df['sum_x']/new_df['sum_y']
    new_df['Sale count%']=100*new_df['count_x']/new_df['count_y']
    new_df['Sum/Mean']= new_df['sum_x']/new_df['mean']
    
    
    functions.LOG("Analysis - preparing result dataframe.")
    # حذف ستون های اضافه و تغییر نام ستون ها
    categorySalesbyHour_df=new_df.drop(['sum_y', 'count_y'], axis=1).rename(columns={"sum_x": "Sale Sum", "count_x": "Count","mean":"Sale Mean"})
    
    functions.LOG("Analysis - writing categorySalesbyHour.csv.")
    categorySalesbyHour_df.to_csv('categorySalesbyHour.csv', encoding='utf-8')
    
    return categorySalesbyHour_df

def categorySalesbyDayname(basicDfForAnalysis):
    '''
    This function generates a dataframe containing analysis for all categories in each day of week.

    Parameters:

    basicDfForAnalysis: A dataframe containing basic data for analysising.

    
    Returns:

    categorySalesbyHour_df: A dataframe containing Sale Sum, Sale Count, Sale Mean, Sale Sum%, Sale count%, Sum/Mean, Means, Sum/Means for all categories in each day of week.
    categorySalesbyHour.csv will be written in project folder
    '''
    

    functions.LOG("Analysis - categorySalesbyDayname.")
    # گروه بندی بر اساس سال و ماه و غذا - میزان فروش و تعداد فروش هر غذا در ماه و سال به دست می اید
    results_grouped1=basicDfForAnalysis.groupby(['dayname','Food_Types'])
    results_grouped1=results_grouped1['Sale'].agg(['sum','count']).reset_index()


    #گروه بندی بر اساس ماه و سال تا میزان فروش و تعداد فروش کل رستوران بر اساس ماه و سال به دست اید
    results_grouped2=basicDfForAnalysis.groupby(['dayname'])
    results_grouped2=results_grouped2['Sale'].agg(['sum','count','mean']).reset_index()
    

    # ترکیب دو جدول بالا برای اینکه تجمیع شوند
    new_df = pd.merge(results_grouped1, results_grouped2,  how='inner', left_on=['dayname'], right_on = ['dayname'])

    # ایجاد، محاسبه ستون های جدید
    new_df['Sale Sum%']=100*new_df['sum_x']/new_df['sum_y']
    new_df['Sale count%']=100*new_df['count_x']/new_df['count_y']
    new_df['Sum/Mean']= new_df['sum_x']/new_df['mean']
    
    
    functions.LOG("Analysis - preparing result dataframe.")
    # حذف ستون های اضافه و تغییر نام ستون ها
    categorySalesbyDayname_df=new_df.drop(['sum_y', 'count_y'], axis=1).rename(columns={"sum_x": "Sale Sum", "count_x": "Count","mean":"Sale Mean"})
    
    functions.LOG("Analysis - writing categorySalesbyDayname.csv.")
    categorySalesbyDayname_df.to_csv('categorySalesbyDayname.csv', encoding='utf-8')
    
    return categorySalesbyDayname_df


def filtered_food(foodname,foodSalesbyYearMonth,foodSalesbyHour,foodSalesbyDayname,foodSalesbyYearMonthDay):
    '''
        This function generates 3 dataframes containing foodSalesbyYearMonth,foodSalesbyHour, foodSalesbyDayname and a specific foodname to  filtering these dataframes for one food.

    Parameters:

    foodname: name of a specific food
    foodSalesbyYearMonth: A dataframe representing foods grouped by year and month of sale.
    foodSalesbyHour: A dataframe representing foods grouped by hour of sale.
    foodSalesbyDayname: A dataframe representing foods grouped by day name of sale.
    foodSalesbyYearMonthDay: A dataframe representing foods grouped by year and month and day of sale.
    Returns:

    filtered_foodSalesbyYearMonth: A dataframe representing foods grouped by year and month of sale for a specific food.
    filtered_foodSalesbyHour: A dataframe representing foods grouped by hour of sale for a specific food.
    filtered_foodSalesbyDayname: A dataframe representing foods grouped by day name of sale for a specific food.
    filtered_foodSalesbyYearMonthDay: A dataframe representing foods grouped by day of sale for a specific food.
    '''
    
    filtered_foodSalesbyYearMonth=foodSalesbyYearMonth[foodSalesbyYearMonth['Food'] == foodname]
    filtered_foodSalesbyHour=foodSalesbyHour[foodSalesbyHour['Food']== foodname]
    filtered_foodSalesbyDayname=foodSalesbyDayname[foodSalesbyDayname['Food']== foodname]
    filtered_foodSalesbyYearMonthDay=foodSalesbyYearMonthDay[foodSalesbyYearMonthDay['Food'] == foodname]
    return filtered_foodSalesbyYearMonth,filtered_foodSalesbyHour,filtered_foodSalesbyDayname,filtered_foodSalesbyYearMonthDay


def filtered_category(category,categorySalesbyYearMonth,categorySalesbyHour,categorySalesbyDayname,categorySalesbyYearMonthDay):
    '''
        This function generates 3 dataframes containing categorySalesbyYearMonth,categorySalesbyHour, categorySalesbyDayname and a specific food category to  filtering these dataframes for one food.

    Parameters:

    category: name of a specific category
    categorySalesbyYearMonth: A dataframe representing foodTypes grouped by year and month of sale.
    categorySalesbyHour: A dataframe representing foodTypes grouped by hour of sale.
    categorySalesbyDayname: A dataframe representing foodTypes grouped by day name of sale.
    categorySalesbyYearMonthDay: A dataframe representing foodTypes grouped by year and month and day of sale
    
    Returns:

    filtered_categorySalesbyYearMonth: A dataframe representing a specific foodType grouped by year and month of sale for a specific food.
    filtered_categorySalesbyHour: A dataframe representing a specific foodType grouped by hour of sale .
    filtered_categorySalesbyDayname: A dataframe representing a specific foodType grouped by day name.
    filtered_categorySalesbyYearMonthDay: A dataframe representing a specific foodType grouped by day

    '''
    
    
    filtered_categorySalesbyYearMonth=categorySalesbyYearMonth[categorySalesbyYearMonth['Food_Types'] == category]
    filtered_categorySalesbyYearMonthDay=categorySalesbyYearMonthDay[categorySalesbyYearMonthDay['Food_Types'] == category]
    filtered_categorySalesbyHour=categorySalesbyHour[categorySalesbyHour['Food_Types']== category]
    filtered_categorySalesbyDayname=categorySalesbyDayname[categorySalesbyDayname['Food_Types']== category]
    return filtered_categorySalesbyYearMonth,filtered_categorySalesbyHour,filtered_categorySalesbyDayname, filtered_categorySalesbyYearMonthDay



# def foodSalesbyYearMonth(factor_df,factorItems_df,foods_df):
#     '''
#     This function generates a dataframe containing analysis for all categories in each month of years.

#     Parameters:

#     foods_df: A dataframe containing information about different foods.
#     foodTypes_df: A dataframe containing information about different categories.
#     factorItems_df: A dataframe representing different factor items.
#     factors_df: A dataframe containing factors categorized by their IDs.

    
#     Returns:

#     categorySalesbyYearMonth_df: A dataframe containing Sale Sum, Sale Count, Sale Mean, Sale Sum%, Sale count%, Sum/Mean, Means, Sum/Means for all categories in each month of years.
#     categorySalesbyYearMonth.csv will be written in project folder
#     '''
    
#     functions.LOG("Analysis - generating main dataframe.")
#     results=pd.DataFrame()
#     for ind in tqdm(factor_df.index):
#         date = factor_df['date'][ind]
#         factor_id= factor_df['id'][ind]
#         year=date[0:4]
#         month=date[5:7]
#         for index, row in factorItems_df.loc[factorItems_df['factor_id'] == factor_id].iterrows():      
#             amount=row['amount']
#             Foodname=row['name']
#             fact_Sale=0
#             for index2, row2 in foods_df.loc[foods_df['id'] == row['food_id']].iterrows():
#                 Sale =row2['price']-row2['real_price']-row2['packing_price']
#                 fact_Sale+=Sale*amount
#                 results=results.append({'year':year,'month':month,'Food':Foodname,'Food_id':row['food_id'],'amount':amount,'price':row['price'],'Sale':fact_Sale,}, ignore_index = True)

#     functions.LOG("Analysis - grouping and merging dataframe.")
#     # گروه بندی بر اساس سال و ماه و غذا - میزان فروش و تعداد فروش هر غذا در ماه و سال به دست می اید
#     results_grouped1=results.groupby(['year','month','Food'])
#     results_grouped1=results_grouped1['Sale'].agg(['sum','count']).reset_index()


#     #گروه بندی بر اساس ماه و سال تا میزان فروش و تعداد فروش کل رستوران بر اساس ماه و سال به دست اید
#     results_grouped2=results.groupby(['year','month'])
#     results_grouped2=results_grouped2['Sale'].agg(['sum','count','mean']).reset_index()
    

#     # ترکیب دو جدول بالا برای اینکه تجمیع شوند
#     new_df = pd.merge(results_grouped1, results_grouped2,  how='inner', left_on=['year','month'], right_on = ['year','month'])

#     # ایجاد، محاسبه ستون های جدید
#     new_df['Sale Sum%']=100*new_df['sum_x']/new_df['sum_y']
#     new_df['Sale count%']=100*new_df['count_x']/new_df['count_y']
#     new_df['Sum/Mean']= new_df['sum_x']/new_df['mean']
    
    
#     functions.LOG("Analysis - preparing result dataframe.")
#     # حذف ستون های اضافه و تغییر نام ستون ها
#     foodSalesbyYearMonth_df=new_df.drop(['sum_y', 'count_y'], axis=1).rename(columns={"sum_x": "Sale Sum", "count_x": "Count","mean":"Sale Mean"})
    
#     functions.LOG("Analysis - writing foodSalesbyYearMonth.csv.")
#     foodSalesbyYearMonth_df.to_csv('foodSalesbyYearMonth.csv', encoding='utf-8')
    
#     return foodSalesbyYearMonth_df

# def categorySalesbyYearMonth(factor_df,factorItems_df,foods_df,foodTypes_df):
#     '''
#     This function generates a dataframe containing analysis for all categories in each month of years.

#     Parameters:

#     foods_df: A dataframe containing information about different foods.
#     foodTypes_df: A dataframe containing information about different categories.
#     factorItems_df: A dataframe representing different factor items.
#     factors_df: A dataframe containing factors categorized by their IDs.

    
#     Returns:

#     categorySalesbyYearMonth_df: A dataframe containing Sale Sum, Sale Count, Sale Mean, Sale Sum%, Sale count%, Sum/Mean, Means, Sum/Means for all categories in each month of years.
#     categorySalesbyYearMonth.csv will be written in project folder
#     '''
#     functions.LOG("Analysis - generating main dataframe.")
#     results=pd.DataFrame()
#     for ind in tqdm(factor_df.index):
#         date = factor_df['date'][ind]
#         factor_id= factor_df['id'][ind]
#         year=date[0:4]
#         month=date[5:7]
#         for index, row in factorItems_df.loc[factorItems_df['factor_id'] == factor_id].iterrows():      
#             amount=row['amount']
#             Foodname=row['name']
#             fact_Sale=0
#             for index2, row2 in foods_df.loc[foods_df['id'] == row['food_id']].iterrows():

#                 for index3 , row3 in foodTypes_df.loc[foodTypes_df['id']==  row2['food_type_id']].iterrows():
#                     fact_Sale=0
#                     #for index4 , row4 in foods_df.loc[foods_df['food_type_id'] == row3['id']].iterrows():
#                     Sale =row2['price']-row2['real_price']-row2['packing_price']
#                     fact_Sale+=Sale*amount
#                     foodTypes = row3['name']
#                     results = results.append({'year':year,'month':month ,'Food_Types': foodTypes , 'Sale':fact_Sale,}, ignore_index = True)

#     functions.LOG("Analysis - grouping and merging dataframe.")
#     # گروه بندی بر اساس سال و ماه و غذا - میزان فروش و تعداد فروش هر غذا در ماه و سال به دست می اید
#     results_grouped1=results.groupby(['year','month','Food_Types'])
#     results_grouped1=results_grouped1['Sale'].agg(['sum','count']).reset_index()


#     #گروه بندی بر اساس ماه و سال تا میزان فروش و تعداد فروش کل رستوران بر اساس ماه و سال به دست اید
#     results_grouped2=results.groupby(['year','month'])
#     results_grouped2=results_grouped2['Sale'].agg(['sum','count','mean']).reset_index()
    

#     # ترکیب دو جدول بالا برای اینکه تجمیع شوند
#     new_df = pd.merge(results_grouped1, results_grouped2,  how='inner', left_on=['year','month'], right_on = ['year','month'])

#     # ایجاد، محاسبه ستون های جدید
#     new_df['Sale Sum%']=100*new_df['sum_x']/new_df['sum_y']
#     new_df['Sale count%']=100*new_df['count_x']/new_df['count_y']
#     new_df['Sum/Mean']= new_df['sum_x']/new_df['mean']
    
    
#     functions.LOG("Analysis - preparing result dataframe.")
#     # حذف ستون های اضافه و تغییر نام ستون ها
#     categorySalesbyYearMonth_df=new_df.drop(['sum_y', 'count_y'], axis=1).rename(columns={"sum_x": "Sale Sum", "count_x": "Count","mean":"Sale Mean"})
    
#     functions.LOG("Analysis - writing categorySalesbyYearMonth.csv.")
#     categorySalesbyYearMonth_df.to_csv('categorySalesbyYearMonth.csv', encoding='utf-8')
    
#     return categorySalesbyYearMonth_df
