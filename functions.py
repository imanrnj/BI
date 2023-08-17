from datetime import datetime, timedelta
from persiantools.jdatetime import JalaliDate, JalaliDateTime
import pandas as pd
import json
import random

def loadDb():
    '''
    This function loads data from the 'tempDb.json' file, which contains information about foods, food types, factors, factor items, and restaurant details.

    Parameters:

    None
    
    Returns:

    foods: A dictionary containing information about different foods.
    foodTypes: A dictionary with details of various food types.
    factorItems: A dictionary representing different factor items.
    factors: A dictionary containing factors categorized by their IDs.
    restaurant: A dictionary with details of the restaurant.
        
    '''
    with open('tempDb.json', encoding='utf-8') as f:
        data = json.load(f)
    foods = data['foods']
    foodTypes = data['foodTypes']
    factors = data['factors']
    factorItems = data['factorItems']
    restaurant = data['restaurant']
    factors = {d['id']: d for d in factors}
    return  foods,foodTypes,factorItems,factors,restaurant

def insertUsers(factor_df,N=1000):
    '''
    This function generates N phone numbers which starts with 08 and assign to factors without phone number

    Parameters:

    factor_df: factors_df: A dataframe containing factors categorized by their IDs.
    N: An int number for generating random phone numbers 
    
    Returns:

    factors_df: A dataframe containing factors with new users.
        
    '''
    phoneNumbers=[]
    genList=range(10000000,999999999)
    for i in range(N):
        phoneNumbers.append("08"+str(random.choice(genList)))
    for ind in factor_df.index:
        if factor_df['phone_number'][ind]=="":
            factor_df['phone_number'][ind]=random.choice(phoneNumbers)
    
    return factor_df

def LOG(message):
    '''
    This function prints log meassage with datetime
    
    Parameters:

    message: A string to print.
    
    Returns:

    None
        
    '''
    print('\033[0;31;47m LOG \033[0;30;47m',datetime.now().strftime("%m/%d/%Y %H:%M:%S.%f"),message,'\033[1;37;40m\n')
    
#     Print color text using ANSI code:
#     print("\033[1;37;40m \033[2;37:40m TextColour BlackBackground          TextColour GreyBackground                WhiteText ColouredBackground\033[0;37;40m\n")
# print("\033[1;30;40m Dark Gray      \033[0m 1;30;40m            \033[0;30;47m Black      \033[0m 0;30;47m               \033[0;37;41m Black      \033[0m 0;37;41m")
# print("\033[1;31;40m Bright Red     \033[0m 1;31;40m            \033[0;31;47m Red        \033[0m 0;31;47m               \033[0;37;42m Black      \033[0m 0;37;42m")
# print("\033[1;32;40m Bright Green   \033[0m 1;32;40m            \033[0;32;47m Green      \033[0m 0;32;47m               \033[0;37;43m Black      \033[0m 0;37;43m")
# print("\033[1;33;40m Yellow         \033[0m 1;33;40m            \033[0;33;47m Brown      \033[0m 0;33;47m               \033[0;37;44m Black      \033[0m 0;37;44m")
# print("\033[1;34;40m Bright Blue    \033[0m 1;34;40m            \033[0;34;47m Blue       \033[0m 0;34;47m               \033[0;37;45m Black      \033[0m 0;37;45m")
# print("\033[1;35;40m Bright Magenta \033[0m 1;35;40m            \033[0;35;47m Magenta    \033[0m 0;35;47m               \033[0;37;46m Black      \033[0m 0;37;46m")
# print("\033[1;36;40m Bright Cyan    \033[0m 1;36;40m            \033[0;36;47m Cyan       \033[0m 0;36;47m               \033[0;37;47m Black      \033[0m 0;37;47m")
# print("\033[1;37;40m White          \033[0m 1;37;40m            \033[0;37;40m Light Grey \033[0m 0;37;40m               \033[0;37;48m Black      \033[0m 0;37;48m")
    
     