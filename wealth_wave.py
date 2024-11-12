import pandas as pd
import numpy as np
from kiteconnect import KiteConnect, KiteTicker
import login
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import pytz
import pymongo
import os
import warnings
from dotenv import load_dotenv
import plotly.express as px
import matplotlib.pyplot as plt
import scipy.optimize as sc
import pandas_market_calendars as mcal

#make folder for storing the efficient frontier plots
if not os.path.exists('plots'):
    os.makedirs('plots')

#make folder for storing constructed portfolios
if not os.path.exists('portfolios'):
    os.makedirs('portfolios')

#make folder for storing the backtest results
if not os.path.exists('backtest_results'):
    os.makedirs('backtest_results')

#make folder for storing the trade log
if not os.path.exists('trade_log'):
    os.makedirs('trade_log')


warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

load_dotenv()
# Access environment variables
user_name = os.getenv("KITE_USER_NAME")
password = os.getenv("KITE_PASSWORD")
totp = os.getenv("KITE_TOTP")
api_key = os.getenv("KITE_API_KEY")
api_secret = os.getenv("KITE_API_SECRET")


print('Connecting to Kite API...')  

#connect to the kite api
kite = KiteConnect(api_key=api_key)
request_token = login.kiteLogin(user_name, password, totp, api_key)
data = kite.generate_session(request_token, api_secret)
kite.set_access_token(data["access_token"])

print('Connected to Kite API successfully!')

#get instruments from the kite api
instrument_dump = kite.instruments()   # get instruments dump from NSE
instrument_df = pd.DataFrame(instrument_dump)  # dump it to a dataframe
# print(instrument_df.columns)

#connect to the mongo db
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["wealth_wave"]
collection = db["monthly_portfolio"]
collection2 = db["backtest_results"]     


#get next trading day
def get_next_trading_day(date):
    nyse = mcal.get_calendar('XNYS')
    schedule = nyse.schedule(start_date=date, end_date=date + timedelta(days=10))
    next_date = mcal.date_range(schedule, frequency='1D')[1]  # Get the next trading day
    return next_date

#check if prior one year data is available for the stock
def check_prior_data(symbol,date):
    #get the instrument token
    instrument_token=getInstrumentToken(symbol)
    # print(instrument_token)
    #get the data for the stock for the prior one year
    to_date = datetime.strptime(date, '%Y-%m-%d')
    from_date = to_date - relativedelta(years=1)

    #get the next trading day of from date
    from_date = get_next_trading_day(from_date)
    # print(from_date)
    next_date = get_next_trading_day(from_date)
    
    #make date into 'YYYY-MM-DD' format
    from_date = from_date.strftime('%Y-%m-%d')
    next_date = next_date.strftime('%Y-%m-%d')

    #get the data for the stock for the from_date
    data = kite.historical_data(instrument_token, from_date, next_date, 'day')
    #change the data to dataframe
    data = pd.DataFrame(data)
    #check if the data is available
    if data.empty:
        return False
    return True

#function to get the instrument token
def getInstrumentToken(symbol):
    #return none if not found
    if len(instrument_df[(instrument_df['tradingsymbol']==symbol) & (instrument_df['exchange']=='NSE')]['instrument_token'].values)==0:
        return None
    return instrument_df[(instrument_df['tradingsymbol']==symbol) & (instrument_df['exchange']=='NSE')]['instrument_token'].values[0]

#function to get the desired environment
def get_desired_env(file_name):

    print('Getting the desired environment from the instrument dump...')
    #get the list of finserv stocks from instrument dump
    bse_stocks=pd.read_csv(file_name)
    #keep only Financial Services stocks
    bse_stocks=bse_stocks[bse_stocks['Sector Name']=='Financial Services']
    #reset the index
    bse_stocks.reset_index(drop=True,inplace=True)

    token_list=[]
    #get the security id and check if it is present in the instrument dump with exchange as NSE
    for index, row in bse_stocks.iterrows():
        symbol=row['Security Id']
        # print(symbol)
        token=getInstrumentToken(symbol)
        # print(token)
        if token:
            # print(symbol,token)
            token_list.append(token)
        else:
            # print('Token not found for symbol:',symbol)
            #remove the row from the dataframe
            bse_stocks.drop(index,inplace=True)

    #add token list to the dataframe
    bse_stocks['Token']=token_list
    #remove if Industry == Mutual Fund Scheme - ETF
    final_env=bse_stocks[bse_stocks['Industry']!='Mutual Fund Scheme - ETF'].copy()
    #reset the index
    final_env.reset_index(drop=True,inplace=True)
    #set the index to 'Security Id'
    final_env.set_index('Security Id',inplace=True)
    #first sort accourding to 'Security Id' and then 'Industry'
    final_env.sort_values(by=['Industry','Security Id'],inplace=True)

    print('Desired environment created successfully!')

    return final_env

#fuction to get zscore accourding to each column
def get_zscore(df):
    print('Calculating Z-score...')

    for column in df.columns:
        # print(df[column])
        df[column] = (df[column] - df[column].mean())/df[column].std()

    print('Z-score calculated successfully!')
    return df

#calulating momentum score of an individual stock using its historical data
def calculate_moment_score(df):

    if len(df) < 2:
        return np.nan
    # Get the pct change from the first to the last value
    pct_change = (df['close'].iloc[-1] / df['close'].iloc[0] - 1)
    # Get the std of pct change
    pct_change_std = np.std(df['close'].pct_change())
    if pct_change_std == 0:
        return np.nan
    score_value = pct_change / pct_change_std
    return score_value

#function to get the moment score of a environment of stocks for a given date
def get_moment_score(df,date):
        print('Calculating moment score...')

        moment_dict = {}
        # print(df.head(10))
        symbollist = df.index.to_list()
        # print(symbollist)
        end_date = datetime.strptime(date, '%Y-%m-%d')
        start_date = end_date - relativedelta(years=1)
        start_date = start_date.strftime('%Y-%m-%d')

        for symbol in symbollist:
            # print(symbol)

            try:
                stock_data = kite.historical_data(getInstrumentToken(symbol), start_date, end_date, "day")
                #make dataframe of the stock data
                stock_data = pd.DataFrame(stock_data)
                data_points = len(stock_data)
                six_month = stock_data.tail(data_points//2)
                data_points = len(six_month)
                three_month = six_month.tail(data_points//2)
                data_points = len(three_month)
                one_month = three_month.tail(data_points//3)
                data_points = len(one_month)
                two_week = one_month.tail(data_points//2)
                data_points = len(two_week)
                one_week = stock_data.tail(5)


                score_1y=calculate_moment_score(stock_data)
                socre_6m = calculate_moment_score(six_month)
                score_3m = calculate_moment_score(three_month)
                score_1m = calculate_moment_score(one_month)
                score_2w = calculate_moment_score(two_week)
                score_1w = calculate_moment_score(one_week)

                # print(socre_t,score_3m,score_1m,score_2w,score_1w,score_3d)

                row={'score_1y':score_1y,'score_6m':socre_6m,'score_3m':score_3m,'score_1m':score_1m,'score_2w':score_2w,'score_1w':score_1w}
                moment_dict[symbol] = row

            except:
                # print("Error in getting data for symbol: ", symbol)
                moment_dict[symbol] = {"Z_score_t": np.nan, "Z_score_3m": np.nan, "Z_score_1m": np.nan, "Z_score_2w": np.nan, "Z_score_1w": np.nan, "Z_score_3d": np.nan}

        moment_score_results = pd.DataFrame(moment_dict).T
        # print(moment_score_results)
        moment_score_results = get_zscore(moment_score_results)

        print('Moment score calculated successfully!')
        return moment_score_results

#calulate beta for each stock
def get_beta(stock_df, market_df):
    try:
        stock_df['Returns'] = stock_df['close'].pct_change()

        market_df['Market Returns'] = market_df['close'].pct_change()

        stock_df = stock_df.dropna()
        stock_df = stock_df.reset_index(drop=True)

        market_df = market_df.dropna()
        market_df = market_df.reset_index(drop=True)

        beta = np.cov(stock_df['Returns'], market_df['Market Returns'])[0][1] / np.var(market_df['Market Returns'])
    except:
        beta = np.nan
    # print(beta)
    return beta

#function to get annualized mean returns of a stock
def get_mean_returns(stock_df):
    if stock_df.empty or 'close' not in stock_df.columns or len(stock_df) < 2:
        return None
    stock_return = stock_df['close'].pct_change()
    stock_return_mean = np.mean(stock_return)*250
    return stock_return_mean

#function to calculate returns of a stock
def calculate_returns(stock_df):
    if stock_df.empty or 'close' not in stock_df.columns or len(stock_df) < 2:
        return None
    stock_return = stock_df['close'].iloc[-1] / stock_df['close'].iloc[0] - 1
    return stock_return

#function to calculate expected returns of a stock
def calculate_expected_returns(beta,market_return):
    rfr=0.07
    expected_return=rfr+beta*(market_return-rfr)
    return expected_return

#function to calculate risk of a stock
def calculate_risk_std(stock_df):
    if stock_df.empty or 'close' not in stock_df.columns or len(stock_df) < 2:
        return None
    stock_return = stock_df['close'].pct_change()
    stock_return_std = np.std(stock_return)
    return stock_return_std

#function to get beta score of all stocks in the environment
def get_beta_score(df,date):
    print('Calculating beta score...\n')

    # print(df.head(10))
    symbollist = df.index.to_list()
    # print(symbollist)
    end_date = datetime.strptime(date, '%Y-%m-%d')
    start_date = end_date - relativedelta(years=1)
    start_date = start_date.strftime('%Y-%m-%d')

    beta_dict={}
    i=0

    one_year_market_data = kite.historical_data(getInstrumentToken('NIFTY 50'), start_date, end_date, "day")
    #make dataframe of the stock data
    one_year_market_data = pd.DataFrame(one_year_market_data)
    data_points = len(one_year_market_data)
    six_month_market_data = one_year_market_data.tail(data_points//2)
    data_points = len(six_month_market_data)
    three_month_market_data = six_month_market_data.tail(data_points//2)
    data_points = len(three_month_market_data)
    one_month_market_data = three_month_market_data.tail(data_points//3)
    data_points = len(one_month_market_data)
    two_week_market_data = one_month_market_data.tail(data_points//2)
    data_points = len(two_week_market_data)
    one_week_market_data = one_month_market_data.tail(5)

    #calulate returns for each time period
    market_return_1y=calculate_returns(one_year_market_data)
    market_return_6m=calculate_returns(six_month_market_data)
    market_return_3m=calculate_returns(three_month_market_data)
    market_return_1m=calculate_returns(one_month_market_data)
    market_return_2w=calculate_returns(two_week_market_data)
    market_return_1w=calculate_returns(one_week_market_data)    

    market_returns={'market_return_1y':market_return_1y,'market_return_6m':market_return_6m,'market_return_3m':market_return_3m,'market_return_1m':market_return_1m,'market_return_2w':market_return_2w,'market_return_1w':market_return_1w}    

    data_points = len(one_year_market_data)
    for symbol in symbollist:
        # print(symbol)
        one_year_stock_data = kite.historical_data(getInstrumentToken(symbol), start_date, end_date, "day")
        #make dataframe of the stock data
        one_year_stock_data = pd.DataFrame(one_year_stock_data)
        data_points = len(one_year_stock_data)



        #6 month data
        six_month_stock_data = one_year_stock_data.tail(data_points//2)
        # data_points = len(six_month_stock_data)
        six_month_market_data = one_year_market_data.tail(data_points//2)
        data_points = len(six_month_market_data)
        #reset the index
        six_month_stock_data.reset_index(drop=True,inplace=True)
        six_month_market_data.reset_index(drop=True,inplace=True)

        #3 month data
        three_month_stock_data = six_month_stock_data.tail(data_points//2)
        # data_points = len(three_month_stock_data)
        three_month_market_data = six_month_market_data.tail(data_points//2)
        data_points = len(three_month_market_data)

        #1 month data
        one_month_stock_data = three_month_stock_data.tail(data_points//3)
        # data_points = len(one_month_stock_data)
        one_month_market_data = three_month_market_data.tail(data_points//3)
        data_points = len(one_month_market_data)

        #2 week data
        two_week_stock_data = one_month_stock_data.tail(data_points//2)
        # data_points = len(two_week_stock_data)
        two_week_market_data = one_month_market_data.tail(data_points//2)
        data_points = len(two_week_market_data)

        #1 week data
        one_week_stock_data = one_month_stock_data.tail(5)
        # data_points = len(one_week_stock_data)
        one_week_market_data = one_month_market_data.tail(5)
        data_points = len(one_week_market_data)

        one_year_beta=get_beta(one_year_stock_data,one_year_market_data)
        six_month_beta=get_beta(six_month_stock_data,six_month_market_data)
        three_month_beta=get_beta(three_month_stock_data,three_month_market_data)
        one_month_beta=get_beta(one_month_stock_data,one_month_market_data)
        two_week_beta=get_beta(two_week_stock_data,two_week_market_data)
        one_week_beta=get_beta(one_week_stock_data,one_week_market_data)

        #calculate returns for each time period
        stock_return_1y=calculate_returns(one_year_stock_data)
        stock_return_6m=calculate_returns(six_month_stock_data)
        stock_return_3m=calculate_returns(three_month_stock_data)
        stock_return_1m=calculate_returns(one_month_stock_data)
        stock_return_2w=calculate_returns(two_week_stock_data)
        stock_return_1w=calculate_returns(one_week_stock_data)

        #calulate expected returns
        expected_return_1y=calculate_expected_returns(one_year_beta,market_return_1y)
        expected_return_6m=calculate_expected_returns(six_month_beta,market_return_6m)
        expected_return_3m=calculate_expected_returns(three_month_beta,market_return_3m)
        expected_return_1m=calculate_expected_returns(one_month_beta,market_return_1m)
        expected_return_2w=calculate_expected_returns(two_week_beta,market_return_2w)
        expected_return_1w=calculate_expected_returns(one_week_beta,market_return_1w)

        risk=calculate_risk_std(one_year_stock_data)
        mean_return=get_mean_returns(one_year_stock_data)


        row={'beta_1y':one_year_beta,'beta_6m':six_month_beta,'beta_3m':three_month_beta,'beta_1m':one_month_beta,'beta_2w':two_week_beta,'beta_1w':one_week_beta,
             'stock_return_1y':stock_return_1y,'stock_return_6m':stock_return_6m,'stock_return_3m':stock_return_3m,'stock_return_1m':stock_return_1m,'stock_return_2w':stock_return_2w,'stock_return_1w':stock_return_1w,
             'expected_return_1y':expected_return_1y,'expected_return_6m':expected_return_6m,'expected_return_3m':expected_return_3m,'expected_return_1m':expected_return_1m,'expected_return_2w':expected_return_2w,'expected_return_1w':expected_return_1w,
             'risk':risk,'mean_return':mean_return}
        beta_dict[symbol] = row

    beta_df=pd.DataFrame(beta_dict).T
    print('Beta score calculated successfully\n')
    print(beta_df)
    return beta_df,market_returns

#function to calculate alpha score of a stock
def get_alpha_score(beta,market_return,stock_return):
    rfr=0.07
    expected_return=rfr+beta*(market_return-rfr)
    jensen_alpha=stock_return-expected_return
    return jensen_alpha

#function to calculate alpha score of all stocks in the environment
def calculate_alpha_score(beta_score_df,market_return):
    print('Calculating alpha score...\n')
    alpha_score={}
    for index,row in beta_score_df.iterrows():
        for time_period in ['1y','6m','3m','1m','2w','1w']:
            beta=row['beta_'+time_period]
            stock_return=row['stock_return_'+time_period]
            #market return is a dictionary
            market_returns=market_return['market_return_'+time_period]
            alpha=get_alpha_score(beta,market_returns,stock_return)
            beta_score_df.loc[index,'alpha_'+time_period]=alpha

            #calculate zscore for columns containing alpha
    zscore=get_zscore(beta_score_df[['alpha_1y','alpha_6m','alpha_3m','alpha_1m','alpha_2w','alpha_1w']])
    
    print('Alpha score calculated successfully!\n')
    return zscore

#funtion to normalize the computed values for each stock in the environment
def normalize(df):

    print('Normalizing the computed values...\n')
    for column in df.columns:
        for index in df.index:
            val=df.loc[index,column]
            if val==np.nan:
                df.loc[index,column]=0
            elif val>0:
                df.loc[index,column]=1+val
            else:
                df.loc[index,column]=1/(1-val)
    print('Normalization done successfully!\n')
    return df

#calulate pct change portfolio to get covariance matrix
def get_portfolio(final_list, date):
    portfolio=pd.DataFrame()
    for index, row in final_list.iterrows():
        # print(index)
        instrument_token = getInstrumentToken(index)
        # print(instrument_token)

        #get last one month data for each stock
        to_date = date
        to_date = datetime.strptime(to_date, '%Y-%m-%d')
        from_date = to_date - relativedelta(years=1)
        from_date = from_date.strftime('%Y-%m-%d')
        # print(from_date, to_date)
        data = kite.historical_data(instrument_token, from_date, to_date, 'day')
        #change the data to dataframe
        data = pd.DataFrame(data)

        # Check if 'close' column exists
        if 'close' not in data.columns:
            print(f"Error: 'close' column not found in data for {index}")
            #remove the row from the dataframe
            final_list.drop(index,inplace=True)
            continue

        data['pct_change'] = data['close'].pct_change()
        data = data[['date', 'pct_change']]
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')
        # print(data)

        #push the data to portfolio
        portfolio[index] = data['pct_change']
    #remove the first row of the portfolio
    portfolio = portfolio.iloc[1:]
    return portfolio

#calculate the covariance matrix
def get_covariance_matrix(final_list,date):
    print('Calculating covariance matrix...')
    portfolio = get_portfolio(final_list, date)
    covariance=portfolio.cov()
    covariance=covariance*250
    print('Covariance matrix calculated successfully!\n')
    return covariance


#creating efficient frontier
def efficient_frontier(final_list, covariance_matrix,date):

    print("Creating efficient frontier...")
    print('Creating efficient frontier using monte carlo simulation...\n')
    portfolio_returns = []
    portfolio_volatility = []

    sharpe_ratio = []   

    stock_weights = []  
    num_assets = len(final_list)
    num_portfolios = 100000

    np.random.seed(3)
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        stock_weights.append(weights)
        returns = np.dot(weights, final_list['weighted_return'])
        portfolio_returns.append(returns)
        portfolio_risk=np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        portfolio_volatility.append(portfolio_risk)

        #sharpe ratio
        sharpe = (returns-0.07)/portfolio_risk
        sharpe_ratio.append(sharpe)
    # Storing the portfolio values
    portfolio_final = {'Returns': portfolio_returns,
                'Volatility': portfolio_volatility,
                'Sharpe Ratio': sharpe_ratio}

    # Add an additional entry to the portfolio such that each indivudal weight is incorporated for its corresponding company
    for counter,symbol in enumerate(final_list.index):
        portfolio_final[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

    # make a nice dataframe of the extended dictionary
    df = pd.DataFrame(portfolio_final)

    print('Efficient frontier data accumulated succesfully using monte carlo simulation!\n')
    plt.style.use('fivethirtyeight')
    df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                    cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Returns')
    plt.title('Efficient Frontier')
    #save the plot to a file with name 'Efficient_Frontier_{date}.png' in plots folder
    plt.savefig(f'plots/Efficient_Frontier_{date}.png')

    print('Efficient frontier created successfully!\n')

#function to get the optimal portfolio
def portfolio_performance(weights, returns, cov_matrix):
    returns=np.sum(returns*weights)
    std=np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights)))
    return returns,std

# to get maximum sharpe ratio, we can minimize negative sharpe ratio
def negative_sharpe_ratio(weights,returns,cov_matrix,rfr=0.07):
    p_ret,p_std=portfolio_performance(weights,returns,cov_matrix)
    return -(p_ret-rfr)/p_std

def maxSR(returns,cov_matrix):
    rfr=0.07
    constraintSet=(0,1)
    num_assets=len(returns)
    args=(returns,cov_matrix,rfr)
    constraints=({'type':'eq','fun':lambda x: np.sum(x)-1})
    bound=constraintSet
    bounds=tuple(bound for asset in range(num_assets))
    #optimize using SLSQP (Sequential Least Squares Programming) algorithm
    result=sc.minimize(negative_sharpe_ratio, num_assets*[1./num_assets,],args=args,method='SLSQP',bounds=bounds,constraints=constraints)
    return result

def calculate_results(returns,covariance):
    
    print('Calculating results for the optimal portfolio...')

    maxSR_portfolio= maxSR(returns,covariance)
    maxSR_portfolio_weights = np.round(maxSR_portfolio.x*100,2)
    maxSR_portfolio_returns, maxSR_portfolio_risk = portfolio_performance(maxSR_portfolio_weights, returns, covariance)

    print('Results calculated successfully!\n')

    return maxSR_portfolio_weights, maxSR_portfolio_returns, maxSR_portfolio_risk

def wealth_wave_make_portfolio(date,final_env):
    print(f'Computing final score for the environment for {date}...\n')
    #compute moment score
    moment_score_df=get_moment_score(final_env,date)

    #compute alpha score

    #to compute alpha score we need beta score and market returns
    beta_score,market_returns=get_beta_score(final_env,date)
    alpha_score_df=calculate_alpha_score(beta_score,market_returns)

    #normalize the computed values
    moment_score_df=normalize(moment_score_df[['score_1y','score_6m','score_3m','score_1m','score_2w','score_1w']])
    alpha_score_df=normalize(alpha_score_df[['alpha_1y','alpha_6m','alpha_3m','alpha_1m','alpha_2w','alpha_1w']])

    print('computing final score...\n')
    #add column of alpha and moment score to the final score 
    final_score_df=pd.concat([moment_score_df,alpha_score_df],axis=1)
    print(final_score_df)

    #calulate final moment score as 0.3 to 1m 0.5 to 3m and 0.2 to 6m
    final_score_df['final_moment_score']=0.3*final_score_df['score_1m']+0.5*final_score_df['score_3m']+0.2*final_score_df['score_6m']


    #calulate final alpha score as 0.3 to 1m 0.5 to 3m and 0.2 to 6m
    final_score_df['final_alpha_score']=0.3*final_score_df['alpha_1m']+0.5*final_score_df['alpha_3m']+0.2*final_score_df['alpha_6m']

    #compute final score as 0.5 to final moment score and 0.5 to final alpha score
    final_score_df['final_score']=0.5*final_score_df['final_moment_score']+0.5*final_score_df['final_alpha_score']

    #sort the final score in descending order
    final_score_df.sort_values(by='final_score',ascending=False,inplace=True)
    print(final_score_df)
    print('Final score computed successfully!\n')
    #save the final score to a csv file
    final_score_df.to_csv('final_score.csv')
    print('Final score saved to final_score.csv')

    #select top 30 stocks
    final_list=final_score_df.head(30)
    final_list['weighted_return']=beta_score.loc[final_list.index,'mean_return']

    #remove stocks with negative expected 
    final_list=final_list[final_list['weighted_return']>0]
    #sort the final list by name of the stock
    final_list.sort_index(inplace=True)
    print(final_list)

    covariance_matrix=get_covariance_matrix(final_list, date)

    efficient_frontier(final_list, covariance_matrix,date)

    print('Calculating optimal portfolio with weights by maximizing the sharpe ratio...\n')
    result=maxSR(final_list['weighted_return'],covariance_matrix)
    print(result)
    print(np.round(result.x*100,2))
    print('Optimal portfolio calculated successfully!\n')


    maxSR_portfolio_weights, maxSR_portfolio_returns, maxSR_portfolio_risk = calculate_results(final_list['weighted_return'], covariance_matrix)

    print("Max Sharpe Ratio Portfolio Weights: ", maxSR_portfolio_weights)
    print("Max Sharpe Ratio Portfolio Returns: ", maxSR_portfolio_returns)
    print("Max Sharpe Ratio Portfolio Risk: ", maxSR_portfolio_risk)

    #add the weights to the final list
    final_list['maxSR_portfolio_weights']=maxSR_portfolio_weights

    print(final_list)
    #save the final list to a csv file in portfolios folder
    final_list.to_csv(f'portfolios/final_list_{date}.csv')

    #push the data to the mongo db
    #make dictionary of the final list
    final_list_dict=final_list.to_dict()
    row_mongo={'date':date,'final_list':final_list_dict,'maxSR_portfolio_returns':maxSR_portfolio_returns,'maxSR_portfolio_risk':maxSR_portfolio_risk}
    #add the row to the mongo db
    collection.insert_one(row_mongo)

    #select the stocks with maxSR_portfolio_weights
    maxSR_portfolio_stocks=final_list[final_list['maxSR_portfolio_weights']>0]
    print(maxSR_portfolio_stocks)

    return maxSR_portfolio_stocks

def backtest_portfolio(maxSR_portfolio_stocks,date):

    print('backtesting the portfolio for next one month...\n') 
    #buy stock on nexr trading day of date
    date = datetime.strptime(date, '%Y-%m-%d')
    next_date = get_next_trading_day(date)
    next_date = next_date.strftime('%Y-%m-%d')  

    returns = {}

    print('Computing returns for each stock in the portfolio...\n') 
    for index, row in maxSR_portfolio_stocks.iterrows():
        # print(index)
        instrument_token = getInstrumentToken(index)
        # print(instrument_token)

        #get next month data for each stock
        from_date = datetime.strptime(next_date, '%Y-%m-%d')
        to_date = from_date + relativedelta(months=1)
        # print(from_date, to_date)
        data = kite.historical_data(instrument_token, from_date, to_date, 'day')
        #change the data to dataframe
        data = pd.DataFrame(data)
        #get one month return by taking the last value and dividing by the first value
        one_month_return=data['close'].iloc[-1]/data['close'].iloc[0]-1
        # print(one_month_return)
        data_push={'one_month_return':one_month_return,'weight':row['maxSR_portfolio_weights']}
        returns[index]=data_push

    #make a dataframe of the returns
    returns_df=pd.DataFrame(returns).T
    print(returns_df)

    print('Returns computed successfully!\n')

    #multiply the returns with the weights
    returns_df['weighted_return']=returns_df['one_month_return']*returns_df['weight']
    returns_df['weighted_return']=returns_df['weighted_return']*100

    #get date in 'YYYY-MM-DD' string format
    date_push=date.strftime('%Y-%m-%d')
    #save the returns to a csv file in backtest_results folder
    returns_df.to_csv(f'backtest_results/returns_{date_push}.csv')
    
    #push the data to the mongo db
    #make dictionary of the returns_df
    returns_df_dict=returns_df.to_dict()
    row_mongo={'date':date,'returns_df':returns_df_dict,'total_return':returns_df['weighted_return'].sum()/100}
    #add the row to the mongo db
    collection2.insert_one(row_mongo)

    print(returns_df)

    print(returns_df['weighted_return'].sum()/100)
    return returns_df['weighted_return'].sum()/100

def wealth_wave(date,final_env):
    print('Starting Wealth Wave...\n')

    print('Creating Portfolio..')
    maxSR_portfolio_stocks=wealth_wave_make_portfolio(date,final_env)
    print('Portfolio created successfully!\n')

    print('Backtesting Portfolio...\n')
    returns=backtest_portfolio(maxSR_portfolio_stocks,date)
    print('Backtesting completed successfully!\n')

    print('Wealth Wave completed successfully!\n')
    return returns

final_env=get_desired_env('bse.csv')
#remove stocks with no prior data
date='2023-11-01'

print('Removing stocks with no prior data...\n')
for index, row in final_env.iterrows():
    if not check_prior_data(index,date):
        print('Stock with no prior data:',index)
        final_env.drop(index,inplace=True)
# final_env.reset_index(drop=True,inplace=True)
print('Stocks with no prior data removed successfully!\n')
print(final_env)
# run the code for the next 12 months
trade_log=[]
for i in range(12):
    date_start = datetime.strptime(date, '%Y-%m-%d')
    date_current = date_start + relativedelta(months=i)
    date_current = date_current.strftime('%Y-%m-%d')
    print('Running Wealth Wave for:',date_current)
    returns=wealth_wave(date_current,final_env)
    print('Wealth Wave completed for:',date_current)
    row={'date':date_current,'returns':returns}
    trade_log.append(row)

trade_log=pd.DataFrame(trade_log)
print(trade_log)

#add 100 to the value of the returns to get the final value
trade_log['returns']=trade_log['returns']+100
#divide the returns by 100 to get the final returns
trade_log['returns']=trade_log['returns']/100
#cumulative product of the returns
trade_log['comp_returns']=trade_log['returns'].cumprod()
print(trade_log)
#add percentage change of the comp_returns
trade_log['comp_returns_pct']=trade_log['comp_returns'].pct_change()
#calculate the std of the comp_returns
std=trade_log['comp_returns_pct'].std()
#calculate the annualized std
std=std*np.sqrt(12)
#calculate the annualized returns
annualized_returns=(trade_log['comp_returns'].iloc[-1]-1)
#sharpe ratio
sharpe_ratio=(annualized_returns-0.07)/std

print('Annualized Returns:',annualized_returns)
print('Annualized Std:',std)
print('Sharpe Ratio:',sharpe_ratio)

#add the annualized returns, std and sharpe ratio to the trade log
trade_log['annualized_returns']=annualized_returns
trade_log['annualized_std']=std
trade_log['sharpe_ratio']=sharpe_ratio

#plot the returns
plt.plot(trade_log['date'],trade_log['comp_returns'])
plt.xlabel('Date')
plt.ylabel('Returns')
plt.title('Wealth Wave Returns')
#save the plot in trade_log folder
plt.savefig('trade_log/Wealth_Wave_Returns.png')

#save the trade log to a csv file in trade_log folder
trade_log.to_csv('trade_log/trade_log.csv')

print('Trade log saved to trade_log.csv')
