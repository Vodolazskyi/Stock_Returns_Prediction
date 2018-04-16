import pandas as pd
import numpy as np
import quandl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import ttest_1samp
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import warnings
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop

def max_drawdown(cum_returns, invert = True):
    """
    Function to calculate maximum drawdown
    """
    highest = [0]
    ret_idx = cum_returns.index
    drawdown = pd.Series(index = ret_idx)

    for t in range(1, len(ret_idx)):
        cur_highest = max(highest[t-1], cum_returns[t])
        highest.append(cur_highest)
        drawdown[t]= (1 + cum_returns[t]) / (1 + highest[t]) - 1
        
    if invert:
        return -1 * drawdown.min()
    else:
        return drawdown.min()

def onesided_ttest(returns, mean = 0, alternative = 'greater'):
    """
    Function returns p-value of one-sided t-test 
    """    
    ttest = ttest_1samp(returns, mean)
    if alternative == 'greater':
        if ttest[0] > 0:
            return ttest[1]/2
        else:
            return 1 - ttest[1]/2
    
    if alternative == 'less':
        if ttest[0] > 0:
            return 1 - ttest[1]/2
        else:
            return ttest[1]/2

def gini_coef(y_true, y_pred):
    """
    Function to calculate Gini coefficient
    """
    return 2*roc_auc_score(y_true, y_pred)-1

def Sharpe(returns, n=252):
    """
    Function to calculate Sharpe ratio
    """    
    sharpe = returns.mean() * np.sqrt(n) / returns.std()
    return sharpe

def train_test_split(df):
    """
    Function to split stock data into train and test data sets
    """    
    X_train, X_test = df.drop('Tomorrow Direction', axis=1)[:'2014-12-30'], \
                    df.drop('Tomorrow Direction', axis=1)['2014-12-31':]
    y_train, y_test = df['Tomorrow Direction'].loc[X_train.first_valid_index():'2014-12-30'], \
                    df['Tomorrow Direction']['2014-12-31':]
    
    return X_train, X_test, y_train, y_test

def lstm_train_test_split(df, window = 10):
    """
    Function to split stock data into train and test data sets for LSTM model
    """    
    X_train, X_test = df.drop('Tomorrow Direction', axis=1)[len(df[:'2006-01-03'])-window:len(df[:'2014-12-30'])], \
                    df.drop('Tomorrow Direction', axis=1)[(len(df[:'2014-12-31'])-window):len(df[:'2017-12-28'])]
    y_train, y_test = df['Tomorrow Direction'].loc['2006-01-03':'2014-12-30'], \
                    df['Tomorrow Direction']['2014-12-31':'2017-12-28']
    
    return X_train, X_test, y_train, y_test

def lstm_preprocess(df):
    """
    Function to preprocess stock data for LSTM model 
    """
    
    # Select only Adjusted columns
    df_copy = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
    df_copy = df_copy.rename(columns={'Adj. Open':'Open','Adj. High':'High','Adj. Low':'Low',
                                      'Adj. Close':'Close','Adj. Volume':'Volume'})
    
    # Compute log returns
    df_copy['Log Returns'] = np.log(df_copy['Close']) - np.log(df_copy['Close'].shift(1))
    df_copy['Log Returns'][0] = 0        
      
    # Add difference between today's Open and yesterday's Close; yeasterday's Open and Close
    df_copy['Open_Close'] = df_copy['Open'] - df_copy['Close'].shift(1)
    df_copy['Open_Lag_1'] = df_copy['Open'].shift(1)
    df_copy['Close_Lag_1'] = df_copy['Close'].shift(1)
    
    # Add month, day and day of week columns
    df_copy['Month'] = df_copy.index.month
    df_copy['Day'] = df_copy.index.day
    df_copy['Day_of_week'] = df_copy.index.dayofweek
    
    df_copy = df_copy.dropna()
    
    # Create a target column, which we want to predict
    df_copy['Tomorrow Direction'] = np.where(df_copy['Log Returns'].shift(-1) < 0, 0, 1)  
    
    return df_copy

def preprocess(df):
    """
    Function to preprocess stock data    
    """    
    # Select only Adjusted columns
    df_copy = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
    df_copy = df_copy.rename(columns={'Adj. Open':'Open','Adj. High':'High','Adj. Low':'Low',
                                      'Adj. Close':'Close','Adj. Volume':'Volume'})
    
    # Compute log returns
    df_copy['Log Returns'] = np.log(df_copy['Close']) - np.log(df_copy['Close'].shift(1))
    df_copy['Log Returns'][0] = 0        
    
    # Add difference between today's Open and yesterday's Close; yeasterday's Open and Close
    df_copy['Open_Close'] = df_copy['Open'] - df_copy['Close'].shift(1)
    df_copy['Open_Lag_1'] = df_copy['Open'].shift(1)
    df_copy['Close_Lag_1'] = df_copy['Close'].shift(1)
    
    # Add month, day and day of week columns
    df_copy['Month'] = df_copy.index.month
    df_copy['Day'] = df_copy.index.day
    df_copy['Day_of_week'] = df_copy.index.dayofweek
    
    df_copy = df_copy.dropna()
    
    # Create a target column, which we want to predict
    df_copy['Tomorrow Direction'] = np.where(df_copy['Log Returns'].shift(-1) < 0, -1, 1)

    return df_copy

def build_score_model(X_train, y_train, X_test, y_test, units=10, dropout=0.2):
    """
    Function to compute average validation accuracy for LSTM model
    """    
    timesteps = X_train.shape[1]
    features = X_train.shape[2]
    model = Sequential()
    model.add(LSTM(units, dropout=dropout, recurrent_dropout=dropout, input_shape=(timesteps, features)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['accuracy'])
    hist = model.fit(X_train, y_train, batch_size=10, epochs=10, validation_split=0.1, shuffle=False, verbose=0)
    
    acc = np.array(hist.history['val_acc']).mean()
    return acc

quandl.ApiConfig.api_key = '-ssfpXQYbXiJG1FnjoUE'

stocks = ['AAPL','AMZN','GOOGL','MSFT','JPM','JNJ','XOM','WMT','INTC','CVX','IBM','PG','BA','KO','PEP','NVDA','MCD','AMGN',
          'GE','HON']

# Download stock data
for stock in stocks:
    vars()[stock] = quandl.get('WIKI/' + stock, start_date='2006-01-01', end_date='2017-12-31')

# Fit Logistic regression and predict for all stocks
best_conf = []
for stock in stocks:
    # Preprocess data
    df = preprocess(globals()[stock])
    X_train, X_test, y_train, y_test = train_test_split(df)
    
    # Find optimal parameters, train model and make predictions
    cv = TimeSeriesSplit(n_splits=10)
    scaler = StandardScaler()
    logreg = LogisticRegression(random_state=42)
    pipeline = Pipeline([
        ('scaler', scaler),
        ('logreg', logreg)
    ])

    param_grid = {
        'logreg__C': np.linspace(0.001,1,20)
    }

    grid = GridSearchCV(pipeline, cv=cv, param_grid=param_grid, scoring='accuracy')
    grid.fit(X_train,y_train)
    pred = grid.predict(X_test)
    
    # Prepare table for test data
    df_test = globals()[stock][['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
    df_test = df_test.rename(columns={'Adj. Open':'Open','Adj. High':'High','Adj. Low':'Low',
                                          'Adj. Close':'Close','Adj. Volume':'Volume'})
    df_test['Log Returns'] = np.log(df_test['Close']) - np.log(df_test['Close'].shift(1))
    df_test = df_test['2015-01-01':]
    df_test['True Direction'] = np.where(df_test['Log Returns'] < 0, -1, 1)
    df_test['Predicted Direction'] = pred[:-1]
    df_test['Predicted Returns'] = df_test['Predicted Direction'] * df_test['Log Returns']
    df_test['Cumulative Returns'] = df_test['Log Returns'].cumsum()
    df_test['Cumulative Predicted Returns'] = df_test['Predicted Returns'].cumsum()
    
    best_conf.append([stock, grid.best_params_['logreg__C'], grid.best_score_, grid.score(X_test, y_test)])
    
    vars()[stock+'_log'] = df_test
    df_test.to_csv('Data/'+stock+'_log.csv')

# Save best parameters for Logistic regression in a csv file
best_conf = pd.DataFrame(data=best_conf, columns=['Ticker','C','Validation accuracy','Test accuracy'])
best_conf.to_csv('Data/Best_conf_log.csv')

# Calculate total results for ARMA-GARCH + Logistic regression
total_results = pd.DataFrame(columns=['Ticker','Model','Accuracy','Gini coefficient','Total return','Average return per trade',
                                      'Return p.a.','Sharpe ratio (annualized)','Maximum Drawdown','T-test (p-value)'])
for stock in stocks:
    
    df_test = globals()[stock+'_log']
    df_results = pd.read_csv('Data/' + stock + '_results_arma.csv')   
    
    df_results.loc[len(df_results), 'Model'] = 'Logistic regression'
    df_results.loc[(len(df_results)-1), 'Ticker'] = stock
    df_results.loc[(len(df_results)-1), 'Accuracy'] = accuracy_score(df_test['True Direction'], df_test['Predicted Direction'])
    df_results.loc[(len(df_results)-1), 'Gini coefficient'] = gini_coef(df_test['True Direction'], df_test['Predicted Direction'])
    df_results.loc[(len(df_results)-1), 'Total return'] = df_test['Predicted Returns'].sum()
    df_results.loc[(len(df_results)-1), 'Average return per trade'] = df_test['Predicted Returns'].mean()
    df_results.loc[(len(df_results)-1), 'Return p.a.'] = df_test['Predicted Returns'].mean() * 252
    df_results.loc[(len(df_results)-1), 'Sharpe ratio (annualized)'] = Sharpe(df_test['Predicted Returns'])
    df_results.loc[(len(df_results)-1), 'Maximum Drawdown'] = max_drawdown(cum_returns=df_test['Cumulative Predicted Returns'], invert=False)
    df_results.loc[(len(df_results)-1), 'T-test (p-value)'] = onesided_ttest(returns=df_test['Predicted Returns'])
    
    df_results.to_csv('Data/' + stock + '_results_log.csv', index=False)
    
    total_results = total_results.append(df_results)
    vars()[stock+'_results_log'] = df_results
    
total_results.reset_index(drop=True, inplace=True)
total_results.to_csv('Data/total_results_log.csv', index=False)

# Fit LightGBM and predict for all stocks
best_conf = []
for stock in stocks:
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    # Preprocess data
    df = preprocess(globals()[stock])
    X_train, X_test, y_train, y_test = train_test_split(df)

    # Find optimal parameters, train and make predictions
    cv = TimeSeriesSplit(n_splits=10)
    lgbm = lgb.LGBMClassifier(random_state=42, max_depth=3, n_estimators=1000, num_leaves=5, subsample=0.8)
    param_grid = {'learning_rate': [0.0001,0.001,0.01,0.1],
                   'colsample_bytree': [0.1,0.25,0.5,0.75,1]}
    grid = GridSearchCV(lgbm, cv=cv, param_grid=param_grid, scoring='accuracy')
    grid.fit(X_train,y_train)
    pred = grid.predict(X_test)
    
    # Prepare table for test data
    df_test = globals()[stock][['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
    df_test = df_test.rename(columns={'Adj. Open':'Open','Adj. High':'High','Adj. Low':'Low',
                                          'Adj. Close':'Close','Adj. Volume':'Volume'})
    df_test['Log Returns'] = np.log(df_test['Close']) - np.log(df_test['Close'].shift(1))
    df_test = df_test['2015-01-01':]
    df_test['True Direction'] = np.where(df_test['Log Returns'] < 0, -1, 1)
    df_test['Predicted Direction'] = pred[:-1]
    df_test['Predicted Returns'] = df_test['Predicted Direction'] * df_test['Log Returns']
    df_test['Cumulative Returns'] = df_test['Log Returns'].cumsum()
    df_test['Cumulative Predicted Returns'] = df_test['Predicted Returns'].cumsum()
    
    best_conf.append([stock, grid.best_params_['learning_rate'], grid.best_params_['colsample_bytree'], 
                      grid.best_score_, grid.score(X_test, y_test)])
    
    vars()[stock+'_gbm'] = df_test
    df_test.to_csv('Data/'+stock+'_gbm.csv')
    print(stock + ' - Done!')

# Write the best parameters to a csv file
best_conf = pd.DataFrame(data=best_conf, columns=['Ticker','Learning rate','Subsample ratio of columns',
                                                  'Validation accuracy','Test accuracy'])
best_conf.to_csv('Data/Best_conf_gbm.csv')

# Calculate total results for ARMA-GARCH + Logistic regression + LightGBM
total_results = pd.DataFrame(columns=['Ticker','Model','Accuracy','Gini coefficient','Total return','Average return per trade',
                                      'Return p.a.','Sharpe ratio (annualized)','Maximum Drawdown','T-test (p-value)'])
for stock in stocks:
    
    df_test = globals()[stock+'_gbm']
    df_results = pd.read_csv('Data/' + stock + '_results_log.csv')   
    
    df_results.loc[len(df_results), 'Model'] = 'LightGBM'
    df_results.loc[(len(df_results)-1), 'Ticker'] = stock
    df_results.loc[(len(df_results)-1), 'Accuracy'] = accuracy_score(df_test['True Direction'], df_test['Predicted Direction'])
    df_results.loc[(len(df_results)-1), 'Gini coefficient'] = gini_coef(df_test['True Direction'], df_test['Predicted Direction'])
    df_results.loc[(len(df_results)-1), 'Total return'] = df_test['Predicted Returns'].sum()
    df_results.loc[(len(df_results)-1), 'Average return per trade'] = df_test['Predicted Returns'].mean()
    df_results.loc[(len(df_results)-1), 'Return p.a.'] = df_test['Predicted Returns'].mean() * 252
    df_results.loc[(len(df_results)-1), 'Sharpe ratio (annualized)'] = Sharpe(df_test['Predicted Returns'])
    df_results.loc[(len(df_results)-1), 'Maximum Drawdown'] = max_drawdown(cum_returns=df_test['Cumulative Predicted Returns'], invert=False)
    df_results.loc[(len(df_results)-1), 'T-test (p-value)'] = onesided_ttest(returns=df_test['Predicted Returns'])
    
    df_results.to_csv('Data/' + stock + '_results_gbm.csv', index=False)
    
    total_results = total_results.append(df_results)
    total_results.reset_index(drop=True, inplace=True)
    vars()[stock+'_results_gbm'] = df_results

total_results.to_csv('Data/total_results_gbm.csv', index=False)

# Fit LSTM network and make predictions for all stocks
for stock in stocks:
    
    # Preprocess data
    window = 10
    df = lstm_preprocess(globals()[stock])
    X_train, X_test, y_train, y_test = lstm_train_test_split(df, window=window)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Transform data to tensors
    X_train, y = [], []
    for i in range(window,X_train_scaled.shape[0]+1):
        X_train.append(X_train_scaled[i-window:i])
        y.append(y_train[i-window])
    X_train = np.array(X_train)
    y_train = np.array(y)
    
    X_test, y_t = [], []
    for i in range(window,X_test_scaled.shape[0]+1):
        X_test.append(X_test_scaled[i-window:i])
        y_t.append(y_test[i-window])
    X_test = np.array(X_test)
    y_test = np.array(y_t)
    
    # Train LSTM network for different input parameters
    df_params = []
    for unit in [5,10,20]:
        for drop in [0.2,0.4,0.6]:
            score = build_score_model(X_train, y_train, X_test, y_test, unit, drop)
            df_params.append([unit,drop,score])
    df_params = pd.DataFrame(data=df_params, columns=['Units','Dropout','Accuracy'])
    
    # Select the best parameters
    best_units = df_params[df_params['Accuracy'] == df_params['Accuracy'].max()]['Units'].values[0]
    best_drop = df_params[df_params['Accuracy'] == df_params['Accuracy'].max()]['Dropout'].values[0]
    
    # Train LSTM network on training data with best parameters
    timesteps = X_train.shape[1]
    features = X_train.shape[2]
    batch = 10
    model = Sequential()
    model.add(LSTM(best_units, dropout=best_drop, recurrent_dropout=best_drop, input_shape=(timesteps, features)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch, epochs=10, shuffle=False, verbose=1)

    # Make predictions
    pred = np.where(model.predict_classes(X_test, batch_size=batch) == 0, -1, 1)
    
    # Prepare table for test data
    df_test = globals()[stock][['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
    df_test = df_test.rename(columns={'Adj. Open':'Open','Adj. High':'High','Adj. Low':'Low',
                                          'Adj. Close':'Close','Adj. Volume':'Volume'})
    df_test['Log Returns'] = np.log(df_test['Close']) - np.log(df_test['Close'].shift(1))
    df_test = df_test['2015-01-01':]
    df_test['True Direction'] = np.where(df_test['Log Returns'] < 0, -1, 1)
    df_test['Predicted Direction'] = pred
    df_test['Predicted Returns'] = df_test['Predicted Direction'] * df_test['Log Returns']
    df_test['Cumulative Returns'] = df_test['Log Returns'].cumsum()
    df_test['Cumulative Predicted Returns'] = df_test['Predicted Returns'].cumsum()
    
    best_conf.append([stock, best_units, best_drop, df_params['Accuracy'].max(), 
                      model.evaluate(X_test, y_test, batch_size=batch)[1]])
    
    vars()[stock+'_lstm'] = df_test
    df_test.to_csv('Data/'+stock+'_lstm.csv')
    print(stock + ' - Done!')

# Write best parameters to a csv file
best_conf = pd.DataFrame(data=best_conf, columns=['Ticker','Units','Dropout','Validation accuracy','Test accuracy'])
best_conf.to_csv('Data/Best_conf_lstm.csv')

# Calculate total results for ARMA-GARCH + Logistic regression + LightGBM + LSTM
total_results = pd.DataFrame(columns=['Ticker','Model','Accuracy','Gini coefficient','Total return','Average return per trade',
                                      'Return p.a.','Sharpe ratio (annualized)','Maximum Drawdown','T-test (p-value)'])
for stock in stocks:
    
    df_test = globals()[stock+'_lstm']
    df_results = pd.read_csv('Data/' + stock + '_results_gbm.csv')   
    
    df_results.loc[len(df_results), 'Model'] = 'LSTM'
    df_results.loc[(len(df_results)-1), 'Ticker'] = stock
    df_results.loc[(len(df_results)-1), 'Accuracy'] = accuracy_score(df_test['True Direction'], df_test['Predicted Direction'])
    df_results.loc[(len(df_results)-1), 'Gini coefficient'] = gini_coef(df_test['True Direction'], df_test['Predicted Direction'])
    df_results.loc[(len(df_results)-1), 'Total return'] = df_test['Predicted Returns'].sum()
    df_results.loc[(len(df_results)-1), 'Average return per trade'] = df_test['Predicted Returns'].mean()
    df_results.loc[(len(df_results)-1), 'Return p.a.'] = df_test['Predicted Returns'].mean() * 252
    df_results.loc[(len(df_results)-1), 'Sharpe ratio (annualized)'] = Sharpe(df_test['Predicted Returns'])
    df_results.loc[(len(df_results)-1), 'Maximum Drawdown'] = max_drawdown(cum_returns=df_test['Cumulative Predicted Returns'], invert=False)
    df_results.loc[(len(df_results)-1), 'T-test (p-value)'] = onesided_ttest(returns=df_test['Predicted Returns'])
    
    df_results.to_csv('Data/' + stock + '_results_lstm.csv', index=False)
    
    total_results = total_results.append(df_results)
    total_results.reset_index(drop=True, inplace=True)
    vars()[stock+'_results_lstm'] = df_results

total_results.to_csv('Data/total_results_lstm.csv', index=False)