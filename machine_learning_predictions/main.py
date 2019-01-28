import pandas as pd
import numpy as np
import quandl

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from performance_metrics import *

import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import warnings
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop


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


def build_score_logreg(X_train, y_train, X_test, y_test, n_splits=10, random_seed=42):
    # Find optimal parameters, train model and make predictions
    cv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()
    logreg = LogisticRegression(random_state=random_seed)
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
    best_C = grid.best_params_['logreg__C']
    val_score =  grid.best_score_
    test_score = grid.score(X_test, y_test)

    df_final = X_test.copy()
    df_final['True Direction'] = y_test
    df_final['Predicted Direction'] = pred
    df_final['Predicted Returns'] = df_final['Predicted Direction'] * df_final['Log Returns']
    df_final['Cumulative Returns'] = df_final['Log Returns'].cumsum()
    df_final['Cumulative Predicted Returns'] = df_final['Predicted Returns'].cumsum()

    return df_final, best_C, val_score, test_score


def build_score_lgbm(X_train, y_train, X_test, y_test, n_splits=10, random_seed=42, 
                     max_depth=3, n_estimators=1000, num_leaves=5, subsample=0.8):
    # Find optimal parameters, train and make predictions
    cv = TimeSeriesSplit(n_splits=n_splits)
    lgbm = lgb.LGBMClassifier(random_state=random_seed, max_depth=max_depth, n_estimators=n_estimators, 
                              num_leaves=num_leaves, subsample=subsample, verbose=-1)

    param_grid = {'learning_rate': [0.0001,0.001,0.01,0.1],
                   'colsample_bytree': [0.1,0.25,0.5,0.75,1]}
    grid = GridSearchCV(lgbm, cv=cv, param_grid=param_grid, scoring='accuracy')
    grid.fit(X_train,y_train)
    pred = grid.predict(X_test)
    best_lr = grid.best_params_['learning_rate']
    best_colsample_bytree = grid.best_params_['colsample_bytree']
    val_score = grid.best_score_
    test_score = grid.score(X_test, y_test)

    df_final = X_test.copy()
    df_final['True Direction'] = y_test
    df_final['Predicted Direction'] = pred
    df_final['Predicted Returns'] = df_final['Predicted Direction'] * df_final['Log Returns']
    df_final['Cumulative Returns'] = df_final['Log Returns'].cumsum()
    df_final['Cumulative Predicted Returns'] = df_final['Predicted Returns'].cumsum()

    return df_final, best_lr, best_colsample_bytree, val_score, test_score


def build_score_lstm_model(X_train, y_train, X_test, y_test, units=10, dropout=0.2):
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

def calculate_metrics(df_test, best_conf):
    """
    Function to calculate different metrics of strategy performance
    """
    best_conf.append(accuracy_score(df_test['True Direction'], df_test['Predicted Direction'])) # Accuracy
    best_conf.append(gini_coef(df_test['True Direction'], df_test['Predicted Direction'])) # Gini coefficient
    best_conf.append(df_test['Predicted Returns'].sum()) # Total return
    best_conf.append(df_test['Predicted Returns'].mean()) # Average return per trade
    best_conf.append(df_test['Predicted Returns'].mean() * 252) # Return p.a.
    best_conf.append(Sharpe(df_test['Predicted Returns'])) # Sharpe ratio (annualized)
    best_conf.append(max_drawdown(cum_returns=df_test['Cumulative Predicted Returns'], invert=False)) # Maximum Drawdown
    best_conf.append(onesided_ttest(returns=df_test['Predicted Returns'])) # T-test (p-value)

    return best_conf

if __name__ == "__main__":
    quandl.ApiConfig.api_key = '-ssfpXQYbXiJG1FnjoUE'
    random_seed = 42

    # List if tickers    
    stocks = ['AAPL','AMZN','GOOGL','MSFT','JPM','JNJ','XOM','WMT','INTC','CVX',
              'IBM','PG','BA','KO','PEP','NVDA','MCD','AMGN','GE','HON']

    # Download stock data
    print('Reading stock data...')
    for stock in stocks:
        vars()[stock] = quandl.get('WIKI/' + stock, start_date='2006-01-01', end_date='2017-12-31')

    # Fit Logistic regression and predict for all stocks
    best_conf_log, best_conf_lgbm = [], []
    for stock in stocks:
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)

        print('Stock {}'.format(stock))
        # Preprocess data
        df = preprocess(globals()[stock])
        X_train, X_test, y_train, y_test = train_test_split(df)
        
        # Logistic regression
        print('Fit logistic regression')
        df_logreg, best_C, val_score, test_score = build_score_logreg(X_train, y_train, X_test, y_test,
                                                                    n_splits=10, random_seed=random_seed)
        
        print('Done!')
        # Prepare table for logreg test data     
        best_conf_log.append([stock, best_C, val_score, test_score])
        best_conf_log = calculate_metrics(df_logreg, best_conf_log)

        print('Fit LightGBM')
        # LightGBM
        df_lgbm, best_lr, best_colsample_bytree, val_score, test_score = build_score_lgbm(X_train, y_train, X_test, y_test, 
                                                                                          n_splits=10, random_seed=random_seed,
                                                                                          max_depth=3, n_estimators=1000, 
                                                                                          num_leaves=5, subsample=0.8)
        
        print('Done!')
        # Prepare table for lgbm test data     
        best_conf_lgbm.append([stock, best_lr, best_colsample_bytree, val_score, test_score])
        best_conf_lgbm = calculate_metrics(df_lgbm, best_conf_lgbm)

    # Save best parameters for Logistic regression in a csv file
    best_conf_log_cols = ['Ticker','C','Validation accuracy','Test accuracy', 'Accuracy', 'Gini coefficient', 'Total return',
                          'Average return per trade', 'Return p.a.', 'Sharpe ratio (annualized)', 'Maximum Draawdown', 'T-test (p-value)']
    best_conf_log = pd.DataFrame(data=best_conf_log, columns=best_conf_log_cols)
    best_conf_log.to_csv('../Data/Best_conf_log.csv')

    # Save best parameters for LightGBM in a csv file
    best_conf_lgbm_cols = ['Ticker','Loss ratio', 'Colsample by tree','Validation accuracy','Test accuracy', 'Accuracy', 'Gini coefficient', 
                           'Total return', 'Average return per trade', 'Return p.a.', 'Sharpe ratio (annualized)', 'Maximum Draawdown', 'T-test (p-value)']
    best_conf_lgbm = pd.DataFrame(data=best_conf_lgbm, columns=best_conf_lgbm_cols)
    best_conf_lgbm.to_csv('../Data/Best_conf_lgbm.csv')

    best_conf_lstm = []
    # Fit LSTM network and make predictions for all stocks
    for stock in stocks:
        print('Stock {}: fit LSTM'.format(stock))
        
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
                score = build_score_lstm_model(X_train, y_train, X_test, y_test, unit, drop)
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
        df_lstm = X_test.copy()
        df_lstm['True Direction'] = y_test
        df_lstm['Predicted Direction'] = pred
        df_lstm['Predicted Returns'] = df_lstm['Predicted Direction'] * df_lstm['Log Returns']
        df_lstm['Cumulative Returns'] = df_lstm['Log Returns'].cumsum()
        df_lstm['Cumulative Predicted Returns'] = df_lstm['Predicted Returns'].cumsum()
        
        best_conf_lstm.append([stock, best_units, best_drop, df_params['Accuracy'].max(), 
                               model.evaluate(X_test, y_test, batch_size=batch)[1]])

        best_conf_lstm = calculate_metrics(df_lstm, best_conf_lstm)     

        print('Done!')

    # Write best parameters of LSTM to a csv file
    best_conf_lstm_cols = ['Ticker','Units', 'Dropout','Validation accuracy','Test accuracy', 'Accuracy', 'Gini coefficient', 'Total return', 
                           'Average return per trade', 'Return p.a.', 'Sharpe ratio (annualized)', 'Maximum Draawdown', 'T-test (p-value)']
    best_conf_lstm = pd.DataFrame(data=best_conf_lstm, columns=best_conf_lstm_cols)
    best_conf_lstm.to_csv('../Data/Best_conf_lstm.csv')