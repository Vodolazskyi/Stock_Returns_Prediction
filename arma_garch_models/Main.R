library(Quandl)
library(tseries)
library(forecast)
library(rugarch)
library(ggplot2)
library(MLmetrics)
library(PerformanceAnalytics)

stocks = c('AAPL','AMZN','GOOGL','MSFT','JPM','JNJ','XOM','WMT','INTC','CVX','IBM',
           'PG','BA','KO','PEP','NVDA','MCD','AMGN','GE','HON')

path = 'University/Data/'
Quandl.api_key('-ssfpXQYbXiJG1FnjoUE')

# Downloading stock prices
for (stock in stocks){
  
  assign(paste(stock), Quandl(paste('WIKI/', stock, sep=''), start_date = "2006-01-01", end_date = "2017-12-31", type = "xts"))

}

# Fitting ARMA+GARCH models and forecasting returns for each stock
for (stock in stocks){
  
  # Calculate log returns and add additional columns
  data = get(stock)
  data$LogReturns = Return.calculate(data$`Adj. Close`, method = 'log')
  data$LogReturns[1] = 0
  data$TrueDirection = ifelse(data$LogReturns < 0, -1, 1)
  data$PredictedReturns = 0
  data$PredictedDirection = 0
  
  test_data = data['2015-01-01/']
  
  predictionsLength = length(data$LogReturns) - length(data$LogReturns['/2015-01-01'])
  
  # Find optimal arma model
  arimaModel = auto.arima(data$LogReturns['/2015-01-01'], ic = 'aic', stepwise = F)
  garch = ugarchspec(variance.model = list(garchOrder = c(1,1)),
                     mean.model = list(armaOrder = arimaorder(arimaModel)[c(1,3)], include.mean = TRUE),
                     distribution.model = 'std')
  
  # Make predictions 
  for (i in 0:(predictionsLength-1)){
    window = length(data$LogReturns['/2015-01-01'])+i
    trainReturns = data$LogReturns[1:window]
    
    garchFit = tryCatch(
      ugarchfit(
        garch, trainReturns, solver = 'hybrid'
      ), error=function(e) e, warning=function(w) w
    )
    
    if(is(garchFit, "warning")) {
      test_data$PredictedDirection[index(data$LogReturns[(window+1)])] = 1
      print(paste(stock, index(trainReturns[window]), 1, "warning", sep=","))
    } else {
      garchForecast = ugarchforecast(garchFit, n.ahead=1)
      prediction = garchForecast@forecast$seriesFor
      test_data$PredictedDirection[index(data$LogReturns[(window+1)])] = ifelse(prediction[1] < 0, -1, 1)
      print(paste(stock, colnames(prediction), ifelse(prediction[1] < 0, -1, 1), sep=","))
    }
  }
  
  # Calculate cumulative returns and save data to csv files
  test_data$PredictedReturns = test_data$LogReturns*test_data$PredictedDirection
  test_data$CumulativeReturns = cumsum(test_data$LogReturns)
  test_data$CumulativePredictedReturns = cumsum(test_data$PredictedReturns)
  assign(paste(stock, '_test', sep=''), test_data)
  write.zoo(get(paste(stock, '_test', sep='')), paste(path,stock,'_test.csv',sep=''), sep = ',')
}

# Calculating performance metrics for each stock
for(stock in stocks){
  
  test_data = get(paste(stock, '_test', sep=''))
  results = data.frame()
  results[1:2,'Ticker'] = stock
  results[1, 'Model'] = 'Benchmark (Buy&Hold)'
  results[2, 'Model'] = 'ARMA+GARCH'
  results[1, 'Accuracy'] = Accuracy(rep(1, length(test_data$LogReturns)),
                                    as.numeric(test_data$TrueDirection))
  results[2, 'Accuracy'] = Accuracy(as.numeric(test_data$PredictedDirection),
                                    as.numeric(test_data$TrueDirection))
  results[1, 'Gini coefficient'] = Gini(rep(1, length(test_data$LogReturns)),
                                        as.numeric(test_data$TrueDirection))
  results[2, 'Gini coefficient'] = Gini(as.numeric(test_data$PredictedDirection),
                                                   as.numeric(test_data$TrueDirection))
  results[1, 'Total return'] = Return.cumulative(test_data$LogReturns, geometric = F)
  results[2, 'Total return'] = Return.cumulative(test_data$LogReturns*test_data$PredictedDirection, geometric = F)
  results[1, 'Average return per trade'] = mean(test_data$LogReturns)
  results[2, 'Average return per trade']  = mean(test_data$PredictedReturns)
  results[1, 'Return p.a.'] = Return.annualized(test_data$LogReturns, scale = 252, geometric = F)
  results[2, 'Return p.a.'] = Return.annualized(test_data$LogReturns*test_data$PredictedDirection, scale = 252, geometric = F)
  results[1, 'Sharpe ratio (annualized)'] = SharpeRatio.annualized(test_data$LogReturns, scale = 252, geometric = F)
  results[2, 'Sharpe ratio (annualized)']  = SharpeRatio.annualized(test_data$PredictedReturns, scale = 252, geometric = F)
  results[1, 'Maximum Drawdown'] = maxDrawdown(test_data$LogReturns, geometric = F, invert = F)
  results[2, 'Maximum Drawdown']  = maxDrawdown(test_data$PredictedReturns, geometric = F, invert = F)
  results[1, 'T-test (p-value)'] = t.test(as.vector(test_data$LogReturns), alternative = 'greater')$p.value[1]
  results[2, 'T-test (p-value)']  = t.test(as.vector(test_data$PredictedReturns), alternative = 'greater')$p.value[1]
  assign(paste(stock,'_results',sep=''), results)
  write.table(get(paste(stock, '_results', sep='')), paste(path,stock,'_results.csv',sep=''), sep = ',', row.names = F)
}

# Merge all results in one table
total_results = data.frame()

for (stock in stocks){
  results = read.table(paste(path,stock, '_results.csv', sep=''), sep = ',', header = T)
  total_results = rbind(total_results, results)
}
