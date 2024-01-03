rm(list = ls())
library(dplyr)
library(pracma)
library(keras)
library(tensorflow)
library(dplyr)
lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
  
  scaled_train = std_train *(fr_max -fr_min) + fr_min
  scaled_test = std_test *(fr_max -fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler= c(min =min(x), max = max(x))) )
}
invert_scaling = function(scaled, scaler, feature_range = c(0, 1)){
  min = scaler[1]
  max = scaler[2]
  t = length(scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(t)
  
  for( i in 1:t){
    X = (scaled[i]- mins)/(maxs - mins)
    rawValues = X *(max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}

communitySizes <- c(10,40)
observationErrors <- c("0.0","0.1","0.01")
trainingSizes <- c(10,30,50)

LSTMPredictions <- data.frame(CommunitySize = integer(),
                              ObservationError = double(),
                              TrainingSize = integer(), 
                              TimeSeriesID = integer(),
                              Time = integer(),
                              PopID = integer(),
                              LogDensity = double(),
                              RealValue = double())

LSTMPredictionsSD <- data.frame(CommunitySize = integer(),
                                 ObservationError = double(),
                                 TrainingSize = integer(), 
                                 TimeSeriesID = integer(),
                                 Time = integer(),
                                 LogDensity = double(),
                                 Lower95 = double(),
                                 Higher95 = double())

for(i in communitySizes){
  for(j in observationErrors){
        for(n in 3:5){
	  print(paste("Testing",i,j,n))
          currTimeSeries <- tryCatch(read.csv(paste(sep="",
                                           paste(sep="_","Data/timeSeries_communitySize",i,
                                                 "observationError",j,"rep",n),".csv"),
                                     header=FALSE,sep='\t'),
				     error=function(e){next})
          for(k in trainingSizes){
            for(m in 1:i){
              supervisedTS <-  lag_transform(t(currTimeSeries[m,]), 1)
              trainData = supervisedTS[2:k,]
              scaledData = scale_data(trainData,trainData,c(-1,1))
              x_train <- scaledData$scaled_train$`x-1`
              y_train <- scaledData$scaled_train$x
              dim(x_train) <- c(length(x_train), 1, 1)
              X_shape2 = dim(x_train)[2]
              X_shape3 = dim(x_train)[3]
              batch_size = 1
              units = 1
              Epochs = 50
              LSTMForecast = Reshape(rep(1,31*4),4,31)
              
              for(l in 1:4){
                model <- keras_model_sequential() 
                model%>%
                  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
                  layer_dense(units = 1)
                model %>% compile(
                  loss = 'mean_squared_error',
                  optimizer = optimizer_adam(),  
                  metrics = c('accuracy')
                )
                
                for(h in 1:Epochs){
                  model %>% fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
                  model %>% reset_states()
                }
                
                L = 30
                scaler = scaledData$scaler
                predictions = numeric(L+1)
                predictions[1] = y_train[k-1]
                scaledPredictions = numeric(L+1)
                scaledPredictions[1] = invert_scaling(predictions[1], scaler, c(-1,1))
                for(r in 1:L){
                  X = predictions[r]
                  dim(X) = c(1,1,1)
                  yhat = model %>% predict(X, batch_size=batch_size)
                  # invert scaling
                  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
                  # invert differencing
                  # store
                  scaledPredictions[r+1] <- yhat
                }
                LSTMForecast[l,] <- scaledPredictions
              }
              predictions <- apply(LSTMForecast,2,mean)
              low95 <- predictions - 1.96*apply(LSTMForecast,2,sd)/2
              hi95 <- predictions + 1.96*apply(LSTMForecast,2,sd)/2
              totalEntries <- length(predictions)
              LSTMPredictions <- rbind(LSTMPredictions,
                                        data.frame(CommunitySize = rep(i,totalEntries),
                                                   ObservationError = rep(as.double(j),totalEntries),
                                                   TrainingSize = rep(k,totalEntries), 
                                                   TimeSeriesID = rep(n,totalEntries),
                                                   Time = k:(k+30),
                                                   PopID = rep(m,totalEntries),
                                                   LogDensity = predictions,
                                                   RealValue = as.vector(t(currTimeSeries[m,k:(k+30)]))))
              LSTMPredictionsSD <- rbind(LSTMPredictionsSD,
                                          data.frame(CommunitySize = rep(i,totalEntries),
                                          ObservationError = rep(as.double(j),totalEntries),
                                          TrainingSize = rep(k,totalEntries), 
                                          TimeSeriesID = rep(n,totalEntries),
                                          Time = k:(k+30),
                                          PopID = rep(m,totalEntries),
                                          LogDensity = predictions,
                                          Lower95 = as.vector(t(low95)),
                                          Higher95 = as.vector(t(hi95))))
            }
          }
        }
      }
    }

LSTMPredictions <- LSTMPredictions %>% mutate(Residuals = LogDensity-RealValue)

save(LSTMPredictions,LSTMPredictionsSD,file="lstm_test_predictions_and_residuals.RData")
