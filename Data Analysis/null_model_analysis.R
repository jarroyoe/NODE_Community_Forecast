rm(list = ls())
library(dplyr)
library(forecast)

communitySizes <- c(10)
observationErrors <- c("0.0","0.1","0.01")
trainingSizes <- c(10,30,50,100)

ARIMAPredictions <- data.frame(CommunitySize = integer(),
                              ObservationError = double(),
                              TrainingSize = integer(), 
                              TimeSeriesID = integer(),
                              Time = integer(),
                              PopID = integer(),
                              LogDensity = double(),
                              RealValue = double())

ARIMAPredictionsSD <- data.frame(CommunitySize = integer(),
                                 ObservationError = double(),
                                 TrainingSize = integer(), 
                                 TimeSeriesID = integer(),
                                 Time = integer(),
                                 LogDensity = double(),
                                 Lower95 = double(),
                                 Higher95 = double())

for(i in communitySizes){
  for(j in observationErrors){
        for(n in 1:2){
          currTimeSeries <- read.csv(paste(sep="",
                                           paste(sep="_","Data/timeSeries_communitySize",i,
                                                 "observationError",j,"rep",n),".csv"),
                                     header=FALSE,sep='\t')
          for(k in trainingSizes){
            trainingData <- currTimeSeries[,1:k]
            for(m in 1:i){
              ARIMAmodel <- auto.arima(as.vector(t(trainingData[m,])))
              forecasts <- forecast(ARIMAmodel,h = 50)
              predictions <- c(trainingData[m,k],forecasts$mean)
              low95 <- c(0,forecasts$lower[,2])
              hi95 <- c(0,forecasts$upper[,2])
              totalEntries <- length(predictions)
              ARIMAPredictions <- rbind(ARIMAPredictions,
                                        data.frame(CommunitySize = rep(i,totalEntries),
                                                   ObservationError = rep(as.double(j),totalEntries),
                                                   TrainingSize = rep(k,totalEntries), 
                                                   TimeSeriesID = rep(n,totalEntries),
                                                   Time = k:(k+50),
                                                   PopID = rep(m,totalEntries),
                                                   LogDensity = predictions,
                                                   RealValue = as.vector(t(currTimeSeries[m,k:(k+50)]))))
              ARIMAPredictionsSD <- rbind(ARIMAPredictionsSD,
                                          data.frame(CommunitySize = rep(i,totalEntries),
                                          ObservationError = rep(as.double(j),totalEntries),
                                          TrainingSize = rep(k,totalEntries), 
                                          TimeSeriesID = rep(n,totalEntries),
                                          Time = k:(k+50),
                                          PopID = rep(m,totalEntries),
                                          LogDensity = predictions,
                                          Lower95 = as.vector(t(low95)),
                                          Higher95 = as.vector(t(hi95))))
            }
          }
        }
      }
    }

ARIMAPredictions <- ARIMAPredictions %>% mutate(Residuals = LogDensity-RealValue)

save(ARIMAPredictions,ARIMAPredictionsSD,file="null_test_predictions_and_residuals.RData")