rm(list = ls())
library(dplyr)
library(rEDM)

communitySizes <- c(10,40)
observationErrors <- c("0.0","0.1","0.01")
trainingSizes <- c(10,30,50)

EDMPredictions <- data.frame(CommunitySize = integer(),
                               ObservationError = double(),
                               TrainingSize = integer(), 
                               TimeSeriesID = integer(),
                               Time = integer(),
                               PopID = integer(),
                               LogDensity = double(),
                               RealValue = double())
EDMPredictionsSD <- data.frame(CommunitySize = integer(),
                                 ObservationError = double(),
                                 TrainingSize = integer(), 
                                 TimeSeriesID = integer(),
                                 Time = integer(),
                                 LogDensity = double(),
                                 Lower95 = double(),
                                 Higher95 = double())

for(i in communitySizes){
  for(j in observationErrors){
    for(n in 1:5){
      currTimeSeries <- read.csv(paste(sep="",
                                       paste(sep="_","Data/timeSeries_communitySize",i,
                                             "observationError",j,"rep",n),".csv"),
                                 header=FALSE,sep='\t')
      for(k in trainingSizes){
        trainingData <- currTimeSeries[,1:k]
        for(m in 1:i){
          EDMforecast <-
            Simplex(
              dataFrame = as.data.frame(t(rbind(seq(1,200),currTimeSeries))),
              lib = paste(1,k),
              pred = paste(k,k+50),
              columns = paste(apply(as.matrix(seq(1,i)),2,function(x){paste("V",x+1,sep="")}),collapse = ' '),
              target = paste("V",m+1,sep=""),
              E = 2
            )
          predictions <- EDMforecast$Predictions[-1]
          low95 <- predictions - 1.96*sqrt(EDMforecast$Pred_Variance[-1])
          hi95 <- predictions + 1.96*sqrt(EDMforecast$Pred_Variance[-1])
          totalEntries <- length(predictions)
          EDMPredictions <- rbind(EDMPredictions,
                                    data.frame(CommunitySize = rep(i,totalEntries),
                                               ObservationError = rep(as.double(j),totalEntries),
                                               TrainingSize = rep(k,totalEntries), 
                                               TimeSeriesID = rep(n,totalEntries),
                                               Time = k:(k+50),
                                               PopID = rep(m,totalEntries),
                                               LogDensity = predictions,
                                               RealValue = as.vector(t(currTimeSeries[m,k:(k+50)]))))
          EDMPredictionsSD <- rbind(EDMPredictionsSD,
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

EDMPredictions <- EDMPredictions %>% mutate(Residuals = LogDensity-RealValue)

save(EDMPredictions,EDMPredictionsSD,file="EDM_test_predictions_and_residuals.RData")