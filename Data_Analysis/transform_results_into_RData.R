rm(list=ls())
library(dplyr)

communitySizes <- c(10,40)
observationErrors <- c("0.0","0.1","0.01")
trainingSizes <- c(10,30,50,100)
timeAware <- c("autonomous","nonautonomous")
architecture <- c("NODE","UDE")

testPredictions <- data.frame(ModelType = character(),
                              ArchitectureType = character(),
                              CommunitySize = integer(),
                              ObservationError = double(),
                              TrainingSize = integer(), 
                              TimeSeriesID = integer(),
                              RepNumber = integer(),
                              Time = integer(),
                              PopID = integer(),
                              LogDensity = double(),
                              RealValue = double())

for(i in communitySizes){
  for(j in observationErrors){
    for(k in trainingSizes){
      for(l in timeAware){
        isAware <- ifelse(l=="autonomous","Time Agnostic","Time Aware")
        for(m in architecture){
          for(n in 1:5){
            currTimeSeries <- read.csv(paste(sep="",
                                             paste(sep='_',"Data/timeSeries_communitySize",i,
                                             "observationError",j,"rep",n),".csv"),
                                       header=FALSE,sep='\t')
            for(p in 1:4){
              currTest <- tryCatch(read.csv(paste(sep="",paste(sep = "_",
                                                                     "Results/test",l,m,"communitySize",
                                                                     i,"observationError",j,
                                                                     "trainingSize",k,"rep",n,p),".csv"),
                                                  sep="\t",header=FALSE),
                                         error = function(e) e)
              if(is.data.frame(currTest)){
                if(isAware == "Time Aware"){currTest <- currTest[-nrow(currTest),]}
                totalEntries <- ncol(currTest)*nrow(currTest)
                testPredictions <- rbind(testPredictions,
                                         data.frame(ModelType = rep(isAware,totalEntries),
                                                    ArchitectureType = rep(m,totalEntries),
                                                    CommunitySize = rep(i,totalEntries),
                                                    ObservationError = rep(as.double(j),totalEntries),
                                                    TrainingSize = rep(k,totalEntries), 
                                                    TimeSeriesID = rep(n,totalEntries),
                                                    RepNumber = rep(p,totalEntries),
                                                    Time = rep(k:(k+50),times=i),
                                                    PopID = rep(1:i,each = 51),
                                                    LogDensity = as.vector(t(currTest)),
                                                    RealValue = as.vector(t(currTimeSeries[,k:(k+50)]))
                                         ))
              }
            }
          }
        }
      }
    }
  }
  }

testPredictions <- testPredictions %>% mutate(Residuals = LogDensity-RealValue)

save(testPredictions,file="test_predictions_and_residuals.RData")