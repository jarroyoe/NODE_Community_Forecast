include("dataGeneration.jl")

#Run conditions
communitySizes = [10,20,50]
observationErrors = [0,1e-3,1e-1]
numberofTimeSeries = 4
#trainingSizes = [10, 30, 50]
#initialWeightsNumber = 32
Tmax = 100

for (communitySize,observationError,trainingSize) in zip(communitySizes,observationErrors,trainingSizes)
    for i in 1:numberofTimeSeries
        timeSeries = Array(log10.(generateTimeSeries(communitySize,Tmax,σobservation = observationError))')
        writedlm("Models/timeSeries_communitySize_"*string(communitySize)*"_observationError_"*
            string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*".csv",timeSeries)
    end
end
