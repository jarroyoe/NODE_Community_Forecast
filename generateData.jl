using DelimitedFiles
using FLoops
include("dataGeneration.jl")

#Run conditions
communitySizes = [10,40]
observationErrors = [0,1e-2,1e-1]
numberofTimeSeries = 5
#trainingSizes = [10, 30, 50]
#initialWeightsNumber = 32
Tmax = 200

@floop for (communitySize,observationError) in Iterators.product(communitySizes,observationErrors)
    for i in 3:numberofTimeSeries
        timeSeries = Array(log10.(generateTimeSeries(communitySize,Tmax,σobservation = observationError))')
        writedlm("Data/timeSeries_communitySize_"*string(communitySize)*"_observationError_"*
            string(observationError)*"_rep_"*string(i)*".csv",timeSeries)
    end
end
