using DelimitedFiles
using FLoops
include("dataGeneration.jl")

#Run conditions
communitySizes = [10,20,50]
observationErrors = [0,1e-3,1e-1]
numberofTimeSeries = 4
#trainingSizes = [10, 30, 50]
#initialWeightsNumber = 32
Tmax = 100

@floop for (communitySize,observationError) in Iterators.product(communitySizes,observationErrors)
    for i in 1:numberofTimeSeries
        timeSeries = Array(log10.(generateTimeSeries(communitySize,Tmax,σobservation = observationError))')
        writedlm("Models/timeSeries_communitySize_"*string(communitySize)*"_observationError_"*
            string(observationError)*"_rep_"*string(i)*".csv",timeSeries)
    end
end
