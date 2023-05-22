using FLoops
include("NODEUtils.jl")
include("dataGeneration.jl")

#Run conditions
communitySizes = [10,40]
observationErrors = [0,1e-3,1e-1]
numberofTimeSeries = 2
trainingSizes = [10, 30, 50]
initialWeightsNumber = 4
Tmax = 100

#Run NODEs
for (communitySize,observationError,trainingSize) in Iterators.product(communitySizes,observationErrors,trainingSizes)
    for i in 1:numberofTimeSeries
        timeSeries = (readdlm("Data/timeSeries_communitySize_"*string(communitySize)*"_observationError_"*
        string(observationError)*"_rep_"*string(i)*".csv"))
        for j in 1:initialWeightsNumber
	    #Check to prevent double work
	    isfile("Models/autonomous_NODE_communitySize_"*string(communitySize)*"_observationError_"*
                    string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".jls") && continue

            #Training of models
            NODEAutonomous = denseLayersLux(communitySize,[communitySize*3,communitySize*2])
            trainedParamsNODEAutonomous = trainNODEModel(NODEAutonomous,timeSeries[:,1:trainingSize])
            saveNeuralNetwork(NODE(NODEAutonomous,trainedParamsNODEAutonomous),
                fileName="Models/autonomous_NODE_communitySize_"*string(communitySize)*"_observationError_"*
                    string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j))
            
            #Testing of models
            NODEAutonomousTest = testNODEModel(trainedParamsNODEAutonomous,NODEAutonomous,timeSeries[:,(trainingSize)],50)
            CUDA.@allowscalar writedlm("Results/test_autonomous_NODE_communitySize_"*string(communitySize)*"_observationError_"*
                string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",NODEAutonomousTest)
        end
    end
end
