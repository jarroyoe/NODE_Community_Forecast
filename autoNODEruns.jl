using FLoops
include("NODEUtils.jl")
include("dataGeneration.jl")

#Run conditions
communitySizes = [10,50]
observationErrors = [0,1e-3,1e-1]
numberofTimeSeries = 4
trainingSizes = [10, 30, 50]
initialWeightsNumber = 4
Tmax = 100

#Run NODEs
@floop for (communitySize,observationError,trainingSize) in Iterators.product(communitySizes,observationErrors,trainingSizes)
    for i in 1:numberofTimeSeries
        timeSeries = readdlm("Data/timeSeries_communitySize_"*string(communitySize)*"_observationError_"*
        string(observationError)*"_rep_"*string(i)*".csv")
        for j in 1:initialWeightsNumber
	    #Check to prevent double work
	    isfile("Models/autonomous_NODE_communitySize_"*string(communitySize)*"_observationError_"*
                    string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)) && continue

            #Training of models
            NODEAutonomous = denseLayersLux(communitySize,[communitySize*3,communitySize*2])
            trainedParamsNODEAutonomous = trainNODEModel(NODEAutonomous,timeSeries[:,1:trainingSize])
            saveNeuralNetwork(NODE(NODEAutonomous,trainedParamsNODEAutonomous),
                fileName="Models/autonomous_NODE_communitySize_"*string(communitySize)*"_observationError_"*
                    string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j))
            
            #Testing of models
            NODEAutonomousTest = testNODEModel(trainedParamsNODEAutonomous,NODEAutonomous,timeSeries[:,(trainingSize)],50)
            writedlm("Results/test_autonomous_NODE_communitySize_"*string(communitySize)*"_observationError_"*
                string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",NODEAutonomousTest)

            #Residuals
            NODEAutonomousResiduals = normalizedResiduals(NODEAutonomousTest,timeSeries[:,trainingSize:(trainingSize+50)])
            writedlm("Results/residuals_autonomous_NODE_communitySize_"*string(communitySize)*"_observationError_"*
                string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",NODEAutonomousResiduals)
        end
    end
end