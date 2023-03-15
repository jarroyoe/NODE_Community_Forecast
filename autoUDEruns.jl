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
	knownDynamics(x,m,q) = -q*ones(communitySize)
        timeSeries = Float32.(readdlm("Data/timeSeries_communitySize_"*string(communitySize)*"_observationError_"*
        string(observationError)*"_rep_"*string(i)*".csv"))
        for j in 1:initialWeightsNumber
            #Training of models
            UDEAutonomous = denseLayersLux(communitySize,[communitySize*3,communitySize*2])
            trainedParamsUDEAutonomous = trainUDEModel(UDEAutonomous,knownDynamics,timeSeries[:,1:trainingSize],p_true=1)
            saveNeuralNetwork(UDE(UDEAutonomous,trainedParamsUDEAutonomous,knownDynamics),
                fileName="Models/autonomous_UDE_communitySize_"*string(communitySize)*"_observationError_"*
                    string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j))
                
            #Testing of models
            UDEAutonomousTest = testUDEModel(trainedParamsUDEAutonomous,UDEAutonomous,knownDynamics,timeSeries[:,(trainingSize)],50,p_true=1)
            writedlm("Results/test_autonomous_UDE_communitySize_"*string(communitySize)*"_observationError_"*
                string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",UDEAutonomousTest)

            #Residuals
            UDEAutonomousResiduals = normalizedResiduals(UDEAutonomousTest,timeSeries[:,trainingSize:(trainingSize+50)])
            writedlm("Results/residuals_autonomous_NODE_communitySize_"*string(communitySize)*"_observationError_"*
                string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",UDEAutonomousResiduals)
        end
    end
end
