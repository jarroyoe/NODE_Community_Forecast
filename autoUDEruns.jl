using FLoops
include("NODEUtils.jl")
include("dataGeneration.jl")

#Run conditions
communitySizes = [10,40]
observationErrors = [0,1e-2,1e-1]
numberofTimeSeries = 5
trainingSizes = [10, 30, 50]
initialWeightsNumber = 4
Tmax = 100

#Run NODEs
@floop for (communitySize,observationError,trainingSize) in Iterators.product(communitySizes,observationErrors,trainingSizes)
    for i in 3:numberofTimeSeries
	knownDynamics(x,m,q) = -q*ones(communitySize)
        timeSeries = (readdlm("Data/timeSeries_communitySize_"*string(communitySize)*"_observationError_"*
        string(observationError)*"_rep_"*string(i)*".csv"))
        for j in 1:initialWeightsNumber
            #Training of models
	    isfile("Models/autonomous_UDE_communitySize_"*string(communitySize)*"_observationError_"*
                    string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".jls") && continue

            UDEAutonomous = denseLayersLux(communitySize,communitySize*2)
            trainedParamsUDEAutonomous = trainUDEModel(UDEAutonomous,knownDynamics,timeSeries[:,1:trainingSize],p_true=1)
            saveNeuralNetwork(UDE(UDEAutonomous,trainedParamsUDEAutonomous,knownDynamics),
                fileName="Models/autonomous_UDE_communitySize_"*string(communitySize)*"_observationError_"*
                    string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j))
                
            #Testing of models
            UDEAutonomousTest = testUDEModel(trainedParamsUDEAutonomous,UDEAutonomous,knownDynamics,timeSeries[:,(trainingSize)],50,p_true=1)
            CUDA.@allowscalar writedlm("Results/test_autonomous_UDE_communitySize_"*string(communitySize)*"_observationError_"*
                string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",UDEAutonomousTest)
        end
    end
end
