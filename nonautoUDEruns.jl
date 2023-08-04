using FLoops
include("NODEUtils.jl")
include("dataGeneration.jl")

#Run conditions
communitySizes = [10,40]
observationErrors = [0,1e-2,1e-1]
numberofTimeSeries = 2
trainingSizes = [30, 50, 100]
initialWeightsNumber = 4
Tmax = 100

#Run NODEs
@floop for (communitySize,observationError,trainingSize) in Iterators.product(communitySizes,observationErrors,trainingSizes)
    for i in 1:numberofTimeSeries
	knownDynamics(x,m,q) = [-q*ones(communitySize);1/365]
        timeSeries = (readdlm("Data/timeSeries_communitySize_"*string(communitySize)*"_observationError_"*
        string(observationError)*"_rep_"*string(i)*".csv"))
        for j in 1:initialWeightsNumber
	    #Check to prevent double work
	    isfile("Models/nonautonomous_UDE_communitySize_"*string(communitySize)*"_observationError_"*
                    string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".jls") && continue

            #Training of models
            UDENonAutonomous = denseLayersLux(communitySize+1,communitySize*2)
            trainedParamsUDENonAutonomous = trainUDEModel(UDENonAutonomous,knownDynamics,[timeSeries[:,1:trainingSize];mod.(collect(1:7:trainingSize*7),365)'/365],p_true=1)
            saveNeuralNetwork(UDE(UDENonAutonomous,trainedParamsUDENonAutonomous,knownDynamics),
                fileName="Models/nonautonomous_UDE_communitySize_"*string(communitySize)*"_observationError_"*
                    string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j))
                
            #Testing of models
            UDENonAutonomousTest = testUDEModel(trainedParamsUDENonAutonomous,UDENonAutonomous,knownDynamics,[timeSeries[:,(trainingSize)];trainingSize/365],30,p_true=1)
            CUDA.@allowscalar writedlm("Results/test_nonautonomous_UDE_communitySize_"*string(communitySize)*"_observationError_"*
                string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",UDENonAutonomousTest)
        end
    end
end
