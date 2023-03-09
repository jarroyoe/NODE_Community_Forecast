include("NODEUtils.jl")
include("dataGeneration.jl")

#Run conditions
communitySizes = [10,20,50]
observationErrors = [0,1e-3,1e-1]
numberofTimeSeries = 4
trainingSizes = [10, 30, 50]
initialWeightsNumber = 32
Tmax = 100
knownDynamics(x,m,q) = -q

#Run NODEs
for (communitySize,observationError,trainingSize) in zip(communitySizes,observationErrors,trainingSizes)
    for i in 1:numberofTimeSeries
        timeSeries = readdlm("Models/timeSeries_communitySize_"*string(communitySize)*"_observationError_"*
        string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*".csv")
        Threads.@threads for j in 1:initialWeightsNumber
            #Training of models
            NODEAutonomous = denseLayersLux(communitySize,[communitySize*3,communitySize*2])
            trainedParamsNODEAutonomous = trainNODEModel(NODEAutonomous,timeSeries[:,1:trainingSize])
            saveNeuralNetwork(NODE(NODEAutonomous,trainedParamsNODEAutonomous),
                fileName="Models/autonomous_NODE_communitySize_"*string(communitySize)*"_observationError_"*
                    string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j))
            
            NODENonAutonomous = denseLayersLux(communitySize+1,[communitySize*3,communitySize*2])
            trainedParamsNODENonAutonomous = trainNODEModel(NODENonAutonomous,[timeSeries[:,1:trainingSize];collect(1:trainingSize)'])
            saveNeuralNetwork(NODE(NODENonAutonomous,trainedParamsNODENonAutonomous),
                fileName="Models/nonautonomous_NODE_communitySize_"*string(communitySize)*"_observationError_"*
                    string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j))

        
            #Testing of models
            NODEAutonomousTest = testNODEModel(trainedParamsNODEAutonomous,NODEAutonomous,timeSeries[:,(trainingSize)],50)
            writedlm("Results/test_autonomous_NODE_communitySize_"*string(communitySize)*"_observationError_"*
                string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",NODEAutonomousTest)
            NODENonAutonomousTest = testNODEModel(trainedParamsNODENonAutonomous,NODENonAutonomous,[timeSeries[:,(trainingSize)];trainingSize],50)
            writedlm("Results/test_nonautonomous_NODE_communitySize_"*string(communitySize)*"_observationError_"*
                string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",NODENonAutonomousTest)
            

            #Residuals
            NODEAutonomousResiduals = normalizedResiduals(NODEAutonomousTest,timeSeries[:,trainingSize:(trainingSize+50)])
            writedlm("Results/residuals_autonomous_NODE_communitySize_"*string(communitySize)*"_observationError_"*
                string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",NODEAutonomousResiduals)
            NODENonAutonomousResiduals = normalizedResiduals(NODEAutonomousTest,[timeSeries[1:communitySize,trainingSize:(trainingSize+50)])
            writedlm("Results/residuals_autonomous_NODE_communitySize_"*string(communitySize)*"_observationError_"*
                string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",NODENonAutonomousResiduals)

        end
    end
end