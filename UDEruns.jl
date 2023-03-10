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
            UDEAutonomous = denseLayersLux(communitySize,[communitySize*3,communitySize*2])
            trainedParamsUDEAutonomous = trainUDEModel(UDEAutonomous,knownDynamics,timeSeries[:,1:trainingSize],p_true=1)
            saveNeuralNetwork(UDE(UDEAutonomous,trainedParamsUDEAutonomous,knownDynamics),
                fileName="Models/autonomous_UDE_communitySize_"*string(communitySize)*"_observationError_"*
                    string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j))
                
            UDENonAutonomous = denseLayersLux(communitySize+1,[communitySize*3,communitySize*2])
            trainedParamsUDENonAutonomous = trainUDEModel(UDENonAutonomous,knownDynamics,[timeSeries[:,1:trainingSize];collect(1:trainingSize)'],p_true=1)
            saveNeuralNetwork(UDE(UDENonAutonomous,trainedParamsUDENonAutonomous,knownDynamics),
                fileName="Models/nonautonomous_UDE_communitySize_"*string(communitySize)*"_observationError_"*
                    string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j))
                
            #Testing of models
            UDEAutonomousTest = testUDEModel(trainedParamsUDEAutonomous,UDEAutonomous,knownDynamics,timeSeries[:,(trainingSize)],50,p_true=1)
            writedlm("Results/test_autonomous_UDE_communitySize_"*string(communitySize)*"_observationError_"*
                string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",UDEAutonomousTest)
            UDENonAutonomousTest = testUDEModel(trainedParamsUDENonAutonomous,UDENonAutonomous,knownDynamics,[timeSeries[:,(trainingSize)];trainingSize],50,p_true=1)
            writedlm("Results/test_nonautonomous_UDE_communitySize_"*string(communitySize)*"_observationError_"*
                string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",UDENonAutonomousTest)

            #Residuals
            UDEAutonomousResiduals = normalizedResiduals(UDEAutonomousTest,timeSeries[:,trainingSize:(trainingSize+50)])
            writedlm("Results/residuals_autonomous_NODE_communitySize_"*string(communitySize)*"_observationError_"*
                string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",UDEAutonomousResiduals)
            UDENonAutonomousResiduals = normalizedResiduals(UDENonAutonomousTest[1:communitySize,:],timeSeries[1:communitySize,trainingSize:(trainingSize+50)])
            writedlm("Results/residuals_autonomous_NODE_communitySize_"*string(communitySize)*"_observationError_"*
                string(observationError)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",UDENonAutonomousResiduals)
        end
    end
end