using FLoops
include("NODEUtils.jl")
include("dataGeneration.jl")

#Run conditions
communitySizes = [10,20,50]
immigrationRates = [0,1e-3,1e-1]
numberofTimeSeries = 4
trainingSizes = [10, 30, 50]
initialWeightsNumber = 32
Tmax = 100
knownDynamics(x,m,q) = -q

#Run NODEs
for (communitySize,immigrationRate,trainingSize) in zip(communitySizes,immigrationRates,trainingSizes)
    for i in 1:numberofTimeSeries
        timeSeries = log10.(generateTimeSeries(communitySize,Tmax,im = immigrationRate))
        writedlm("Models/timeSeries_communitySize_"*string(communitySize)*"_immigrationRate_"*
            string(immigrationRate)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*".csv",timeSeries)
        Threads.@threads for j in 1:initialWeightsNumber
            #Training of models
            NODEAutonomous = denseLayersLux(communitySize,[communitySize*3,communitySize*2])
            trainedParamsNODEAutonomous = trainNODEModel(NODEAutonomous,timeSeries[:,trainingSize])
            saveNeuralNetwork(NODE(NODEAutonomous,trainedParamsNODEAutonomous),
                fileName="Models/autonomous_NODE_communitySize_"*string(communitySize)*"_immigrationRate_"*
                    string(immigrationRate)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j))
            
            NODENonAutonomous = denseLayersLux(communitySize+1,[communitySize*3,communitySize*2])
            trainedParamsNODENonAutonomous = trainNODEModel(NODENonAutonomous,[timeSeries[:,trainingSize];collect(1:trainingSize)'])
            saveNeuralNetwork(NODE(NODENonAutonomous,trainedParamsNODENonAutonomous),
                fileName="Models/nonautonomous_NODE_communitySize_"*string(communitySize)*"_immigrationRate_"*
                    string(immigrationRate)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j))

            UDEAutonomous = denseLayersLux(communitySize,[communitySize*3,communitySize*2])
            trainedParamsUDEAutonomous = trainUDEModel(UDEAutonomous,knownDynamics,timeSeries[:,trainingSize],p_true=1)
            saveNeuralNetwork(UDE(UDEAutonomous,trainedParamsUDEAutonomous,knownDynamics),
                fileName="Models/autonomous_UDE_communitySize_"*string(communitySize)*"_immigrationRate_"*
                    string(immigrationRate)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j))
                
            UDENonAutonomous = denseLayersLux(communitySize+1,[communitySize*3,communitySize*2])
            trainedParamsUDENonAutonomous = trainUDEModel(UDENonAutonomous,knownDynamics,[timeSeries[:,trainingSize];collect(1:trainingSize)'],p_true=1)
            saveNeuralNetwork(UDE(UDENonAutonomous,trainedParamsUDENonAutonomous,knownDynamics),
                fileName="Models/nonautonomous_UDE_communitySize_"*string(communitySize)*"_immigrationRate_"*
                    string(immigrationRate)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j))
                
            #Testing of models
            NODEAutonomousTest = testNODEModel(trainedParamsNODEAutonomous,NODEAutonomous,timeSeries[:,(trainingSize)],50)
            writedlm("Results/test_autonomous_NODE_communitySize_"*string(communitySize)*"_immigrationRate_"*
                string(immigrationRate)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",NODEAutonomousTest)
            NODENonAutonomousTest = testNODEModel(trainedParamsNODENonAutonomous,NODENonAutonomous,[timeSeries[:,(trainingSize)];trainingSize],50)
            writedlm("Results/test_nonautonomous_NODE_communitySize_"*string(communitySize)*"_immigrationRate_"*
                string(immigrationRate)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",NODENonAutonomousTest)
            UDEAutonomousTest = testUDEModel(trainedParamsUDEAutonomous,UDEAutonomous,knownDynamics,timeSeries[:,(trainingSize)],50,p_true=1)
            writedlm("Results/test_autonomous_UDE_communitySize_"*string(communitySize)*"_immigrationRate_"*
                string(immigrationRate)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",UDEAutonomousTest)
            UDENonAutonomousTest = testUDEModel(trainedParamsUDENonAutonomous,UDENonAutonomous,knownDynamics,[timeSeries[:,(trainingSize)];trainingSize],50,p_true=1)
            writedlm("Results/test_nonautonomous_UDE_communitySize_"*string(communitySize)*"_immigrationRate_"*
                string(immigrationRate)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",UDENonAutonomousTest)

            #Residuals
            NODEAutonomousResiduals = normalizedResiduals(NODEAutonomousTest,timeSeries[:,trainingSize:(trainingSize+50)])
            writedlm("Results/residuals_autonomous_NODE_communitySize_"*string(communitySize)*"_immigrationRate_"*
                string(immigrationRate)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",NODEAutonomousResiduals)
            NODENonAutonomousResiduals = normalizedResiduals(NODEAutonomousTest,[timeSeries[1:communitySize,trainingSize:(trainingSize+50)])
            writedlm("Results/residuals_autonomous_NODE_communitySize_"*string(communitySize)*"_immigrationRate_"*
                string(immigrationRate)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",NODENonAutonomousResiduals)
            UDEAutonomousResiduals = normalizedResiduals(NODEAutonomousTest,timeSeries[:,trainingSize:(trainingSize+50)])
            writedlm("Results/residuals_autonomous_NODE_communitySize_"*string(communitySize)*"_immigrationRate_"*
                string(immigrationRate)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",UDEAutonomousResiduals)
            UDENonAutonomousResiduals = normalizedResiduals(NODEAutonomousTest,timeSeries[1:communitySize,trainingSize:(trainingSize+50)])
            writedlm("Results/residuals_autonomous_NODE_communitySize_"*string(communitySize)*"_immigrationRate_"*
                string(immigrationRate)*"_trainingSize_"*string(trainingSize)*"_rep_"*string(i)*"_"*string(j)*".csv",UDENonAutonomousResiduals)
        end
    end
end