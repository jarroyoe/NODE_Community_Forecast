using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse, DiffEqFlux
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Lux, Zygote, Plots, Random, StatsBase, LinearAlgebra
using CUDA
using DelimitedFiles, Serialization
rng = Random.default_rng()

#Set up structs
struct NODE
    neuralNetwork
    parameters
end

struct UDE
    neuralNetwork
    parameters
    knownDynamics
end
function knownDynamics end #create function to point methods of UDE objects to

#Model architectures
function denseLayersLux(inputSize,hiddenSize;functions=nothing)
    if isnothing(functions)
        functions = tanh
    end
    nn = Lux.Chain(Lux.Dense(inputSize,hiddenSize,functions),
                    Lux.Dense(hiddenSize,inputSize))
    return nn
end

#Training and testing models
function trainNODEModel(neuralNetwork,training_data)
    pinit, st = Lux.setup(rng,neuralNetwork)
    st = st |> Lux.gpu
    p64 = Float64.(Lux.gpu(ComponentArray(pinit)))
    training_data = Lux.gpu(training_data)
    x0 = training_data[:,1] |> Lux.gpu
    neuralode = NeuralODE(neuralNetwork, (1.,Float64(size(training_data,2))), AutoTsit5(Rosenbrock23()),saveat=1.)

    function predict_neuralode(p)
        Lux.gpu(first(neuralode(x0, p,st)))
    end
    
    lipschitz_regularizer = 0.01
    function loss_function(p)
	W1 = CUDA.@allowscalar reshape(p.layer_1[1:(end-length(pinit.layer_1[2]))],size(pinit.layer_1[1]))
	W2 = CUDA.@allowscalar reshape(p.layer_2[1:(end-length(pinit.layer_2[2]))],size(pinit.layer_2[1]))
        lipschitz_constant = spectralRadius(W1)*spectralRadius(W2)

        pred = predict_neuralode(p)
        loss = sum(abs2,training_data .- pred)/size(training_data,2) + lipschitz_regularizer*lipschitz_constant
        return loss, pred
    end


    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_function(x), adtype)
    optprob = Optimization.OptimizationProblem(optf,p64)

    result_neuralode = Optimization.solve(optprob,
                                           ADAM(0.005),
                                           #callback = callback,
                                           maxiters = 300)

    optprob2 = remake(optprob,u0 = result_neuralode.u)
    result_neuralode2 = Optimization.solve(optprob2,
                                            Optim.BFGS(initial_stepnorm=0.01),
                                            #callback=callback,
                                            allow_f_increases = false)


    return result_neuralode2.u
end

function testNODEModel(params,neuralNetwork,x0,T)
    p, st = Lux.setup(rng,neuralNetwork) |> Lux.gpu
    x0 = Lux.gpu(x0)
    neuralode = NeuralODE(neuralNetwork,(0.,Float64(T)),AutoTsit5(Rosenbrock23()),saveat=1)
    return Lux.gpu(first(neuralode(x0,params,st)))                                                                                                                                                                                                                                                                                                                 
end

#SDE Models WIP
# function trainSDEModel(driftneuralNetwork,diffusionneuralNetwork,training_data)
#     p1, re1 = Flux.destructure(driftneuralNetwork)
#     p2, re2 = Flux.destructure(diffusionneuralNetwork)

#     neuralsde = NeuralDSDE(driftneuralNetwork,diffusionneuralNetwork,(Float32(1),Float32(size(training_data,2))),SOSRI(),saveat = 1)

#     function predict_neuralsde(p,u=training_data[:,1])
#         Array(neuralsde(u, p))
#     end
    
#     function loss_function(p;n=100)
#         u = repeat(reshape(training_data[:,1], :, 1), 1, n)
#         samples = predict_neuralsde(p, u)
#         means = mean(samples, dims = 2)
#         vars = var(samples, dims = 2, mean = means)[:, 1, :]
#         means = means[:, 1, :]
#         loss = sum(abs2, training_data - means)
#         return loss, means, vars
#     end

#     callback = function (p, loss, means, vars; doplot = false)
#         global list_plots, iter
      
#         if iter == 0
#           list_plots = []
#         end
#         iter += 1
      
#         # loss against current data
#         display(loss)
      
#         # plot current prediction against data
#         plt = Plots.scatter(tsteps, ode_data[1,:], yerror = sde_data_vars[1,:],
#                            ylim = (-4.0, 8.0), label = "data")
#         Plots.scatter!(plt, tsteps, means[1,:], ribbon = vars[1,:], label = "prediction")
#         push!(list_plots, plt)
      
#         if doplot
#           display(plt)
#         end
#         return false
#       end

#     opt = ADAM(0.025)

#     # First round of training with n = 10
#     adtype = Optimization.AutoZygote()
#     optf = Optimization.OptimizationFunction((x,p) -> loss_function(x, n=10), adtype)
#     optprob = Optimization.OptimizationProblem(optf, neuralsde.p)
#     result1 = Optimization.solve(optprob, opt,
#                                  callback = callback, maxiters = 100)

#     optf2 = Optimization.OptimizationFunction((x,p) -> loss_function(x, n=100), adtype)
#     optprob2 = Optimization.OptimizationProblem(optf2, result1.u)
#     result2 = Optimization.solve(optprob2, opt,
#                                 callback = callback, maxiters = 100)
#     return result2
# end

# function testSDEModel(params,driftneuralNetwork,diffusionneuralNetwork,x0,T)
#     neuralsde = NeuralSDE(driftneuralNetwork,diffusionneuralNetwork,(Float32(0),Float32(T)),SOSRI(),saveat = 1)
    
#     function loss_function(p;n=100)
#         u = repeat(reshape(x0, :, 1), 1, n)
#         samples = predict_neuralsde(p, u)
#         means = mean(samples, dims = 2)
#         vars = var(samples, dims = 2, mean = means)[:, 1, :]
#         means = means[:, 1, :]
#         loss = sum(abs2, sde_data - means) + sum(abs2, sde_data_vars - vars)
#         return loss, means, vars
#     end                             
    
#     _, means, vars = loss_function(params, n = 1000)

#     return means,vars
# end

function trainUDEModel(neuralNetwork,knownDynamics,training_data;needed_ps = Float32[],p_true = Float32[])
    ps, st = Lux.setup(rng, neuralNetwork)

    #ps_dynamics = Lux.ComponentArray((predefined_params = rand(Float32, needed_ps), model_params = ps))
    

    function ude!(du,u,p,t,q)
        #knownPred = knownDynamics(u,p.predefined_params,q)
        knownPred = knownDynamics(u,nothing,q)
        #nnPred = Array(neuralNetwork(u,p.model_params,st)[1])
        nnPred = Array(neuralNetwork(u,p,st)[1])

        for i in 1:length(u)
            du[i] = knownPred[i]+nnPred[i]
        end
    end

    # Closure with the known parameter
    nn_dynamics!(du,u,p,t) = ude!(du,u,p,t,p_true)
    # Define the problem
    prob_nn = ODEProblem(nn_dynamics!,training_data[:, 1], (Float32(1),Float32(size(training_data,2))), ps)
    ## Function to train the network
    # Define a predictor
    function predict(p, X = training_data[:,1], T = 1:size(training_data,2))
        _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = p)
        Array(solve(_prob, Tsit5(), saveat = T,
                abstol=1e-6, reltol=1e-6
                ))
    end

    # Simple L2 loss
    function loss_function(p)
        X̂ = predict(p)
        sum(abs2, training_data .- X̂)
    end

    losses = Float64[]

    callback = function (p, l)
      push!(losses, l)
    if length(losses)%50==0
          println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
    end

    ## Training
    pinit = ComponentVector(ps)
    #callback(pinit, loss_function(pinit)...; doplot=true)

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_function(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, pinit)

    result_neuralode = Optimization.solve(optprob,
                                           ADAM(),
                                           #callback = callback,
                                           maxiters = 300)

    optprob2 = remake(optprob,u0 = result_neuralode.u)
    result_neuralode2 = Optimization.solve(optprob2,
                                            Optim.BFGS(initial_stepnorm=0.01),
                                            #callback=callback,
                                            allow_f_increases = false)


    return result_neuralode2.u
end

function testUDEModel(params,neuralNetwork,knownDynamics,x0,T;p_true = nothing)
    ps, st = Lux.setup(rng, neuralNetwork)
    
    function ude!(du,u,p,t,q)
        knownPred = knownDynamics(u,nothing,q)
        nnPred = Array(neuralNetwork(u,p,st)[1])

        for i in 1:length(u)
            du[i] = knownPred[i]+nnPred[i]
        end
    end
    # Closure with the known parameter
    nn_dynamics!(du,u,p,t) = ude!(du,u,p,t,p_true)
    # Define the problem
    prob_nn = ODEProblem(nn_dynamics!,x0, (Float32(0),Float32(T)), params)
    prediction = Array(solve(prob_nn, Tsit5(), saveat = 1,
                abstol=1e-6, reltol=1e-6
                ))
    return prediction
end

#Other functions

function normalizedResiduals(predicted,observed)
    (observed.-predicted)./observed
end

function rowwiseCoefficientOfVariation(data)
    [variation(data[:,i]) for i in 1:size(data,2)]
end

function spectralRadius(X,niters=10)
    y = randn!(similar(X, size(X, 2)))
    tmp = X * y
    for i in 1:niters
        tmp = X*y
        tmp = tmp / norm(tmp)
        y = X' * tmp
        y = y / norm(y)
    end
    return norm(X*y)
end

#Save functions

function saveNeuralNetwork(model::NODE;fileName = "fit_neural_network")
    serialize(fileName*".jls",model)
end

function saveNeuralNetwork(model::UDE;fileName = "fit_neural_network")
    methodToSave = methods(model.knownDynamics)[1]
    modelToSave = UDE(model.neuralNetwork,model.parameters,methodToSave)
    serialize(fileName*".jls",modelToSave)
end

function loadNeuralNetwork(fileName)
    return deserialize(fileName)
end
