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
    
    lipschitz_regularizer = 0.5
    function loss_function(p)
	W1 = p.layer_1.weight
	W2 = p.layer_2.weight
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

function trainUDEModel(neuralNetwork,knownDynamics,training_data;needed_ps = Float64[],p_true = Float64[])
    pinit, st = Lux.setup(rng,neuralNetwork)
    st = st |> Lux.gpu
    p64 = Float64.(Lux.gpu(ComponentArray(pinit)))
    training_data = Lux.gpu(training_data)
    x0 = training_data[:,1] |> Lux.gpu

   function ude(u,p,t,q)
        knownPred = convert(CuArray,knownDynamics(u,nothing,q))
        nnPred = convert(CuArray,first(neuralNetwork(u,p,st)))

        knownPred .+ nnPred
    end

    # Closure with the known parameter
    nn_dynamics(u,p,t) = ude(u,p,t,p_true)
    # Define the problem
    prob_nn = ODEProblem(nn_dynamics,x0, (Float64(1),Float64(size(training_data,2))), p64)
    ## Function to train the network
    # Define a predictor
    function predict(p, X = x0)
        _prob = remake(prob_nn, u0 = X, tspan = (Float64(1),Float64(size(training_data,2))), p = p)
        CUDA.@allowscalar convert(CuArray,solve(_prob, AutoTsit5(Rosenbrock23()), saveat = 1.,
                abstol=1e-6, reltol=1e-6
                ))
    end

    lipschitz_regularizer = 0.5
    function loss_function(p)
	    W1 = p.layer_1.weight
	    W2 = p.layer_2.weight
        lipschitz_constant = spectralRadius(W1)*spectralRadius(W2)

        pred = predict(p)
        loss = sum(abs2,training_data .- pred)/size(training_data,2) + lipschitz_regularizer*lipschitz_constant
        return loss
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
    #callback(pinit, loss_function(pinit)...; doplot=true)

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_function(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p64)

    result_neuralode = Optimization.solve(optprob,
                                           ADAM(),
                                           callback = callback,
                                           maxiters = 300)

    optprob2 = remake(optprob,u0 = result_neuralode.u)
    result_neuralode2 = Optimization.solve(optprob2,
                                            Optim.BFGS(initial_stepnorm=0.01),
                                            callback=callback,
                                            allow_f_increases = false)


    return result_neuralode2.u
end

function testUDEModel(params,neuralNetwork,knownDynamics,x0,T;p_true = nothing)
    ps, st = Lux.setup(rng, neuralNetwork) |> Lux.gpu
    
    function ude(u,p,t,q)
        knownPred = convert(CuArray,knownDynamics(u,nothing,q))
        nnPred = convert(CuArray,first(neuralNetwork(u,p,st)))

        knownPred .+ nnPred
    end

    # Closure with the known parameter
    nn_dynamics(u,p,t) = ude(u,p,t,p_true)
    # Define the problem
    prob_nn = ODEProblem(nn_dynamics,Lux.gpu(x0), (Float64(0),Float64(T)), params)
    prediction = Array(solve(prob_nn, AutoTsit5(Rosenbrock23()), saveat = 1,
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
