using Flux, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random, StatsBase
using Plots
using DelimitedFiles, Serialization
rng = Random.default_rng()

#Set up structs
struct NODE
    chain
    parameters
end

struct UDE
    chain
    parameters
    knownDynamics
end

#Model architectures
function denseLayersLux(inputSize,hiddenSize;functions=nothing)
    if isnothing(functions)
        functions = [tanh,tanh]
    end
    nn = Lux.Chain(Lux.Dense(inputSize,hiddenSize[1],functions[1]),
                    Lux.Dense(hiddenSize[1],hiddenSize[2],functions[2]),
                    Lux.Dense(hiddenSize[2],inputSize))
    return nn
end

function denseLayersFlux(inputSize,hiddenSize;functions=nothing)
    if isnothing(functions)
        functions = [tanh,tanh]
    end
    nn = Flux.Chain(Flux.Dense(inputSize,hiddenSize[1],functions[1]),
                    Flux.Dense(hiddenSize[1],hiddenSize[2],functions[2]),
                    Flux.Dense(hiddenSize[2],inputSize))
    return nn
end


#Training and testing models
function trainNODEModel(chain,training_data)
    p, st = Lux.setup(rng,chain)
    neuralode = NeuralODE(chain, (Float32(1),Float32(size(training_data,2))), AutoTsit5(Rosenbrock23()),saveat=1)

    function predict_neuralode(p)
        Array(neuralode(training_data[:,1], p,st)[1])
    end
    
    function loss_function(p)
        pred = predict_neuralode(p)
        loss = sum(abs2,training_data .- pred)
        return loss, pred
    end

    callback = function (p, l, pred; doplot = true)
        println(l)
        # plot current prediction against data
        if doplot
          plt = scatter(1:size(training_data,2), training_data[1,:], label = "data")
          scatter!(plt, 1:size(training_data,2), pred[1,:], label = "prediction")
          display(plot(plt))
        end
        return false  
    end

    pinit = Lux.ComponentArray(p)
    callback(pinit, loss_function(pinit)...; doplot=true)

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_function(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, pinit)

    result_neuralode = Optimization.solve(optprob,
                                           ADAM(0.005),
                                           callback = callback,
                                           maxiters = 300)

    optprob2 = remake(optprob,u0 = result_neuralode.u)
    result_neuralode2 = Optimization.solve(optprob2,
                                            Optim.BFGS(initial_stepnorm=0.01),
                                            callback=callback,
                                            allow_f_increases = false)


    return result_neuralode2.u
end

function testNODEModel(params,chain,x0,T)
    p, st = Lux.setup(rng,chain)
    neuralode = NeuralODE(chain,(Float32(0),Float32(T)),AutoTsit5(Rosenbrock23()),saveat=1)
    return Array(neuralode(x0,params,st)[1])                                                                                                                                                                                                                                                                                                                 
end

#SDE Models WIP
# function trainSDEModel(driftchain,diffusionchain,training_data)
#     p1, re1 = Flux.destructure(driftchain)
#     p2, re2 = Flux.destructure(diffusionchain)

#     neuralsde = NeuralDSDE(driftchain,diffusionchain,(Float32(1),Float32(size(training_data,2))),SOSRI(),saveat = 1)

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

# function testSDEModel(params,driftchain,diffusionchain,x0,T)
#     neuralsde = NeuralSDE(driftchain,diffusionchain,(Float32(0),Float32(T)),SOSRI(),saveat = 1)
    
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

function trainUDEModel(chain,knownDynamics,training_data;needed_ps = Float32[],p_true = Float32[])
    ps, st = Lux.setup(rng, chain)

    ps_dynamics = Lux.ComponentArray((predefined_params = rand(Float32, needed_ps), model_params = ps))

    function ude!(du,u,p,t,q)
        knownPred = knownDynamics(u,p.predefined_params,q)
        nnPred = Array(chain(u,p.model_params,st)[1])

        for i in 1:length(u)
            du[i] = knownPred[i]+nnPred[i]
        end
    end

    # Closure with the known parameter
    nn_dynamics!(du,u,p,t) = ude!(du,u,p,t,p_true)
    # Define the problem
    prob_nn = ODEProblem(nn_dynamics!,training_data[:, 1], (Float32(1),Float32(size(training_data,2))), ps_dynamics)
    ## Function to train the network
    # Define a predictor
    function predict(p, X = training_data[:,1], T = 1:size(training_data,2))
        _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = p)
        Array(solve(_prob, Tsit5(), saveat = T,
                abstol=1e-6, reltol=1e-6
                ))
    end

    # Simple L2 loss
    function loss(p)
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

    # First train with ADAM for better convergence -> move the parameters into a
    # favourable starting positing for BFGS
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x,p)->loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ps_dynamics)
    res1 = Optimization.solve(optprob, ADAM(0.1), callback=callback, maxiters = 200)
    println("Training loss after $(length(losses)) iterations: $(losses[end])")
    # Train with BFGS
    optprob2 = Optimization.OptimizationProblem(optf, res1.minimizer)
    res2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.01), callback=callback, maxiters = 10000)
    println("Final training loss after $(length(losses)) iterations: $(losses[end])")

    return res2.minimizer
end

function testUDEModel(params,chain,knownDynamics,x0,T;p_true = nothing)
    ps, st = Lux.setup(rng, chain)
    
    function ude!(du,u,p,t,q)
        knownPred = knownDynamics(u,params.predefined_params,q)
        nnPred = chain(u,params.model_params,st)

        for i in 1:length(u)
            du[i] = knownPred[i]+nnPred[i]
        end
    end
    # Closure with the known parameter
    nn_dynamics!(du,u,p,t) = ude!(du,u,p,t,p_true)
    # Define the problem
    prob_nn = ODEProblem(nn_dynamics!,x0, (Float32(1),Float32(T)), params)



end

#Other functions

function normalizedResiduals(predicted,observed)
    (observed.-predicted)./observed
end

function saveNeuralNetwork(model;fileName = "fit_neural_network")
    serialize(fileName*".jls",model)
end


function loadNeuralNetwork(fileName)
    return deserialize(fileName)
end