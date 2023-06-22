using OrdinaryDiffEq, DiffEqFlux, Lux
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Lux, Zygote, Random, StatsBase, LinearAlgebra
using CUDA
rng = Random.default_rng()

training_data = rand(10,20)
neuralNetwork = Lux.Chain(Lux.Dense(10,20),Lux.Dense(20,10))
knownDynamics(x,p,q) = -x
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

nn_dynamics(u,p,t) = ude(u,p,t,nothing)
prob_nn = ODEProblem(nn_dynamics,x0, (Float64(1),Float64(size(training_data,2))), p64)
function predict(p, X = x0)
        _prob = remake(prob_nn, u0 = X, tspan = (Float64(1),Float64(size(training_data,2))), p = p)
        CUDA.@allowscalar convert(CuArray,solve(_prob, AutoTsit5(Rosenbrock23()), saveat = 1.,
                abstol=1e-6, reltol=1e-6
                ))
end

function loss_function(p)
	pred = predict(p)
	loss = sum(abs2,training_data .- pred)/size(training_data,2)
	return loss
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_function(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p64)

result_neuralode = Optimization.solve(optprob,
                                           ADAM(),maxiters=2)
