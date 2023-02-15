using DifferentialEquations
using Random
using DelimitedFiles

#Random.seed!(12033131)

#SDE Definition
function traitModel!(dN,N,p,t)
    Tmean = p[1]
    Tamp = p[2]
    τ = p[3]
    σtrait = p[4]
    μmax = p[5]
    m = p[6]
    Rtot = p[7]
    z = p[8]

    T = Tmean+Tamp*sin(2pi*t/τ)
    for i in 1:length(N)
        dN[i] = N[i]*(μmax*exp(-(T-z[i])^2/σtrait^2)*(Rtot-sum(N))-m)+1e-3
    end
end
noiseProcess!(dN,N,p,t) = (dN .= N)
brownianMean = 1.
brownianVariance = 0.5
W = GeometricBrownianMotionProcess(brownianMean,brownianVariance,0.,1.)

#Set fixed parameters of Kremer and Klausmeier, 2017
Tmean = 0
Tamp = 5
τ = 20
σtrait = 8
μmax = 1
m = 1
Rtot = 1000
Tmax = 1000.

#Set number of species
numSpecies = 10
z = rand(-10:0.1:10,numSpecies)
ic = rand(1:Rtot,numSpecies)/Rtot

#Solve SDE
p = [Tmean,Tamp,τ,σtrait,μmax,m,Rtot,z]
prob = ODEProblem(traitModel!,ic,(0.,Tmax),p)
sol = solve(prob,Tsit5(),saveat = 1.,abstol=1e-13,reltol=1e-13)
#prob = SDEProblem(traitModel!,noiseProcess!,ic,(0.,Tmax),p,noise = W)
#sol = solve(prob,ISSEulerHeun(),saveat = 1.,abstol=1e-13,reltol=1e-13)

writedlm("simulation.csv",sol)

#Pure forecasting power (RMSE)
#Final community size (number of species at t goes to infinity)
#UDE might have environmental fluctuation given
#Possible source of variation: Varying μmax
#Vary number of species 
#Add immigration (1e-3 might be big)
#Start on the attractor to see if this works
#Log scale on population densities might be of most interest