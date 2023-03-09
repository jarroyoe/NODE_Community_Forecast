using DifferentialEquations

#Random.seed!(12033131)

#ODE Definition
function traitModel!(dN,N,p,t)
    Tmean = p[1]
    Tamp = p[2]
    τ = p[3]
    σtrait = p[4]
    μmax = p[5]
    m = p[6]
    Rtot = p[7]
    z = p[8]
    im = p[9]

    T = Tmean+Tamp*sin(2pi*t/τ)
    for i in 1:length(N)
        dN[i] = N[i]*(μmax*exp(-(T-z[i])^2/σtrait^2)*(Rtot-sum(N))-m)+im
    end
end

function generateTimeSeries(communitySize,Tmax;Tmean=0,Tamp=5,τ=20,σtrait=8,μmax=1,m=1,Rtot=1000,im=1e-3,traitRange=10,σobservation = 0)
    z = rand(-traitRange:0.1:traitRange,communitySize)
    p = [Tmean,Tamp,τ,σtrait,μmax,m,Rtot,z,im]
    initialConditions = rand(1:Rtot,communitySize)
    initialConditions *= 0.8*Rtot/sum(initialConditions) #Adjust initial conditions to consume 80% of the total resources

    prob = ODEProblem(traitModel!,initialConditions,(0.,Tmax),p)
    sol = solve(prob,Tsit5(),saveat = 1.)
    trueSol = Float32.([sol.u[i][j] for i in 1:Tmax,j in 1:communitySize])
    observedSol = max.(trueSol.*(1 .+σobservation*randn(Float32,(Tmax,communitySize))),0)
    return observedSol
end