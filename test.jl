using CUDA, LinearAlgebra, Zygote, Random

function loss(X,niters=10)
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

dL(W) = gradient(X->loss(X),W)
dL(CUDA.rand(3,2))
