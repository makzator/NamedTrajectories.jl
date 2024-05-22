module Utils

export save
export load_traj
export derivative
export integral

using JLD2

using ..StructNamedTrajectory
using ..StructKnotPoint

function JLD2.save(filename::String, traj::NamedTrajectory; as_dict=false)
    parts = split(filename, ".")
    @assert parts[end] == "jld2"
    if as_dict
        filename = join(parts[1:end-2]) * parts[end-1] * "_dict.jld2"
        d = Dict(name => traj[name] for name in traj.names)
        save(filename, d)
    else
        save(filename, "traj", traj)
    end
end

function load_traj(filename::String)
    @assert split(filename, ".")[end] == "jld2"
    return load(filename, "traj")
end

"""
Forward difference (consistent with DerivativeIntegrator in QuantumCollocation.jl)
"""
function derivative(X::AbstractMatrix, Δt::AbstractVector)
    @assert size(X, 2) == length(Δt) "number of columns of X ($(size(X, 2))) must equal length of Δt ($(length(Δt))"
    dX = similar(X)
    dX[:, end] = zeros(size(X, 1))
    for t = 1:size(X, 2)-1
        Δx = X[:, t + 1] - X[:, t]
        h = Δt[t]
        dX[:, t] .= Δx / h
    end
    return dX
end

"""
Trapezoidal rule.
Maybe want to use "inverted forward difference" instead for consistency?
"""
function integral(X::AbstractMatrix, Δt::AbstractVector)
    ∫X = similar(X)
    ∫X[:, 1] = zeros(size(X, 1))
    for t = 2:size(X, 2)
        # trapezoidal rule
        ∫X[:, t] = ∫X[:, t-1] + (X[:, t] + X[:, t-1])/2 * Δt[t-1]
    end
    return ∫X
end


end
