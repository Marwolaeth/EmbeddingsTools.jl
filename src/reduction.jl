import Statistics: mean
import LinearAlgebra: eigen

"""
    reduce_pca(X::Matrix{Float32}, r::Int=2)::Matrix{Float32}

Reduce a matrix using Principal component analysis. This function returns the transformed input matrix `X` using `r` first principal components. The function assumes that the matrix is transposed with observations in columns and variables in rows.

*Note:* This function doesn't use `MultivariateStats.jl` to avoid unnecessary dependencies. We recommend using `PCA` from `MultivariateStats.jl` for principal component analysis.
"""
function reduce_pca(X::Matrix{Float32}, k::Int=2)::Matrix{Float32}
    # Original Dimensions
    p, n = size(X)
    # Limit k so that it can be computed
    k = min(k, p, n)
    # Indices of λ and eigenvectors to be used
    idx = p:-1:(p-(k-1))

    # Pre-allocate
    X₀ = zeros(Float32, p, n)  # Centered source data
    Σ = zeros(Float32, p, p)   # The covariance matrix
    P = zeros(Float32, p, k)   # The projection (Selected eigenvectors)
    Y = zeros(Float32, k, n)   # Transformed data (the result)

    # The Mean Vector
    μ = mean(X, dims=2)

    # Center
    @inbounds X₀ .= X .- μ

    # The Covariance Matrix
    @inbounds Σ .= (X₀ * X₀') / (n - 1)

    # Eigenstuff
    λ, V = eigen(Σ)
    @assert maximum(λ) ≡ λ[end]

    # Check and report the explained variance
    λₖ = λ[idx]
    variance_explained = sum(λₖ) / sum(λ)
    println("PCA dimensionality reduction: $p => $k dimensions")
    println(
        "The first $k components account for \
        $(round(Float64(variance_explained*100.), digits = 3))% \
        of the total variance\n"
    )

    @inbounds P .= V[:, idx]
    @inbounds Y .= P'X₀

    return Y
end
