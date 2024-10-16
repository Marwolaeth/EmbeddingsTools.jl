import Statistics: mean
import LinearAlgebra: eigen, svd, Diagonal

"""
    reduce_pca(X::Matrix{Float32}, k::Int=2)::Matrix{Float32}

Reduce a matrix using Principal component analysis. This function returns the transformed input matrix `X` using `k` first principal components. The function assumes that the matrix is transposed with observations in columns and variables in rows.

*Note:* This function doesn't use `MultivariateStats.jl` to avoid unnecessary dependencies. We recommend using `PCA` from `MultivariateStats.jl` for principal component analysis.
"""
function reduce_pca(X::Matrix{Float32}, k::Int=2)::Matrix{Float32}
    # Original Dimensions
    p, n = size(X)
    # Limit k so that it can be computed
    # k = min(k, p, n) # bypass as the check is done by the parent function
    # Indices of λ and eigenvectors to be used
    idx = p:-1:(p-(k-1))

    # Pre-allocate
    X₀ = zeros(Float32, p, n)   # Centered source data
    Σ = zeros(Float32, p, p)   # The covariance matrix
    P = zeros(Float32, p, k)   # The projection (Selected eigenvectors)
    Y = zeros(Float32, k, n)   # Transformed data (the result)

    # The Mean Vector
    μ = mean(X, dims=2)

    # Center
    X₀ .= X .- μ

    # The Covariance Matrix
    Σ .= (X₀ * X₀') / (n - 1)

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

    P .= V[:, idx]
    Y .= P'X₀

    return Y
end

"""
    reduce_svd(X::Matrix{Float32}, k::Int=2)::Matrix{Float32}

Reduce a matrix using Singular value decomposition. This function returns the transformed input matrix `X` using `k` first singular values. The function assumes that the matrix is transposed with observations in columns and variables in rows.
"""
function reduce_svd(X::Matrix{Float32}, k::Int=2)::Matrix{Float32}
    # Original Dimensions
    p, n = size(X)
    # Limit k so that it can be computed
    # k = min(k, p, n) # bypass as the check is done by the parent function
    # Indices of singular values and vectors to be used
    idx = 1:k

    # Pre-allocate
    Y = zeros(Float32, k, n)   # Transformed data (the result)
    Σ = zeros(Float32, k, k)   # Singular values matrix

    # The decomposition
    U, d, _ = svd(X')
    dₖ = d[idx]
    Σ .= Diagonal(dₖ)

    # Check and report the explained variance
    variance_explained = sum(dₖ .^ 2) / sum(d .^ 2)
    println("SVD dimensionality reduction: $p => $k dimensions")
    println(
        "The first $k singular values account for \
        $(round(Float64(variance_explained*100.), digits = 3))% \
        of the total variance\n"
    )

    Y' .= U[:, idx] * Σ

    return Y
end
