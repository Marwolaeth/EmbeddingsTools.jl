import Statistics: mean
import LinearAlgebra: eigvecs

"""
reduce_pca(X::Matrix{Float32}, n_components::Int=2)::Matrix{Float32}

A limited PCA function that only returns the transformed input matrix `X`, assuming the matrix is transposed with observations in columns and variables in rows.
"""
function reduce_pca(X::Matrix{Float32}, n_components::Int=2)::Matrix{Float32}
    # Original Dimensions
    m, n = size(X)

    # Pre-allocate
    C = zeros(Float32, m, m)
    P = zeros(Float32, n_components, n)

    # The Mean Vector
    μ = mean(X, dims=2)

    # Center
    @inbounds X .= X .- μ

    # The Covariance Matrix
    @inbounds C .= (X * X') / (n - 1)

    # Eigenstuff
    λ, vectors = eigen(C)
    ## For now, we don't need the eigenvalues
    # λ_nc = λ[end-(n_components-1):end]
    vectors_nc = vectors[:, end-(n_components-1):end]

    @inbounds P' .= X' * vectors_nc

    return P
end