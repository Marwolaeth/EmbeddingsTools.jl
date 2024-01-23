__precompile__(true)

module EmbeddingsTools

# https://docs.julialang.org/en/v1/manual/functions/#Argument-type-declarations

export AbstractEmbedding, WordEmbedding, IndexedWordEmbedding
export read_vec, read_embedding, read_emb
export index, get, subspace, limit
export write_embedding

import CSV: CSV, CSV.File, CSV.Tables
using JLD2

include("functions.jl")

end