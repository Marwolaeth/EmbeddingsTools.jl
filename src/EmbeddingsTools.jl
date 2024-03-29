__precompile__(true)

module EmbeddingsTools

# https://docs.julialang.org/en/v1/manual/functions/#Argument-type-declarations

export AbstractEmbedding, WordEmbedding, IndexedWordEmbedding
export EmbeddingVectorView, EmbeddingDict
export read_vec, read_embedding, read_emb, read_indexed_emb
export index, get_vector, subspace, limit, reduce_emb
export OutOfVocabularyException, UnknownReductionMethodException
export write_embedding

include("functions.jl")

end