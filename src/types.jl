# Constants ----
const global BINARY_EXTS_SIMPLE::Vector{String} = [".emb", ".jld", ".wem"]
const global BINARY_EXTS_INDEXD::Vector{String} = [".jld2", ".iem"]
const global TOKEN_NOT_FOUND::Symbol = Symbol('n')

# TYPES ----
EmbeddingVectorView = SubArray{
    Float32,
    1,
    Array{Float32,2},
    Tuple{Base.Slice{Base.OneTo{Int64}},Int64},
    true
}
EmbeddingDict = Dict{String,EmbeddingVectorView}

# CLASSES ----
abstract type AbstractEmbedding end
struct WordEmbedding <: AbstractEmbedding
    embeddings::Matrix{Float32}  # Embeddings Matrix (transposed)
    vocab::Vector{String}        # Token Vocabulary
    ntokens::Int                 # Vocabulary Size
    ndims::Int                   # Embedding Dimensionality
end

struct IndexedWordEmbedding <: AbstractEmbedding
    embeddings::Matrix{Float32}  # Embeddings Matrix (transposed)
    vocab::Vector{String}        # Token Vocabulary
    dict::EmbeddingDict          # Lookup Dictionary (Token ⟹ Embedding Vector)
    ntokens::Int                 # Vocabulary Size
    ndims::Int                   # Embedding Dimensionality

    # Constructed from unindexed embedding by adding a lookup dictionary
    function IndexedWordEmbedding(emb::WordEmbedding)
        d::EmbeddingDict = Dict((emb.vocab .=> eachcol(emb.embeddings))...)

        new(emb.embeddings, emb.vocab, d, emb.ntokens, emb.ndims)
    end
end

# Errors ----
struct OutOfVocabularyException <: Exception
    token::String
end
function Base.showerror(io::IO, e::OutOfVocabularyException)::Nothing
    print(io, "Token “$(e.token)” not found!\n")
end

struct UnknownReductionMethodException <: Exception
    meth::String
end
function Base.showerror(io::IO, e::UnknownReductionMethodException)::Nothing
    print(io, "Uknown reduction method: “$(e.meth)”\n")
end
