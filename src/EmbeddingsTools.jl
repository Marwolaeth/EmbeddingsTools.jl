module EmbeddingsTools

export AbstractEmbedding
export WordEmbedding
export IndexedWordEmbedding
export read_vec, read_embedding, index, get, subspace, limit


# using DelimitedFiles
using CSV

# TYPES ----
EmbeddingVectorView = SubArray{
    Float32,
    1,
    Matrix{Float32},
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

# FUNCTIONS ----
## Check if tokens are present in an embedding vocabulary ----
"""
function _check_tokens(
    words::Vector{String},
    vocab::Vector{String}
)::Vector{Bool}

Returns a vector of logical values indicating whether each of the `words` is present in the vocabulary `vocab`.
"""
@inline function _check_tokens(
    words::Vector{String},
    vocab::Vector{String}
)::Vector{Bool}
    n_words = length(words)
    in_vocab = zeros(Bool, n_words)
    @inbounds Threads.@threads for i ∈ 1:n_words
        in_vocab[i] = words[i] ∈ vocab
    end

    return in_vocab
end

## Search for words in an embedding vocabulary ----
"""
_get_vocab_indices(
    words::Vector{String},
    vocab::Vector{String}
)::Vector{Int}

Returns a vector of indices of each `word` in `words` in the vocabulary `vocab` if `word` is present in the vocabulary.
"""
@inline function _get_vocab_indices(
    words::Vector{String},
    vocab::Vector{String}
)::Vector{UInt32}
    n_words = length(words)

    idx = zeros(UInt32, n_words)
    @inbounds Threads.@threads for i ∈ 1:n_words
        idx[i] = UInt32(findfirst(vocab .≡ words[i]))
    end
    return idx
end

## Find index of a sinle word ----
"""
_get_vocab_index(
    word::String,
    vocab::Vector{String}
)::UInt32

Returns an index of a `word` in the vocabulary `vocab`. Asserts that `word` is present in the vocabulary.
"""
@inline function _get_vocab_index(
    word::String,
    vocab::Vector{String}
)::UInt32

    idx::UInt32 = UInt32(findfirst(vocab .≡ word))

    return idx
end

"""
_get_vocab_index(
    word::String,
    vocab::Vector{String}
)::UInt32

Returns an index of a `word` in the vocabulary `vocab`. Asserts that `word` is present in the vocabulary. A word can be a substring.
"""
@inline function _get_vocab_index(
    word::SubString{String},
    vocab::Vector{String}
)::UInt32

    idx::UInt32 = UInt32(findfirst(vocab .≡ String(word)))

    return idx
end

## Reading Embedding Vector Files ----
"""
read_vec(path::AbstractString; delim::AbstractChar=' ')::WordEmbedding

The function `read_vec()` reads a local embedding matrix from the given `path` with delimiter `delim` and creates a `WordEmbedding` object using the CSV.jl package. This function is a simplified version of `read_embedding()` and always reads in the entire embedding table, which results in more straightforward logic.
"""
function read_vec(path::AbstractString; delim::AbstractChar=' ')::WordEmbedding
    # Read dimensionality
    ntokens, ndims = parse.(Int, split(readline(path), delim))

    # Don't try to read an entire huge vector of tokens
    ## Tackle it line-by-line
    (ntokens ≥ 600_000) && return read_giant_vec(path, delim=delim)

    # Pre-allocate
    emb = WordEmbedding(
        Array{Float32}(undef, ndims, ntokens),  # Embeddings Matrix (transposed)
        Array{String}(undef, ntokens),          # Token Vocabulary
        ntokens,                                # Vocabulary Size
        ndims                                   # Embedding Dimensionality
    )

    # Read CSV & Write Vectors column-wise
    @inbounds emb.embeddings .= CSV.Tables.matrix(
        CSV.File(
            path;
            skipto=2,
            delim=delim,
            header=false,
            types=Float32,
            drop=[1]
        )
    )'

    # Note: is it Ok to read the file twice? How can we avoid it?
    @inbounds emb.vocab .= CSV.Tables.getcolumn(
        CSV.File(
            path;
            skipto=2,
            delim=delim,
            header=false,
            types=String,
            select=[1]
        ),
        1
    )[:, 1]
    return emb
end

"""
read_giant_vec(path::AbstractString; delim::AbstractChar=' ')::WordEmbedding

The special conservative version of `read_embedding()`. It is designed to handle very large embedding tables, such as those used in FastText. This function reads a local embedding matrix from the specified `path` using the specified `delim`, line by line. It then creates a `WordEmbedding` object. However, this function can be very slow, so it is recommended to set the `max_vocab_size` parameter to a value less than 150k.
"""
function read_giant_vec(
    path::AbstractString;
    delim::AbstractChar=' ',
    max_vocab_size::Union{Int,Nothing}=nothing,
    keep_words::Union{Vector{String},Nothing}=nothing
)::WordEmbedding
    # Handle f***ups
    if !isnothing(keep_words)
        if length(keep_words) ≤ 0
            keep_words = nothing
        else
            max_vocab_size = length(keep_words)
        end
    end

    # Read dimensionality
    ntokens, ndims = parse.(Int, split(readline(path), delim))

    if isnothing(max_vocab_size) || !(0 < max_vocab_size < ntokens)
        max_vocab_size = ntokens
    end


    # Pre-allocate
    emb = WordEmbedding(
        Array{Float32}(undef, ndims, max_vocab_size),  # Embeddings Matrix (transposed)
        Array{String}(undef, max_vocab_size),          # Token Vocabulary
        max_vocab_size,                                # Vocabulary Size
        ndims                                          # Embedding Dimensionality
    )

    index::UInt32 = one(UInt32)
    open(path; read=true) do fh
        readline(fh)
        @inbounds while !eof(fh)
            l::String = readline(fh)
            embedding = split(l, delim)
            word = embedding[1]
            if isnothing(keep_words) || (word ∈ keep_words)
                if !isnothing(keep_words)
                    ind::UInt32 = _get_vocab_index(word, keep_words)
                else
                    ind = index
                end
                emb.vocab[ind] = word
                emb.embeddings[:, ind] .= parse.(Float32, @view embedding[2:end])
                index += 1
            end

            index > max_vocab_size && break
        end
    end

    return emb
end

"""
read_embedding(path::AbstractString; delim::AbstractChar=' ', max_vocab_size::Union{Int,Nothing}=nothing, keep_words::Union{Vector{String},Nothing}=nothing)::WordEmbedding

The purpose of the function `read_embedding()` is to read embedding files in a conventional way. It uses CSV.jl to create a `WordEmbedding` object, and has two optional keyword arguments: `max_vocab_size` and `keep_words`.

When you call this function, you need to provide the path to the local embedding vector. If `max_vocab_size` is specified, the function limits the size of the vector to that number, and if `keep_words` is specified, it only keeps the words in the provided vector. If a word in `keep_words` is not found, the function returns a zero vector for that word.

Notes
=====

Note that if you set `max_vocab_size` ≥ 45k, the function's performance may suffer compared to `limit(read_embedding(path), max_vocab_size)`. This is because using this parameter restricts CSV.jl jobs to a single thread. However, using `max_vocab_size` results in less memory allocation than reading the entire file.
    
In addition, using `keep_words` with as many as 1k selected words is significantly slower yet more memory-efficient compared to `subspace(index(read_embedding(path)), keep_words)`. For 10k selected words reading may take more than 5 seconds.
"""
function read_embedding(
    path::AbstractString;
    delim::AbstractChar=' ',
    max_vocab_size::Union{Int,Nothing}=nothing,
    keep_words::Union{Vector{String},Nothing}=nothing
)::WordEmbedding
    # Read dimensionality
    ntokens, ndims = parse.(Int, split(readline(path), delim))

    # Where is the best limit?
    (ntokens ≥ 600_000) && return read_giant_vec(
        path, delim=delim, max_vocab_size=max_vocab_size, keep_words=keep_words
    )

    # Values that depend on whether pythonistic parameters are used
    nthreads::UInt8 = Threads.nthreads()    # If no `max_vocab_size`
    keep_only_selected_words::Bool = false  # If no `keep_words`

    # From now on, `max_vocab_size` is used as the vocabulary size
    ## Check if it makes sense, otherwise use signature value
    if isnothing(max_vocab_size) || !(0 < max_vocab_size < ntokens)
        max_vocab_size = ntokens
    else
        nthreads = one(UInt8)
    end

    # Check if `keep_words` are provided & limit the vocabulary size
    if !isnothing(keep_words) && (length(keep_words) > 0)
        # Shorthand of the above condition
        keep_only_selected_words = true
        max_vocab_size = length(keep_words)
        initial_vocab::Array{String} = keep_words
        # Pre-allocate the entire vocabulary from the file
        vocab_placeholder = Array{String}(undef, ntokens)
    else
        initial_vocab = Array{String}(undef, max_vocab_size)
    end

    # Pre-allocate the WordEmbedding object
    emb = WordEmbedding(
        zeros(Float32, ndims, max_vocab_size), # Embeddings Matrix (transposed)
        initial_vocab,                         # Token Vocabulary
        max_vocab_size,                        # Vocabulary Size
        ndims                                  # Embedding Dimensionality
    )

    if keep_only_selected_words
        # Very slow even for a collection of words that is only moderate in size
        ## Pythonistas won't mind

        # Read the full vocabulary
        @inbounds vocab_placeholder .= CSV.Tables.getcolumn(
            CSV.File(
                path;
                skipto=2,
                delim=delim,
                header=false,
                types=String,
                select=[1]
            ),
            1
        )[:, 1]

        # Connect the matrix inside the .vec file
        tab = CSV.File(
            path;
            skipto=2,
            delim=delim,
            header=false,
            types=Float32,
            drop=[1]
        )

        # Find the required words in the embedding vocabulary
        in_vocab::Vector{Bool} = _check_tokens(keep_words, vocab_placeholder)
        idx::Vector{UInt32} = _get_vocab_indices(keep_words[in_vocab], vocab_placeholder)

        # Fill the word-vectors for words that have been found
        @inbounds emb.embeddings[:, in_vocab] .= CSV.Tables.matrix(tab[idx])'
    else
        # Read CSV & Write Vectors column-wise
        ## Note: `max_vocab_size` requires using a single thread
        ### perhaps due to a data race when `limit` is provided
        @inbounds emb.embeddings .= CSV.Tables.matrix(
            CSV.File(
                path;
                skipto=2,
                limit=max_vocab_size,
                delim=delim,
                header=false,
                types=Float32,
                drop=[1],
                ntasks=nthreads
            )
        )'

        # Read the first `max_vocab_size` word strings
        @inbounds emb.vocab .= CSV.Tables.getcolumn(
            CSV.File(
                path;
                skipto=2,
                limit=max_vocab_size,
                delim=delim,
                header=false,
                types=String,
                select=[1],
                ntasks=nthreads
            ),
            1
        )[:, 1]
    end

    return emb
end

## Embedding Indexing ----
@inline function index(emb::WordEmbedding)::IndexedWordEmbedding
    IndexedWordEmbedding(emb)
end

"""
get(emb::IndexedWordEmbedding, query::String)

`get()` interface to indexed word embedding objects: returns embedding vector (Float32) for a given token `query`. Called with the embedding object rather than with the dictionary. Type-stable: returns a view of the embedding vector or throws an exception.
"""
@inline function get(
    emb::IndexedWordEmbedding,
    query::String
)::EmbeddingVectorView
    exception_string::String = "Token $query not found!"
    v = Base.get(emb.dict, query, 'n')

    (v ≡ 'n') && throw(ErrorException(exception_string))

    return v
end

"""
getsafe(emb::IndexedWordEmbedding, query::String)

For internal use only. This function is similar to `get()` but returns a zero vector if the `query` is not in the vocabulary.
"""
@inline function getsafe(
    emb::IndexedWordEmbedding,
    query::String
)::EmbeddingVectorView
    v = Base.get(emb.dict, query, 'n')

    (v ≡ 'n') && return eachcol(zeros(Float32, emb.ndims))[1]

    return v
end

## Embedding Slicing ----
"""
subspace(emb::AbstractEmbedding, tokens::Vector{String})::WordEmbedding

The `subspace()` function takes an existing embedding and a subset of its vocabulary as input and creates a new `WordEmbedding` object. The order of embedding vectors in the new embedding corresponds to the order of input `tokens`. If a token is not found in the source embedding vocabulary, a zero vector is returned for that token. 

It's worth noting that this method is relatively slow and doesn't assume the source embedding to be indexed. Therefore, it doesn't take advantage of a lookup dictionary. It's recommended to index an embedding before subsetting it for better performance.
"""
@inline function subspace(emb::AbstractEmbedding, tokens::Vector{String})::WordEmbedding
    ntokens = length(tokens)

    # Pre-allocate
    sub = WordEmbedding(
        zeros(Float32, emb.ndims, ntokens),
        tokens,
        ntokens,
        emb.ndims
    )

    # Find the required words in the embedding vocabulary
    in_vocab::Vector{Bool} = _check_tokens(tokens, emb.vocab)
    idx::Vector{UInt32} = _get_vocab_indices(tokens[in_vocab], emb.vocab)

    # Fill the word-vectors for words that have been found
    @inbounds sub.embeddings[:, in_vocab] .= emb.embeddings[:, idx]

    return sub
end

"""
subspace(emb::IndexedWordEmbedding, tokens::Vector{String})::WordEmbedding

The `subspace()` function takes an existing indexed embedding and a subset of its vocabulary as input and creates a new `WordEmbedding` object. It takes two arguments: `emb`, which is the existing indexed embedding, and `tokens`, which is a vector of strings representing the subset of vocabulary.

The order of embedding vectors in the new embedding corresponds to the order of input `tokens`. If a token is not found in the source embedding vocabulary, a zero vector is returned.

It is recommended to `index()` an embedding before subsetting it, as this method assumes that the source embedding is indexed and can use its lookup dictionary. This makes it relatively fast.

Note that the result is not an indexed embedding, so if the user wants it indexed, it needs to be done manually.
"""
@inline function subspace(emb::IndexedWordEmbedding, tokens::Vector{String})::WordEmbedding
    ntokens = length(tokens)

    # Pre-allocate
    sub = WordEmbedding(
        Array{Float32}(undef, emb.ndims, ntokens),
        tokens,
        ntokens,
        emb.ndims
    )

    # Fill
    # @inbounds sub.embeddings .= cat([getsafe(emb, t) for t ∈ tokens]...; dims=2)

    @inbounds @simd for i ∈ 1:ntokens
        sub.embeddings[:, i] .= getsafe(emb, tokens[i])
    end

    return sub
end

"""
limit(emb::AbstractEmbedding, n::Integer)::WordEmbedding

The `limit()` function creates a copy of an existing word embedding, containing only the first `n` tokens. This function is similar to using a `max_vocab_size` argument in `read_embedding()`. However, using `read_vec()` + `limit()` or `read_vec()` + `subspace()` is generally faster than using `max_vocab_size` or `keep_words` arguments, respectively.
"""
function limit(emb::AbstractEmbedding, n::Integer)::WordEmbedding
    # Early check
    (0 < n < emb.ntokens) || return emb

    WordEmbedding(
        emb.embeddings[:, 1:n],
        emb.vocab[1:n],
        n,
        emb.ndims
    )
end

"""
limit(emb::IndexedWordEmbedding, n::Integer)::IndexedWordEmbedding

The `limit()` function creates a copy of an existing indexed word embedding, containing only the first `n` tokens. This function is similar to using a `max_vocab_size` argument in `read_embedding()`. However, using `read_vec()` + `limit()` or `read_vec()` + `subspace()` is generally faster than using `max_vocab_size` or `keep_words` arguments, respectively.
"""
function limit(emb::IndexedWordEmbedding, n::Integer)::IndexedWordEmbedding
    # Early check
    (0 < n < emb.ntokens) || return emb

    sub = WordEmbedding(
        emb.embeddings[:, 1:n],
        emb.vocab[1:n],
        n,
        emb.ndims
    )

    new::IndexedWordEmbedding = index(sub)
    return new
end

# P.S.
#GloryToUkraine
#PeaceForUkraine
#ArmUkraineASAP

end
