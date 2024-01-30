import CSV: CSV, CSV.File, CSV.Tables
using JLD2
include("types.jl")
include("reduction.jl")

# FUNCTIONS ----
## Utilities ----
"""
    _ext(path::AbstractString)::String

Returns the extension of file in `path`, if any, and an empty string otherwise.
"""
@inline function _ext(path::AbstractString)::String
    ext = Base.splitext(path)[2]
    return ext
end

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

function _check_tokens(words::Nothing, vocab::Vector{String})
    nothing
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
)::Vector{Int}
    n_words = length(words)

    idx = zeros(Int, n_words)
    @inbounds Threads.@threads for i ∈ 1:n_words
        found = Base.findfirst(vocab .≡ words[i])
        if !isnothing(found)
            idx[i] = found
        end
    end
    return idx
end

## Find index of a sinle word ----
"""
    _get_vocab_index(
        word::String,
        vocab::Vector{String}
    )::Int

Returns an index of a `word` in the vocabulary `vocab`. Asserts that `word` is present in the vocabulary.
"""
@inline function _get_vocab_index(
    word::String,
    vocab::Vector{String}
)::Int

    idx = Base.findfirst(vocab .≡ word)

    return (isnothing(idx) ? 0 : idx)
end

"""
    _get_vocab_index(
        word::String,
        vocab::Vector{String}
    )::Int

Returns an index of a `word` in the vocabulary `vocab`. Asserts that `word` is present in the vocabulary. A word can be a substring.
"""
@inline function _get_vocab_index(
    word::SubString{String},
    vocab::Vector{String}
)::Int

    idx = Base.findfirst(vocab .≡ String(word))

    return (isnothing(idx) ? 0 : idx)
end

## Reading Embedding Vector Files ----
"""
    read_vec(path::AbstractString; delim::AbstractChar=' ')::WordEmbedding

The function `read_vec()` is used to read a local embedding matrix from a text file (.txt, .vec, etc) at a given `path`. It creates a `WordEmbedding` object using the CSV.jl package. The delimiter used for the text file can be set using the `delim` parameter. This function is a simplified version of `read_embedding()` and it always reads in the entire embedding table, making the logic more straightforward.
"""
function read_vec(path; delim=' ')::WordEmbedding
    # Read dimensionality
    ntokens, ndims = Base.parse.(Int, split(readline(path), delim))

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
            quotechar='`',
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
            quotechar='`',
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
    read_giant_vec(
        path::AbstractString;
        delim::AbstractChar=' ',
        max_vocab_size::Union{Int,Nothing}=nothing,
        keep_words::Union{Vector{String},Nothing}=nothing
    )::WordEmbedding

The conservative version of `read_embedding()` that handles large embedding tables, such as those used in FastText. It is adapted from a similar function in Embeddings.jl. The function reads a local embedding matrix from a specified `path` by going through each line and creates a `WordEmbedding` object. Additionally, you can provide the delimiter using `delim` and retain only certain words by specifying a list `keep_words`. However, this function can be slow, so we recommend setting the `max_vocab_size` parameter to a value less than 150k.
"""
function read_giant_vec(
    path;
    delim=' ',
    max_vocab_size::Union{Int,Nothing}=nothing,
    keep_words::Union{Vector{String},Nothing}=nothing
)::WordEmbedding
    # Read dimensionality
    ntokens, ndims = Base.parse.(Int, split(readline(path), delim))

    if isnothing(max_vocab_size) || !(0 < max_vocab_size < ntokens)
        max_vocab_size = ntokens
    end

    # Handle f***ups
    if !isnothing(keep_words)
        if length(keep_words) ≤ 0
            keep_words = nothing
        else
            # Don't collect `keep_words` that are out of `max_vocab_size`
            max_vocab_size = min(length(keep_words), max_vocab_size)
            keep_words = keep_words[1:max_vocab_size]
        end
    end

    # Pre-allocate
    emb = WordEmbedding(
        Array{Float32}(undef, ndims, max_vocab_size),  # Embeddings Matrix (transposed)
        Array{String}(undef, max_vocab_size),          # Token Vocabulary
        max_vocab_size,                                # Vocabulary Size
        ndims                                          # Embedding Dimensionality
    )

    # Copy the workflow from Embedding.jl
    index = 1
    open(path; read=true) do fh
        readline(fh)
        @inbounds while !eof(fh)
            l = readline(fh)
            embedding = split(l, delim)
            word = embedding[1]
            if isnothing(keep_words) || (word ∈ keep_words)
                if !isnothing(keep_words)
                    ind = _get_vocab_index(word, keep_words)
                else
                    ind = index
                end
                if ind > 0
                    emb.vocab[ind] = word
                    emb.embeddings[:, ind] .= Base.parse.(Float32, @view embedding[2:end])
                end
                index += 1
            end

            index > max_vocab_size && break
        end
    end

    return emb
end

"""
    read_emb(path::AbstractString)::WordEmbedding

The function reads word embeddings from local binary embedding table files in `.jld` and `.emb` formats. These files are Julia binary files that contain a `WordEmbedding` object under the name `"embedding"`.
"""
function read_emb(path::AbstractString)::WordEmbedding
    f::JLD2.JLDFile = jldopen(path, "r")
    emb::WordEmbedding = f["embedding"]
    close(f)

    return emb
end

"""
    read_indexed_emb(path::AbstractString)::WordEmbedding

The function reads indexed word embeddings from local binary embedding table files in `.jld2` and `.iem` formats. These files are Julia binary files that contain an `IndexedWordEmbedding` object under the name `"embedding"`.
"""
function read_indexed_emb(path::AbstractString)::IndexedWordEmbedding
    f::JLD2.JLDFile = jldopen(path, "r")
    emb::IndexedWordEmbedding = f["embedding"]
    close(f)

    return emb
end

"""
    read_embedding(
        path::AbstractString;
        delim::AbstractChar=' ',
        max_vocab_size::Union{Int,Nothing}=nothing,
        keep_words::Union{Vector{String},Nothing}=nothing
    )::WordEmbedding

The function `read_embedding()` is used to read embedding files in a conventional way. It creates a `WordEmbedding` object using CSV.jl. The function takes a path to the local embedding vector as an argument and has two optional keyword arguments: `max_vocab_size` and `keep_words`.

If `max_vocab_size` is specified, the function limits the size of the vector to that number. If a vector `keep_words` is provided, it only keeps those words. If a word in `keep_words` is not found, the function returns a zero vector for that word.

If the file is a `WordEmbedding` object within a Julia binary file (with extension `.jld` or in specific formats `.emb` or `.wem`), the entire embedding is loaded, and keyword arguments are not applicable. You can also use the `read_emb()` function directly on binary files.

Notes
=====

Note that if you set `max_vocab_size` ≥ 45k, the function's performance may suffer compared to `limit(read_embedding(path), max_vocab_size)`. This is because using this parameter restricts CSV.jl jobs to a single thread. However, using `max_vocab_size` results in less memory allocation than reading the entire file.
    
In addition, using `keep_words` with as many as 1k selected words is significantly slower yet more memory-efficient compared to `subspace(index(read_embedding(path)), keep_words)`. For 10k selected words reading may take more than 5 seconds.
"""
function read_embedding(
    path;
    delim=' ',
    max_vocab_size::Union{Int,Nothing}=nothing,
    keep_words::Union{Vector{String},Nothing}=nothing
)::WordEmbedding
    file_ext = _ext(path)

    # If Binary
    (file_ext ∈ BINARY_EXTS_SIMPLE) && return read_emb(path)

    # Read dimensionality
    ntokens, ndims = Base.parse.(Int, split(readline(path), delim))

    # Where is the best limit?
    (ntokens ≥ 600_000) && return read_giant_vec(
        path, delim=delim, max_vocab_size=max_vocab_size, keep_words=keep_words
    )

    # Values that depend on whether pythonistic parameters are used
    nthreads = Threads.nthreads()     # If no `max_vocab_size`
    keep_only_selected_words = false  # If no `keep_words`

    # From now on, `max_vocab_size` is used as the vocabulary size
    ## Check if it makes sense, otherwise use signature value
    if isnothing(max_vocab_size) || !(0 < max_vocab_size < ntokens)
        max_vocab_size = ntokens
    else
        nthreads = 1
    end

    # Check if `keep_words` are provided & limit the vocabulary size
    if !isnothing(keep_words) && (length(keep_words) > 0)
        # Shorthand of the above condition
        keep_only_selected_words = true

        # Don't collect `keep_words` that are out of `max_vocab_size`
        max_vocab_size = min(length(keep_words), max_vocab_size)
        keep_words = keep_words[1:max_vocab_size]

        # Pre-allocate the final vocabulary
        initial_vocab::Array{String} = keep_words
        # Pre-allocate the entire vocabulary from the file
        ## We wouldn't need this value if no `keep_words` provided
        vocab_placeholder = Array{String}(undef, ntokens)
    else
        # Pre-allocate the final vocabulary
        initial_vocab = Array{String}(undef, max_vocab_size)
    end

    # Pre-allocate the WordEmbedding object
    emb = WordEmbedding(
        zeros(Float32, ndims, max_vocab_size), # Embeddings Matrix (transposed)
        initial_vocab,                         # Token Vocabulary
        max_vocab_size,                        # Vocabulary Size
        ndims                                  # Embedding Dimensionality
    )

    if keep_only_selected_words && !isnothing(keep_words) # JET please get it don't be so dumb
        # Very slow even for a collection of words that is only moderate in size
        ## Pythonistas won't mind

        # Read the full vocabulary
        @inbounds vocab_placeholder .= CSV.Tables.getcolumn(
            CSV.File(
                path;
                quotechar='`',
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
            quotechar='`',
            skipto=2,
            delim=delim,
            header=false,
            types=Float32,
            drop=[1]
        )

        # Find the required words in the embedding vocabulary
        in_vocab = _check_tokens(keep_words, vocab_placeholder)
        idx = _get_vocab_indices(keep_words[in_vocab], vocab_placeholder)

        # Fill the word-vectors for words that have been found
        @inbounds emb.embeddings[:, in_vocab] .= CSV.Tables.matrix(tab[idx])'
    else
        # Read CSV & Write Vectors column-wise
        ## Note: `max_vocab_size` requires using a single thread
        ### perhaps due to a data race when `limit` is provided
        @inbounds emb.embeddings .= CSV.Tables.matrix(
            CSV.File(
                path;
                quotechar='`',
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
                quotechar='`',
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
    get_vector(emb::WordEmbedding, query::String)

`get_vector()` returns embedding vector (Float32) for a given token `query`. Called with the embedding object rather than with the dictionary. Type-stable: returns a view of the embedding vector or throws an exception.
"""
@inline function get_vector(
    emb::WordEmbedding,
    query::String
)::EmbeddingVectorView
    (query ∈ emb.vocab) || throw(TokenNotFoundException(query))
    idx = _get_vocab_index(query, emb.vocab)
    v::EmbeddingVectorView = view(emb.embeddings, :, idx)

    return v
end

"""
    get_vector(emb::IndexedWordEmbedding, query::String)

`get_vector()` returns embedding vector (Float32) for a given token `query`. Called with the embedding object rather than with the dictionary. Type-stable: returns a view of the embedding vector or throws an exception.
"""
@inline function get_vector(
    emb::IndexedWordEmbedding,
    query::String
)::EmbeddingVectorView
    v = Base.get(emb.dict, query, TOKEN_NOT_FOUND)

    (v ≡ TOKEN_NOT_FOUND) && throw(TokenNotFoundException(query))

    return v
end

"""
    safe_get(emb::IndexedWordEmbedding, query::String)

For internal use only. This function is similar to `get()` but returns a zero vector if the `query` is not in the vocabulary.
"""
@inline function safe_get(
    emb::IndexedWordEmbedding,
    query::String
)::EmbeddingVectorView
    v = Base.get(emb.dict, query, TOKEN_NOT_FOUND)

    (v ≡ TOKEN_NOT_FOUND) && return eachcol(zeros(Float32, emb.ndims))[1]

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
    in_vocab = _check_tokens(tokens, emb.vocab)
    idx = _get_vocab_indices(tokens[in_vocab], emb.vocab)

    # Fill the word-vectors for words that have been found
    @inbounds sub.embeddings[:, in_vocab] .= emb.embeddings[:, idx]

    return sub
end

"""
    subspace(emb::IndexedWordEmbedding, tokens::Vector{String})::WordEmbedding

The `subspace()` function takes an already indexed embedding and a subset of its vocabulary as input and generates a new `WordEmbedding` object. It requires two arguments: `emb` which is the indexed embedding, and `tokens` which is a vector of strings representing the subset of vocabulary.

The order of embedding vectors in the new embedding corresponds to the order of the input `tokens`. If an out-of-vocabulary token is encountered, a zero vector is returned.

This method assumes that the source embedding is indexed and can use its lookup dictionary, making it relatively fast. It is recommended to `index()` an embedding before subsetting it.

Note that the output of the `subspace()` function is not an indexed embedding. So, if the user wants it indexed, it needs to be done manually.
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
    @inbounds @simd for i ∈ 1:ntokens
        sub.embeddings[:, i] .= safe_get(emb, tokens[i])
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

"""
    reduce_emb(emb::AbstractEmbedding, k::Integer; method::String="pca")::WordEmbedding

The following function takes an existing word embedding and reduces its embedding vectors to a specified number of dimensions `k`. The function returns a new WordEmbedding object. You can choose between two reduction techniques by setting the `method` parameter to either `pca` for Principal Component Analysis or `svd` for Singular Value Decomposition.
"""
function reduce_emb(emb::AbstractEmbedding, k::Integer; method::String="pca")::WordEmbedding
    # Current dimensions
    p, n = size(emb.embeddings)
    # Limit k so that it can be computed
    k = min(k, p, n)
    # Wrap the method argument
    method = lowercase(method)

    sub = WordEmbedding(
        emb.embeddings[1:k, :],
        emb.vocab,
        n,
        k
    )

    if method ≡ "pca"
        sub.embeddings .= reduce_pca(emb.embeddings, k)
    elseif method == "svd"
        sub.embeddings .= reduce_svd(emb.embeddings, k)
    else
        throw(UnknownReductionMethodException(method))
    end

    return sub
end

"""
    reduce_emb(emb::IndexedWordEmbedding, k::Integer; method::String="pca")::WordEmbedding

The following function takes an existing indexed word embedding and reduces its embedding vectors to a specified number of dimensions `k`. The function returns a new IndexedWordEmbedding object. You can choose between two reduction techniques by setting the `method` parameter to either `pca` for Principal Component Analysis or `svd` for Singular Value Decomposition.
"""
function reduce_emb(
    emb::IndexedWordEmbedding,
    k::Integer;
    method::String="pca"
)::IndexedWordEmbedding
    # Current dimensions
    p, n = size(emb.embeddings)
    # Limit k so that it can be computed
    k = min(k, p, n)
    # Wrap the method argument
    method = lowercase(method)

    sub = WordEmbedding(
        emb.embeddings[1:k, :],
        emb.vocab,
        n,
        k
    )

    if method ≡ "pca"
        sub.embeddings .= reduce_pca(emb.embeddings, k)
    elseif method == "svd"
        sub.embeddings .= reduce_svd(emb.embeddings, k)
    else
        throw(UnknownReductionMethodException(method))
    end

    new::IndexedWordEmbedding = index(sub)
    return new
end

"""
    write_embedding(
        emb::WordEmbedding,
        path::AbstractString;
        max_vocab_size::Union{Int,Nothing}=nothing,
        keep_words::Union{Vector{String},Nothing}=nothing
    )::Nothing

The `write_embedding()` function saves a `WordEmbedding` object to a binary file specified by `path`. The vocabulary can be filtered with `keep_words` and limited to `max_vocab_size`.
"""
function write_embedding(
    emb::WordEmbedding,
    path::AbstractString;
    max_vocab_size::Union{Int,Nothing}=nothing,
    keep_words::Union{Vector{String},Nothing}=nothing
)::Nothing

    # Should we preprocess the object?
    do_limit = !isnothing(max_vocab_size) && !(0 < max_vocab_size < emb.ntokens)
    do_subspace = !isnothing(keep_words) && (length(keep_words) > 0)
    if !(do_limit || do_subspace)
        jldsave(path; embedding=emb)
    end

    do_subspace && (emb = subspace(emb, keep_words))
    do_limit && (emb = limit(emb, max_vocab_size))
    jldsave(path; embedding=emb)
end

"""
    write_embedding(
        emb::IndexedWordEmbedding,
        path::AbstractString;
        max_vocab_size::Union{Int,Nothing}=nothing,
        keep_words::Union{Vector{String},Nothing}=nothing
    )::Nothing

The `write_embedding()` function saves an `IndexedWordEmbedding` object to a binary file specified by `path`. The vocabulary can be filtered with `keep_words` and limited to `max_vocab_size`.
"""
function write_embedding(
    emb::IndexedWordEmbedding,
    path::AbstractString;
    max_vocab_size::Union{Int,Nothing}=nothing,
    keep_words::Union{Vector{String},Nothing}=nothing
)::Nothing

    # Should we preprocess the object?
    do_limit = !isnothing(max_vocab_size) && !(0 < max_vocab_size < emb.ntokens)
    do_subspace = !isnothing(keep_words) && (length(keep_words) > 0)
    if !(do_limit || do_subspace)
        jldsave(path; embedding=emb)
    end

    do_subspace && (emb = subspace(emb, keep_words))
    do_limit && (emb = limit(emb, max_vocab_size))
    jldsave(path; embedding=emb)
end
