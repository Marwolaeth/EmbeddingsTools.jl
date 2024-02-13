# EmbeddingsTools

[![Build Status](https://github.com/Marwolaeth/EmbeddingsTools.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Marwolaeth/EmbeddingsTools.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![codecov](https://codecov.io/gh/Marwolaeth/EmbeddingsTools.jl/graph/badge.svg?token=3LLFUVWWFV)](https://codecov.io/gh/Marwolaeth/EmbeddingsTools.jl)
<!-- [![Codacy Badge](https://app.codacy.com/project/badge/Grade/f43ddb9608ec4d03a37f3aa0130f1c1e)](https://app.codacy.com/gh/Marwolaeth/EmbeddingsTools.jl/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) -->

**EmbeddingsTools.jl** is a Julia package that provides additional tools for working with word embeddings, complementing existing packages such as **[Embeddings.jl](https://github.com/JuliaText/Embeddings.jl)**. Please note that the compatibility with other packages is currently limited, namely, type conversions are currently missing. Still, this package can be used as a standalone tool for working with embedding vectors.

## Installation

You can install **EmbeddingsTools.jl** from GitHub through the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add https://github.com/Marwolaeth/EmbeddingsTools.jl.git
```

Or, within your Julia environment, use the following command:

```julia
using Pkg
Pkg.add("https://github.com/Marwolaeth/EmbeddingsTools.jl.git")
```

## Usage

The package is intended to read local embedding files, and it currently supports only text files (e.g., `.vec`) and binary Julia files. The package can perform basic operations on these embedding files.

The embeddings are represented as either `WordEmbedding` or `IndexedWordEmbedding` types. Both types contain an embedding table and a token vocabulary that is similar to embedding objects in **Embeddings.jl**. They also have `ntokens` and `ndims` fields to store the dimensionality of an embedding table. In addition, `IndexedWordEmbedding` objects have an extra lookup dictionary that maps its tokens to corresponding embedding vectors' views.

Indexing is useful when the embedding table must be aligned with a pre-existing vocabulary, such as the one obtained from a corpus of texts.

### Loading Word Embeddings

The original goal of the package was to allow users to read local embedding vectors in Julia. We discovered that this feature was quite limited in **Embeddings.jl**. For example, a user can manually download an embedding table, e.g. from [the FastText repository](https://fasttext.cc/docs/en/crawl-vectors.html) or [RusVectōrēs project](https://rusvectores.org/en/) (a collection of Ukrainian and Russian embeddings) and then read it into Julia using:

```julia
using EmbeddingsTools

# download and unzip the embedding file
## unless you prefer to do it manually
download(
    "https://rusvectores.org/static/models/rusvectores4/taiga/taiga_upos_skipgram_300_2_2018.vec.gz",
    "taiga_upos_skipgram_300_2_2018.vec.gz"
)
run(`gzip -dk taiga_upos_skipgram_300_2_2018.vec.gz`);

# Load word embeddings from a file
embtable = read_vec("taiga_upos_skipgram_300_2_2018.vec")
```

The `read_vec()` function is a basic function that reads embeddings. It takes two arguments: `path` and `delim` (the delimiter), and creates a `WordEmbedding` object using CSV.jl. This function reads the entire embedding table, which results in better performance due to its straightforward logic. However, it may fail to read embeddings with more than 500k words.

`read_embedding()` is an alternative function that provides more control options through keyword arguments. If `max_vocab_size` is specified, the function limits the size of the vector to that number. If a vector `keep_words` is provided, it only keeps those words. If a word in `keep_words` is not found, the function returns a zero vector for that word.

If the file is a `WordEmbedding` object within a Julia binary file (with extension `.jld` or in specific formats `.emb` or `.wem`), the entire embedding is loaded, and keyword arguments are not applicable. You can also use the `read_emb()` function directly on binary files. See `?write_embedding` for saving embedding objects to Julia binary files to read them faster in the future.

```julia
# Load word embeddings for 10k most frequent words in a model
embtable = read_embedding(
    "taiga_upos_skipgram_300_2_2018.vec",
    max_vocab_size=10_000
)
```

### Creating Embedding Indices

There are some differences in the behavior of certain functions in **EmbeddingsTools.jl**, depending on whether the embedding table object contains a lookup dictionary or not. If the object contains a lookup dictionary, then it is referred to as an object of type `IndexedWordEmbedding`, which is considerably faster to operate on. On the other hand, if it does not contain the lookup dictionary, then it is referred to as an object of type `WordEmbedding`, which takes a bit of time to index and should only be done when necessary. To index an embedding object, you can either call `IndexedWordEmbedding()` (which is a constructor function) or `index()` on the object.

```julia
# These are equivalent
embtable_ind = IndexedWordEmbedding(embtable)
embtable_ind = index(embtable)
```

### Quering Embeddings

We can use the `get_vector()` function with either indexed or simple embeddings table to obtain a word-vector for a given word:

```julia
get_vector(embtable, "человек_NOUN")
get_vector(embtable_ind, "человек_NOUN")
```

### Limiting Embedding Vocabulary

Regardles of whether we have read the embedding with limited vocabulary size or not, we can limit it with the `limit()` function:

```julia
small_embtable = limit(embtable, 111)
```

### Embedding Subspaces

At times, we may need to adjust an embedding table to match a set of words or tokens. This could be the result of pre-processing a corpus of text documents using the **[TextAnalysis.jl](https://github.com/JuliaText/TextAnalysis.jl)** package. The `subspace()` function can be used to create a new `WordEmbedding` object from an existing embedding and a vector of strings containing the words or tokens of interest. The order of the new embedding vectors corresponds to the order of the input tokens. If a token is not present in the source embedding vocabulary, a zero vector is returned for that token.

It's important to note that the `subspace()` method performs much faster when used with an indexed embedding object.

```julia
words = embtable.vocab[13:26]
embtable2 = subspace(embtable_ind, words)
```

### Dimensionality Reduction

The `reduce_emb()` function allows you to decrease the size of embedding objects, whether they are indexed or not. You can choose between two reduction techniques (specified using the `method` keyword): `pca` (the default) for Principal Component Analysis, or `svd` for Singular Value Decomposition.

```julia
# Reduce the dimensionality of the word embeddings using PCA or SVD
embtable20 = reduce_emb(embtable, 20)
embtable20_svd = reduce_emb(embtable, 20, method="svd")
```

## Compatibility

As of the current version, **EmbeddingsTools.jl** has limited compatibility with the package that has inspired the entire project. We are actively working on expanding compatibility and interoperability with a wider range of packages.

## Contributing

We welcome contributions from the community to enhance the functionality and compatibility of **EmbeddingsTools.jl**. If you encounter any issues or have ideas for improvement, please feel free to open an issue or submit a pull request on our GitHub repository.

## License

EmbeddingsTools.jl is provided under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](https://github.com/Marwolaeth/EmbeddingsTools.jl/blob/main/LICENSE) file for more details.
