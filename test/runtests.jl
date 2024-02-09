using EmbeddingsTools
using Test
using Aqua
using JET

@testset "EmbeddingsTools.jl" failfast = true verbose = true begin
    @testset "Code quality (Aqua.jl)" verbose = true begin
        Aqua.test_all(
            EmbeddingsTools,
            ambiguities=(recursive = false)
        )
    end
    @testset "Code linting (JET.jl)" begin
        JET.report_package(EmbeddingsTools; target_defined_modules=true)
    end
    @testset "EmbeddingsTools.jl package functionality" verbose = true begin
        @testset "Reading .vec files" begin
            # Simple Reading
            @test read_vec("tiny.vec").vocab[end] ≡ "!"
            # Vocabulary Limit
            @test read_embedding("tiny.vec", max_vocab_size=2).vocab[end] ≡ "to"
            # Word List
            @test read_embedding("tiny.vec", keep_words=["!"]).vocab[1] ≡ "!"
            # Conventional reading function
            @test EmbeddingsTools.read_giant_vec("tiny.vec").vocab[end] ≡ "!"
            @test EmbeddingsTools.read_giant_vec(
                "tiny.vec",
                max_vocab_size=2
            ).vocab[end] ≡ "to"
            @test EmbeddingsTools.read_giant_vec(
                "tiny.vec",
                keep_words=["to", "!"]
            ).vocab[1] ≡ "to"
            # Empty `keep_words`
            @test EmbeddingsTools.read_giant_vec(
                "tiny.vec",
                keep_words=Vector{String}()
            ).vocab[end] ≡ "!"
        end
        @testset "Utility Functions" begin
            emb = read_vec("tiny.vec")
            emb_ind = index(emb)
            words = ["!", "vainly", "I", "had", "sought", "to", "borrow"]
            @test sum(EmbeddingsTools._check_tokens(words, emb.vocab)) ≡ 2
            @test sum(EmbeddingsTools._check_tokens(words, emb_ind.vocab)) ≡ 2
            @test length(
                EmbeddingsTools._get_vocab_indices_safe(words, emb.vocab)
            ) ≡ length(words)
            @test length(
                EmbeddingsTools._get_vocab_indices_safe(words, emb_ind.vocab)
            ) ≡ length(words)
            @test length(
                EmbeddingsTools._get_vocab_indices(["!", "to"], emb.vocab)
            ) ≡ 2
            @test EmbeddingsTools._get_vocab_index(words[1], emb.vocab) ≡ 4
        end
        @testset "Embedding Operations" begin
            # Indexing
            @testset "Indexing Embeddings" begin
                emb = read_vec("tiny.vec")
                @test isa(index(read_vec("tiny.vec")).dict, Dict)
                @test length(index(emb).dict["!"]) ≡ read_vec("tiny.vec").ndims
            end
            # Limiting
            @testset "Embedding Limiting" begin
                emb = read_vec("tiny.vec")
                emb_ind = index(emb)
                @test limit(emb, 2).ntokens ≡ 2
                @test limit(emb, 2).vocab[end] ≡ "to"
                @test limit(emb_ind, 3).ntokens ≡ 3
                @test limit(emb_ind, 4).vocab[end] ≡ "!"
            end
            # Quering
            @testset "Embedding Quering" begin
                emb = read_vec("tiny.vec")
                emb_ind = index(emb)
                @test try
                    # This will throw an error in any case:
                    get_vector(emb_ind, "Sinister")
                catch e
                    # This is what we are interested in: is the exception right?
                    sprint(showerror, e)
                    @show isa(e, OutOfVocabularyException)
                    isa(e, OutOfVocabularyException)
                end
                @test all(
                    EmbeddingsTools.safe_get(
                        emb_ind,
                        "Sinister"
                    ) .≡ zeros(Float32, emb.ndims)
                )
                # Equivalence across embedding classes
                @test all(get_vector(emb, "!") .≡ get_vector(emb_ind, "!"))
                # Type stability
                @test isa(
                    get_vector(emb, "!"),
                    EmbeddingsTools.EmbeddingVectorView
                )
                @test isa(
                    get_vector(emb_ind, "!"),
                    EmbeddingsTools.EmbeddingVectorView
                )
                @test isa(
                    EmbeddingsTools.safe_get(emb_ind, "Sinister"),
                    EmbeddingsTools.EmbeddingVectorView
                )
            end
            # Subspacing
            @testset "Embedding Subspaces" begin
                emb = read_vec("tiny.vec")
                emb_ind = index(emb)
                tokens = ["!", "to"]
                @test subspace(emb, tokens).vocab[1] ≡ "!"
                @test subspace(emb, tokens).ndims ≡ emb.ndims
                @test subspace(emb, tokens).ntokens ≡ length(tokens)
                @test subspace(emb_ind, tokens).vocab[2] ≡ "to"
                @test subspace(emb_ind, tokens).ndims ≡ emb.ndims
                @test subspace(emb_ind, tokens).ntokens ≡ length(tokens)
            end
            # Dimensionality Reduction
            @testset "Embedding Dimensionality Reduction" begin
                emb = read_vec("tiny.vec")
                emb_ind = index(emb)
                @test isa(reduce_emb(emb, 2), WordEmbedding)
                @test isa(reduce_emb(emb, 2, method="SVD"), WordEmbedding)
                @test isa(reduce_emb(emb_ind, 2), IndexedWordEmbedding)
                @test isa(reduce_emb(emb_ind, 2, method="SVD"), IndexedWordEmbedding)
                @test try
                    # This should throw an error:
                    reduce_emb(emb, 2, method="ppca")
                catch e
                    # A specific kind of error
                    sprint(showerror, e)
                    @show isa(e, UnknownReductionMethodException)
                    isa(e, UnknownReductionMethodException)
                end
                @test size(reduce_emb(emb, 2).embeddings) ≡ (2, 4)
                @test size(reduce_emb(emb, 13).embeddings) ≡ (4, 4)
                @test reduce_emb(emb_ind, 1).ndims ≡ 1
                @test reduce_emb(emb_ind, 13, method="svd").ndims ≡ 4
            end
        end
        # Binary IO
        @testset "Binary Embeddings IO" begin
            emb = read_vec("tiny.vec")
            emb_ind = index(emb)
            @testset "Write Binary Embeddings" begin
                # Limited vocabulary
                @test begin
                    write_embedding(emb, "tiny.wem", max_vocab_size=1)
                    isfile("tiny.wem")
                end
                @test begin
                    write_embedding(emb_ind, "tiny.iem", max_vocab_size=1)
                    isfile("tiny.iem")
                end
                rm("tiny.iem")
                rm("tiny.wem")
                # Selected vocabulary
                @test begin
                    write_embedding(emb, "tiny.wem", keep_words=["to", "!"])
                    isfile("tiny.wem")
                end
                @test begin
                    write_embedding(emb_ind, "tiny.iem", keep_words=["to", "!"])
                    isfile("tiny.iem")
                end
                rm("tiny.iem")
                rm("tiny.wem")
                # No options
                @test begin
                    write_embedding(emb, "tiny.wem")
                    isfile("tiny.wem")
                end
                @test begin
                    write_embedding(emb_ind, "tiny.iem")
                    isfile("tiny.iem")
                end
            end
            @testset "Read Binary Embeddings" begin
                @test isa(read_emb("tiny.wem"), WordEmbedding)
                @test isa(read_indexed_emb("tiny.iem"), IndexedWordEmbedding)
                @test isa(read_embedding("tiny.wem"), WordEmbedding)
            end
        end
    end
end

# Cleanup
rm("tiny.iem")
rm("tiny.wem")

# Coverage and Cleanup
# Pkg.add("Coverage")
# using Coverage
# # process '*.cov' files
# coverage = process_folder()
# covered_lines, total_lines = get_summary(coverage)
# println("Coverage: $(round((covered_lines / total_lines) * 100, digits=2))%")
# LCOV.writefile("julia-lcov.info", coverage)
# Pkg.rm("Coverage")
# Pkg.resolve()
