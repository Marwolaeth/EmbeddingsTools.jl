using EmbeddingsTools
using Test
using Aqua
using JET

@testset "EmbeddingsTools.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(
            EmbeddingsTools,
            ambiguities=false
        )
    end
    @testset "Code linting (JET.jl)" begin
        JET.report_package(EmbeddingsTools; target_defined_modules=true)
    end
    @testset "EmbeddingsTools.jl package functionality" begin
        @testset "Reading .vec files" begin
            # Simple Reading
            @test read_vec("tiny.vec").vocab[end] ≡ "!"
            # Vocabulary Limit
            @test read_embedding("tiny.vec", max_vocab_size=2).vocab[end] ≡ "to"
            # Word List
            @test read_embedding("tiny.vec", keep_words=["!"]).vocab[1] == "!"
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
                    EmbeddingsTools.get_vector(index(read_vec("tiny.vec")), "Sinister")
                catch e
                    # This is what we are interested in: is the exception right?
                    isa(e, EmbeddingsTools.TokenNotFoundException)
                end
                @test all(
                    EmbeddingsTools.safe_get(
                        emb_ind,
                        "Sinister"
                    ) .≡ zeros(Float32, emb.ndims)
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
        end
    end
end
