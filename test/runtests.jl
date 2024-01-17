using EmbeddingsTools
using Test
using Aqua
using JET

@testset "EmbeddingsTools.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(EmbeddingsTools)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(EmbeddingsTools; target_defined_modules = true)
    end
    # Write your tests here.
end
