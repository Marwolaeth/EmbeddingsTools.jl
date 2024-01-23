using EmbeddingsTools
using Test
using Aqua
using JET
import CSV: CSV, CSV.File, CSV.Tables
using JLD2

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
    # Write your tests here.
end
