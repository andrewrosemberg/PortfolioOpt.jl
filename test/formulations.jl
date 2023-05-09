@testset "formulations.jl" begin
    @testset "PieceWiseUtility" begin
        @testset "constructor" begin
            @test PieceWiseUtility() isa PieceWiseUtility
            @test PieceWiseUtility([1.0]) isa PieceWiseUtility
            @test PieceWiseUtility([1.0], [0.0]) isa PieceWiseUtility
        end
        @testset "coefficients" begin
            @test coefficients(PieceWiseUtility()) == [1.0]
            @test coefficients(PieceWiseUtility([1.0])) == [1.0]
            @test coefficients(PieceWiseUtility([1.0], [0.0])) == [1.0]
            @test coefficients(PieceWiseUtility([1.0, 2.0], [0.0, 1.0])) == [1.0, 2.0]
        end
        @testset "intercepts" begin
            @test intercepts(PieceWiseUtility()) == [0.0]
            @test intercepts(PieceWiseUtility([1.0])) == [0.0]
            @test intercepts(PieceWiseUtility([1.0], [0.0])) == [0.0]
            @test intercepts(PieceWiseUtility([1.0, 2.0], [0.0, 1.0])) == [0.0, 1.0]
        end
    end
    @testset "PortfolioRiskMeasure" begin
        d = generate_gaussian_distribution(2, rng)
        @testset "ExpectedReturn" begin
            @testset "constructor" begin
                @test ExpectedReturn(d) isa ExpectedReturn
                @test ExpectedReturn(d, EstimatedCase) isa ExpectedReturn
                @test ExpectedReturn(d, WorstCase) isa ExpectedReturn
            end
            @testset "ambiguityset" begin
                @test ambiguityset(ExpectedReturn(d)) == d
                @test ambiguityset(ExpectedReturn(d, EstimatedCase)) == d
                @test ambiguityset(ExpectedReturn(d, WorstCase)) == d
            end
        end
        @testset "Variance" begin
            @testset "constructor" begin
                @test Variance(d) isa Variance
                @test Variance(d, EstimatedCase) isa Variance
                @test Variance(d, WorstCase) isa Variance
            end
            @testset "ambiguityset" begin
                @test ambiguityset(Variance(d)) == d
                @test ambiguityset(Variance(d, EstimatedCase)) == d
                @test ambiguityset(Variance(d, WorstCase)) == d
            end
        end
        @testset "SqrtVariance" begin
            @testset "constructor" begin
                @test SqrtVariance(d) isa SqrtVariance
                @test SqrtVariance(d, EstimatedCase) isa SqrtVariance
                @test SqrtVariance(d, WorstCase) isa SqrtVariance
            end
            @testset "ambiguityset" begin
                @test ambiguityset(SqrtVariance(d)) == d
                @test ambiguityset(SqrtVariance(d, EstimatedCase)) == d
                @test ambiguityset(SqrtVariance(d, WorstCase)) == d
            end
        end
        @testset "ConditionalExpectedReturn" begin
            numS = 100
            α = 0.05
            cexp1 = ConditionalExpectedReturn(d, numS)
            cexp2 = ConditionalExpectedReturn(d, numS; α=0.01, R=WorstCase)
            @testset "constructor" begin
                @test cexp1 isa ConditionalExpectedReturn
                @test cexp2 isa ConditionalExpectedReturn
            end
            @testset "ambiguityset" begin
                @test ambiguityset(cexp1) == d
                @test ambiguityset(cexp2) == d
            end
            @testset "sample_size" begin
                @test sample_size(cexp1) == numS
                @test sample_size(cexp2) == numS
            end
            @testset "alpha_quantile" begin
                @test alpha_quantile(cexp1) == α
                @test alpha_quantile(cexp2) == 0.01
            end
        end
        @testset "ExpectedUtility" begin
            @testset "constructor" begin
                @test ExpectedUtility(d, PieceWiseUtility()) isa ExpectedUtility
                @test ExpectedUtility(d, PieceWiseUtility(), EstimatedCase) isa ExpectedUtility
                @test ExpectedUtility(d, PieceWiseUtility(), WorstCase) isa ExpectedUtility
            end
            @testset "ambiguityset" begin
                @test ambiguityset(ExpectedUtility(d, PieceWiseUtility())) == d
                @test ambiguityset(ExpectedUtility(d, PieceWiseUtility(), EstimatedCase)) == d
                @test ambiguityset(ExpectedUtility(d, PieceWiseUtility(), WorstCase)) == d
            end
            @testset "utility" begin
                @test utility(ExpectedUtility(d, PieceWiseUtility())) == PieceWiseUtility()
                @test utility(ExpectedUtility(d, PieceWiseUtility(), EstimatedCase)) == PieceWiseUtility()
                @test utility(ExpectedUtility(d, PieceWiseUtility(), WorstCase)) == PieceWiseUtility()
            end
        end
    end
    @testset "Portfolio Terms" begin
        numA = 3
        d = generate_gaussian_distribution(numA, rng)
        max_std = 0.03
        R = -0.06
        l1_penalty = -0.0003
    
        m = VolumeMarket(numA)
        model, pf_w = market_model(m, DEFAULT_SOLVER; sense=MAX_SENSE)
        num_base_constraints = sum(num_constraints(model, F, S) for (F, S) in list_of_constraint_types(model))

        risk_m = ExpectedReturn(d)
        _cone = ConeRegularizer(MOI.NormOneCone(numA+1))

        @testset "ConeRegularizer" begin
            con = ConeRegularizer(MOI.NormOneCone(numA+1))
            @test con isa ConeRegularizer
            @test PortfolioOpt.cone(con) isa MOI.AbstractVectorSet
            @test PortfolioOpt.weights(con) isa UniformScaling
            con = ConeRegularizer(MOI.NormOneCone(numA+1), rand(numA, numA))
            @test con isa ConeRegularizer
            @test PortfolioOpt.cone(con) isa MOI.AbstractVectorSet
            @test PortfolioOpt.weights(con) isa Matrix

            @test PortfolioOpt.calculate_measure!(con, pf_w) isa JuMP.VariableRef
            @test sum(num_constraints(model, F, S) for (F, S) in list_of_constraint_types(model)) == num_base_constraints + 1
            num_base_constraints += 1
        end
        @testset "RiskConstraint Type" for con_type in [GreaterThan , LessThan, EqualTo]
            con = RiskConstraint(risk_m, con_type(0.0))
            @test con isa RiskConstraint
            @test PortfolioOpt.risk_measure(con) isa PortfolioOpt.PortfolioRiskMeasure
            @test PortfolioOpt.constant(con) == 0.0
            @test PortfolioOpt.uid(con) isa UUID

            @test PortfolioOpt.constraint!(model, con, pf_w) isa Union{JuMP.AbstractJuMPScalar, JuMP.VariableRef}
            @test sum(num_constraints(model, F, S) for (F, S) in list_of_constraint_types(model)) == num_base_constraints + 1
            num_base_constraints += 1
        end
        @testset "ObjectiveTerm" begin
            obj = ObjectiveTerm(risk_m, 0.01)
            @test obj isa ObjectiveTerm
            @test PortfolioOpt.term(obj) isa PortfolioOpt.PortfolioRiskMeasure
            @test PortfolioOpt.weight(obj) == 0.01
            @test PortfolioOpt.uid(obj) isa UUID
            obj = ObjectiveTerm(_cone, -0.01)
            @test obj isa ObjectiveTerm
            @test PortfolioOpt.term(obj) isa ConeRegularizer
            @test PortfolioOpt.weight(obj) == -0.01
            @test PortfolioOpt.uid(obj) isa UUID
        end
        @testset "PortfolioFormulation" begin
            risk_sqrt = SqrtVariance(d)
            con1 = RiskConstraint(risk_sqrt, LessThan(max_std))
            obj1 = ObjectiveTerm(risk_m)
            con2 = RiskConstraint(risk_sqrt, LessThan(max_std * 2))
            obj2 = ObjectiveTerm(_cone, l1_penalty)
            @test PortfolioFormulation(MAX_SENSE, obj1, con1) isa PortfolioFormulation
            @test PortfolioFormulation(MIN_SENSE, obj1, [con1, con2]) isa PortfolioFormulation
            @test PortfolioFormulation(MIN_SENSE, [obj1, obj2], con1) isa PortfolioFormulation
            pf = PortfolioFormulation(MAX_SENSE, [obj1, obj2], [con1, con2])
            @test pf isa PortfolioFormulation
            @test PortfolioOpt.sense(pf) == MAX_SENSE

            m = VolumeMarket(numA)
            model, pf_w = market_model(m, DEFAULT_SOLVER; sense=MAX_SENSE)
            num_base_constraints = sum(num_constraints(model, F, S) for (F, S) in list_of_constraint_types(model))

            pointers = portfolio_model!(model, pf, pf_w; record_measures=true)
            @test sum(num_constraints(model, F, S) for (F, S) in list_of_constraint_types(model)) == num_base_constraints + 5
            @test length(pointers) == 4
        end
    end
end
