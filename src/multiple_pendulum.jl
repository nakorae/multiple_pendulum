using DifferentialEquations
using LinearAlgebra


function calc_m′(m) #calculate m^\prime_i = \sum_{k = i}^n m_k
    m′ = similar(m)
    n = length(m)

    m′[end] = m[end]
    for i in reverse(1:n-1)
        m′[i] = m[i] + m′[i+1]
    end
    return m′
end

function A(m′, l, θ)
    n = length(θ)
    A = Matrix{eltype(θ)}(undef, n, n)
    for j in 1:n
        for i in 1:n
            A[i, j] = m′[max(i, j)] * l[j] * cos(θ[i] - θ[j])
        end
    end
    return A
end

function b(m′, l, g, θ, θ̇)
    b = similar(θ)
    for i in eachindex(b)
        b[i] = -sum(m′[max(i, j)] * l[j] * θ̇[j]^2 * sin(θ[i] - θ[j]) for j in eachindex(b)) - m′[i] * g * sin(θ[i])
    end
    return b
end

function f(u, p, t)
    n = length(u) ÷ 2
    θ = @view u[begin:n]
    θ̇ = @view u[n+1:end]
    m′ = @view p[begin:n]
    l = @view p[n+1:2n]
    g = p[end]

    return vcat(θ̇, inv(A(m′, l, θ)) * b(m′, l, g, θ, θ̇))
end

function simulation(m, l, g, θ0, θ̇0, tspan, saveat, abstol)
    u0 = vcat(θ0, θ̇0)
    m′ = calc_m′(m)
    p = vcat(m′, l, [g])
    prob = ODEProblem(f, u0, tspan, p)
    solve(prob, saveat=saveat, abstol=abstol)
end

function plot_pensdulum(θ, l)
    L = ceil(sum(l))
    n = length(θ)
    x = similar(θ, eltype(θ), n+1)
    y = similar(θ, eltype(θ), n+1)
    x[begin] = 0
    y[begin] = 0
    for i in 2:n+1
        x[i] = x[i-1] + l[i-1] * sin(θ[i-1])
        y[i] = y[i-1] - l[i-1] * cos(θ[i-1])
    end
    p = plot(x, y, xlim=(-L, L), ylim=(-L, L), aspect_ratio=1)
    # scatter!(p, x, y)
    p
end
#using sincos may be faster than this(?)


function plot_result(sol, l, fname, fps)
    n = length(sol.u[1]) ÷ 2
    anim = @animate for (i, e) in enumerate(sol.u)
        plot_pensdulum(e[1:n], l)
    end
    gif(anim, fname, fps=fps)
end

function main()
    simulation(ones(Float64,15), fill(0.1,15), 1.0, fill(5π/8, 15), zeros(Float64, 15), (0, 40.0), 0.2, 1e-9)
    plot_result(sol, ones(Float64, 15), "./result/result_15.gif", 10)
end