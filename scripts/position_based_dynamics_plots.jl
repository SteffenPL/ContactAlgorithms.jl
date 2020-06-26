using Plots
using LinearAlgebra
using ForwardDiff
using ProgressMeter
using NearestNeighbors

# This is a one-file project. Simplicity is key.

# Setup

# walls
width = 10.
height = 10.

w1 = (pos=[0., 0.], dir=[0.,1.])
w2 = (pos=[0., height], dir=[0.,-1.])
w3 = (pos=[0., 0.], dir=[1.,0.])
w4 = (pos=[width, 0.], dir=[-1.,0.])

# wall constraints
function c_wall(x, wall)
    return dot( x - wall.pos, wall.dir )
end
# friction constraint
function c_wall_friction(x_old,x,wall)
    return dot( x_old - x, [-wall.dir[2],wall.dir[1]])
end

# initial state
N = 30
sqN = Int(round(sqrt(N)))
N = sqN * sqN
d = 2
x0 = hcat( [[x,y]
            for x in range(0.45*width, stop=0.55*width, length=sqN)
            for y in range(0.15*height, stop=0.75*height, length=sqN)]...  )
x0 += 0.1 * randn(d,N)
v0 = zeros(d,N)
R = 0.3

M = ones(N)' # mass
M_inv = M.^(-1) # inverse mass
μ = 0.3
# collision constraints
function c_coll(x, y, R)
    return sqrt(sum((x-y).^2)) - 2*R
end
function c_coll_friction(x_old,x,y_old,y,R)
    normal = x - y / sqrt(sum((x-y).^2))
    tangential = [-normal[2],normal[1]]
    return dot(x_old-y_old - (x - y),tangential)
end
# forces
F(x) = [0., -1.]
#F(x) = [5., 5.] .- x


scatter(x0[1,:], x0[2,:], markersize=R* (11*3.3),
        xlims=(-0.5,10.5), ylims=(-0.5,10.5), aspect_ratio=:equal)
savefig("initial_plot.png")

anim = Animation()
show_frames = true

dt = 0.5
n_stab = 2
n_steps = 10 # internal steps
dtₛ = dt / n_steps
t_end = 10.
α = 0.9
γ = 0.3

# simulate
let x0=x0, v0=v0
    global t, X, V
    t = 0
    X = [x0]
    V = [v0]


    p = Progress(Int(t_end/dt))
    while t < t_end
        t += dt

        x = X[end]
        v = V[end]

        # detect collisions
        balltree = KDTree(x, reorder=false)

        idxs_1 = inrange(balltree, x, 2.2*R, false)

        x_estim = x + dt * v + dt^2 * M_inv .* ( F(x) .- γ * v)
        idxs_2 = inrange(balltree, x_estim, 2.2*R, false)

        idxs = [ union(a,b) for (a,b) in zip(idxs_1,idxs_2) ]


        # solve collisions
        for n = 1:((t <= dt) ? 20*n_stab : n_stab)
            # particles constraints
            for i = 1:N
                for j = idxs[i]
                    if j != i
                        c = c_coll(x[:,i], x[:,j], R)
                        if c < 0
                            Dcᵢ = ForwardDiff.gradient(z -> c_coll(z, x[:,j], R), x[:,i])
                            Dcⱼ = ForwardDiff.gradient(z -> c_coll(x[:,i], z, R), x[:,j])
                            Δλᵢ = -c / ( (Dcᵢ)' * M_inv[i] * Dcᵢ + α )
                            Δλⱼ = -c / ( (Dcⱼ)' * M_inv[j] * Dcⱼ + α )
                            Δxᵢ = M_inv[i] * Dcᵢ * Δλᵢ
                            Δxⱼ = M_inv[j] * Dcⱼ * Δλⱼ
                            x[:,i] += Δxᵢ/n_stab
                            x[:,j] += Δxⱼ/n_stab
                        end
                    end
                end
            end
        end

        # solve collisions
        for n = 1:n_steps
            # predict next step
            x_old, x = x, x + dtₛ * v + dtₛ^2 * M_inv .* ( F(x) .- γ * v)
            #x_old, x = x, x + dtₛ * M_inv .* F(x)

            # wall constraints
            for wall in (w1,w2,w3,w4)
                for i = 1:N
                    c = c_wall(x[:,i], wall)
                    if c < 0
                        Dc = ForwardDiff.gradient(z -> c_wall(z,wall), x[:,i])
                        Δλ = -c / ( Dc' * M_inv[i] * Dc + α )
                        Δx = M_inv[i] * Dc * Δλ
                        x[:,i] += Δx
                        cf = c_wall_friction(x_old[:,i],x[:,i],wall)
                        Dcf = ForwardDiff.gradient(z-> c_wall_friction(x_old[:,i],z,wall), x[:,i])
                        Δλf = -cf/(Dcf' * M_inv[i] * Dcf + α)
                        Δλf = sign(Δλf)*min(μ*Δλ,abs(Δλf))
                        Δxf = M_inv[i]*Dcf*Δλf
                        x[:,i] += Δxf
                    end
                end
            end

            # particles constraints
            for i = 1:N
                for j = idxs[i]
                    if j != i
                        c = c_coll(x[:,i], x[:,j], R)
                        if c < 0
                            Dcᵢ = ForwardDiff.gradient(z -> c_coll(z, x[:,j], R), x[:,i])
                            Dcⱼ = ForwardDiff.gradient(z -> c_coll(x[:,i], z, R), x[:,j])
                            Δλᵢ = -c / ( (Dcᵢ)' * M_inv[i] * Dcᵢ + α )
                            Δλⱼ = -c / ( (Dcⱼ)' * M_inv[j] * Dcⱼ + α )
                            Δxᵢ = M_inv[i] * Dcᵢ * Δλᵢ
                            Δxⱼ = M_inv[j] * Dcⱼ * Δλⱼ
                            x[:,i] += 0.5*Δxᵢ
                            x[:,j] += 0.5*Δxⱼ
                            # friction(x_old,x,y_old,y,R)
                            Dcfᵢ = ForwardDiff.gradient(z -> c_coll_friction(x_old[:,i],z,x_old[:,j],x[:,j], R), x[:,i])
                            Dcfⱼ = ForwardDiff.gradient(z -> c_coll_friction(x_old[:,i], x[:,i],x_old[:,j],z, R), x[:,j])
                            cf = c_coll_friction(x_old[:,i], x[:,i],x_old[:,j],x[:,j], R)
                            Δλfᵢ = -cf / ( (Dcfᵢ)' * M_inv[i] * Dcfᵢ + α )
                            Δλfⱼ = -cf / ( (Dcfⱼ)' * M_inv[j] * Dcfⱼ + α )
                            Δλfᵢ = sign(Δλfᵢ)*min(μ*Δλᵢ,abs(Δλfᵢ))
                            Δλfⱼ = sign(Δλfⱼ)*min(μ*Δλⱼ,abs(Δλfⱼ))
                            Δxfᵢ = M_inv[i] * Dcfᵢ * Δλfᵢ
                            Δxfⱼ = M_inv[j] * Dcfⱼ * Δλfⱼ
                            x[:,i] += 0.5*Δxfᵢ
                            x[:,j] += 0.5*Δxfⱼ

                        end
                    end
                end
            end

            # update velocity
            v = (x - x_old) / dtₛ
            #v += dtₛ * M_inv .* F(x);
        end

        push!(X, x)
        push!(V, v)

        if show_frames
            scatter(X[end][1,:], X[end][2,:], markersize=R*(11*3.3),
                    xlims=(-0.5,10.5), ylims=(-0.5,10.5), aspect_ratio=:equal)
            frame(anim)
            display(current())
        end

        next!(p)
    end
end


if !show_frames
    anim = @animate for i = 1:length(X)
        scatter(X[i][1,:], X[i][2,:], markersize=R* (11*3.3),
                xlims=(-0.5,10.5), ylims=(-0.5,10.5), aspect_ratio=:equal)
    end
end

gif(anim, "output.gif", fps=30)
