using Plots
using LinearAlgebra
using ForwardDiff
using ProgressMeter
using NearestNeighbors

using ContactAlgorithms

# This is a one-file project. Simplicity is key.

# Setup

# walls
width = 10.
height = 10.

w1 = (pos=[0., 0.], dir=[0.,1.])
w2 = (pos=[0., height], dir=[0.,-1.])
w3 = (pos=[0., 0.], dir=[1.,0.])
w4 = (pos=[width, 0.], dir=[-1.,0.])
walls = (w1,w2,w3,w4)

# initial state
N = 30
sqN = Int(round(sqrt(N)))
N = sqN * sqN
d = 2
x0 = hcat( [[x,y]
            for x in range(0.25*width, stop=0.75*width, length=sqN)
            for y in range(0.15*height, stop=0.75*height, length=sqN)]...  )
x0 += 0.1 * randn(d,N)
v0 = zeros(d,N)
R = 0.3

M = ones(N)' # mass
M_inv = M.^(-1) # inverse mass
μ = 0.1

# small steps stuff
d0 = zeros(1,N)
d0wall = zeros(1,N)
v_max = 5
# collision constraints
function c_coll(x, y, R)
    return sqrt(sum((x-y).^2)) - 2*R
end

function c_coll_vmax(x, y, R, i)
    return sqrt(sum((x-y).^2)) - 2*R+max(d0[i]-v_max*dtₛ,0)
end
# wall constraints
function c_wall_vmax(x, wall,i)
    return dot( x - wall.pos, wall.dir )+max(d0wall[i]-v_max*dtₛ,0)
end

# forces
F(x) = [0., -5.]
#F(x) = [5., 5.] .- x


scatter(x0[1,:], x0[2,:], markersize=R* (11*3.3),
        xlims=(-0.5,10.5), ylims=(-0.5,10.5), aspect_ratio=:equal)
savefig("initial_plot.png")

anim = Animation()
show_frames = true

dt = 0.5
#n_stab = 2
n_steps = 10 # internal steps
dtₛ = dt / n_steps
t_end = 10.
α = 0.0
γ = 0.3
R_collision_detection = 4. * R

# simulate
let x0=x0, v0=v0
    global t, X, V, d0, d0wall
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

        idxs = inrange(balltree, x, R_collision_detection, false)
        for i = 1:N
            for j = idxs[i]
                if j < i
                    c = c_coll(x[:,i], x[:,j], R)
                    if c < 0
                        d0[i] = -c
                    end
                end
            end
        end
        for wall in (w1,w2,w3,w4)
            for i = 1:N
                c = c_wall(x[:,i], wall)
                if c < 0
                    d0wall[i] = -c
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
                    c = c_wall_vmax(x[:,i], wall,i)
                    if c < 0
                        Dc = ForwardDiff.gradient(z -> c_wall_vmax(z,wall,i), x[:,i])
                        Δλ = -c / ( Dc' * M_inv[i] * Dc + α )
                        Δx = M_inv[i] * Dc * Δλ
                        x[:,i] += Δx
                    end
                end
            end

            # particles constraints
            for i = 1:N
                for j = idxs[i]
                    if j < i
                        c = c_coll_vmax(x[:,i], x[:,j], R,i)
                        if c < 0
                            Dcᵢ = ForwardDiff.gradient(z -> c_coll_vmax(z, x[:,j], R, i), x[:,i])
                            Dcⱼ = ForwardDiff.gradient(z -> c_coll_vmax(x[:,i], z, R, i), x[:,j])
                            denom = (Dcᵢ)' * M_inv[i] * Dcᵢ + (Dcⱼ)' * M_inv[j] * Dcⱼ + α
                            Δλᵢ = -c / denom
                            Δλⱼ = -c / denom
                            Δxᵢ = M_inv[i] * Dcᵢ * Δλᵢ
                            Δxⱼ = M_inv[j] * Dcⱼ * Δλⱼ
                            x[:,i] += Δxᵢ
                            x[:,j] += Δxⱼ
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

gif(anim, "output.gif", fps=10)
