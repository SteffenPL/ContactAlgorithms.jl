#using Makie
using Plots
using LinearAlgebra
using ForwardDiff
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

# initial state
N = 500
sqN = Int(round(sqrt(N)))
N = sqN * sqN
d = 2
x0 = hcat( [[x,y]
            for x in range(0.25*width, stop=0.75*width, length=sqN)
            for y in range(0.25*height, stop=0.75*height, length=sqN)]...  )
x0 += 0.1 * randn(d,N)
v0 = zeros(d,N)
R = 0.3
M = ones(N)' # mass
M_inv = M.^(-1) # inverse mass

# collision constraints
function c_coll(x, y, R)
    return sqrt(sum((x-y).^2)) - R
end

# forces
F(x) = [0., -1.]
#F(x) = [5., 5.] .- x


using ProgressMeter

using NearestNeighbors

#positions = Node(Point2f0.(x0[1,:], x0[2,:]))
positions = x0[1,:], x0[2,:]
gr() # We will continue onward using the GR backend
plot( x0[1,:],  x0[2,:], seriestype = :scatter, title = "ParticlePosition")
#scatter!(scene, positions, markersize=R)
#xlims!(scene, -0.5, width+0.5)
#ylims!(scene, -0.5, height+0.5)
#save("initial.png", scene)

dt = 0.1
n_stab = 2
n_steps = 10 # internal steps
dtₛ = dt / n_steps
t_end = 1.
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
        balltree = KDTree(x, leafsize=3, reorder=false)

        idxs_1 = inrange(balltree, x, 3*R, false)

        x_estim = x + dt * v + dt^2 * M_inv .* ( F(x) .- γ * v)
        idxs_2 = inrange(balltree, x_estim, 3*R, false)

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
                            x[:,i] += Δxᵢ
                            x[:,j] += Δxⱼ
                        end
                    end
                end
            end

            # update velocity
            v = (x - x_old) / dtₛ
        end

        push!(X, x)
        push!(V, v)

        #positions[] = Point2f0.(x[1,:],x[2,:])
        next!(p)
    end
end



#positions = Node(Point2f0.(X[1][1,:], X[1][2,:]))
#scene = Scene(resolution =(1000,1000))
#scatter!(scene, positions, markersize=R)
#xlims!(scene, -0.5, width+0.5)
#ylims!(scene, -0.5, height+0.5)

#record(scene, "example.gif", 1:length(X)) do i
#    positions[] = Point2f0.(X[i][1,:],X[i][2,:])
#end
