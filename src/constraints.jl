using LinearAlgebra

# wall constraints
function c_wall(x, wall)
    return dot( x - wall.pos, wall.dir )
end
# friction constraint
function c_wall_friction(x_old,x,wall)
    return dot( x_old - x, [0 -1; 1 0] * wall.dir)
end
