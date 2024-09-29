#version 1.4.1
#18.Sep.24

#not a notebook, but a script.

#A 500 x 500 grid has 25000 cell.

#We shall find approx. 1000 shockwaves in the simulation. +/- 1 dimension.

#TODO:
#Multithreading -> Parallelization
#More parameters for the function "process" - perhaps add few global constants?
    #Exporting pictures optional
#Modularize process() and this script in general, into julia module on git?
#Try/Catch blocks for dangerous parts 
#Profiling 
#Make nicer plots


# TODO
# NORMAL VECTORS -> finisihed
# ALIGNMENT WITH CIRCLE/ELLIPSE/LINE THROUGH HOUGHES TRANSFORM; MAYBE LEAST SQUARE FIT, maybe houghes transform????????? -> Hard
# EXPORT IN NUMERIC DATA (MATRIX OR TAPE)   -> EASY


using Markdown
using InteractiveUtils
using Base.Threads

using Profile


begin
	using Euler2D
	using LaTeXStrings
	using LinearAlgebra
	using Plots
	using Printf
	using ShockwaveProperties
	using Tullio
	using Unitful
	using Dates
	using FileIO
    using StaticArrays

    using Images
    using ImageFeatures
    using NearestNeighbors
	
end


#global flags
begin
    #Stores picture plots
    STORE_PLOT = true
    #Stores outputs as text files
    STORE_LOG_PLOT = false
    #Store final results, overriding the previous flags potentially. Yes = Yes. No != No.
    STORE_RES = true

    #verbose = 0 : Zero logging output
    #verbose = 1 : Some output
    #verbose = 2: Some more output
    #verboes > 10: Maximum
    verbose = 2
end

#Extract information from .tape and save as EulerSim
function load_sim_data(filename; T=Float64)
	## change this if you like
	#output_data_dir = joinpath(pwd(), "../data") |> realpath
	return Euler2D.load_euler_sim((filename); T)
end

#Returns plot boundaries for 1D case. 
function plot_bounds(sim::EulerSim{N,NAXES,T}) where {N,NAXES,T}
	bounds = [(minimum(u), maximum(u)) for u ∈ eachslice(sim.u; dims=1)]
	return map(bounds) do (low, up)
		diffhalf = abs(up - low)/2
		mid = (up + low) / 2
		return (mid-1.1*diffhalf, mid+1.1*diffhalf)
	end
end

#At the moment being ( september 2024), this struct is not used yet. it will be used to group shockwave s into groups (fronts)
struct Shockwave
	xBegin::Float64
	xEnd::Float64
	yBegin::Float64
	yEnd::Float64
end



# At 300K. This constant is used to work with older versions of Euler2D because of compatibility issues.
const DRY_AIR = CaloricallyPerfectGas(1004.9u"J/kg/K", 717.8u"J/kg/K", 0.0289647u"kg/mol")

"""Plots d1p, d2p according to the paper "Accurate detection of shock waves and shock interactions in
two-dimensional shock-capturing solutions"
save_dir : directory where all output files shall be stored. "dataSim/" by default.
data : EulerSim file which shall be loaded from a .tape file.
frames: list of frames/timesteps that the detector should process. default all.
"""
function process(data::EulerSim{2,4,T}, save_dir::AbstractString = "dataSim/", frames = 1:data.nsteps) where {T}
    datestr = Dates.format(now(), "mm-dd-HH-MM-SS")
    for frame in frames
        if verbose > 0
        @info "Started processing frame $frame"
        end

        #DETECTION ALGORITHM

        #find shock points
        #d2p, d1p according to article. 
        d1p1 = delta_1p(frame, data);

        d2p = delta_2p(frame, data, d1p1);

        blanked = blank(d1p1, d2p, 0.15, 10e-5);

        begin
            #Indicate shock points as booleans and store the points in a list. Probably not directly necessary.
            
            #OLD: to adapt to  hough_circle_gradient, we need to convert blanked to a boolean matrix to indicate edges.
            blanked_bool = falses(size(blanked))
            blanked_bool .= blanked .> 0 
            
            #store schock points
            #shocklist = findall(x -> x == true, blanked_bool)
            
            shocklist = []
            for i in 1:size(blanked_bool, 1)
                for j in 1:size(blanked_bool, 2)
                    if blanked_bool[i, j] == true
                        push!(shocklist, (i, j))
                    end
                end
            end
        end

        #normal vector ^= pressure gradient direction.
        begin
        p = compute_pressure_data(frame,data)
        dp = gradient_2d(frame,data,compute_pressure_data)

        shock_dir = [normalize(dp[i,j]) for (i,j) in shocklist]
        end

        #PLOTTING

        begin
                delta_1rho_plot = heatmap(
                d1p1, 
                title="δ_1ρ step $frame", 
                xlabel="X-axis", 
                ylabel="Y-axis", 
                color=:viridis,
                aspect_ratio=1,  # Ensures the heatmap is square
                #size= (5000,5000)
                )
                plot(delta_1rho_plot)

                if STORE_PLOT
                    filename = joinpath(save_dir, "delta_1p_$(datestr)_frame_$(lpad(frame, 3, '0')).png")
                    savefig(delta_1rho_plot, filename)
                end

                if STORE_LOG_PLOT
                    filename_w = joinpath(save_dir, "delta_1p_$(datestr)_frame_$(lpad(frame, 3, '0')).txt")
                    open(filename_w, "w") do file
                        print(file, d1p1)
                    end        
                end
                
        end

        begin
                #Plotting
                delta_2rho_plot = heatmap(
                d2p, 
                title="δ_2ρ step $frame", 
                xlabel="X-axis", 
                ylabel="Y-axis", 
                color=:viridis,
                aspect_ratio=1,  # Ensures the heatmap is square
                #size= (5000,5000)
                )
                
                plot(delta_2rho_plot)
                
        end

        if STORE_PLOT
            filename = joinpath(save_dir, "delta_2p_$(datestr)_frame_$(lpad(frame, 3, '0')).png")
            savefig(delta_2rho_plot, filename)
        end

        if STORE_LOG_PLOT
            filename_w = joinpath(save_dir, "delta_2p_$(datestr)_frame_$(lpad(frame, 3, '0')).txt")
            open(filename_w, "w") do file
                print(file, d2p)
            end        
        end

        begin
            #Plotting
                shockpoint_plot = heatmap(
                    blanked, 
                    title="Proposed Shocks at step $frame", 
                    xlabel="X-axis", 
                    ylabel="Y-axis", 
                    color=:viridis,
                    aspect_ratio=1,  # Ensures the heatmap is square
                    #size= (5000,5000)
                    )
                
                plot(shockpoint_plot)
        end

        if STORE_PLOT
            filename = joinpath(save_dir, "shockpoints_$(datestr)_frame_$(lpad(frame, 3, '0')).png")
            savefig(shockpoint_plot, filename)
        end

        if STORE_LOG_PLOT
            filename_w = joinpath(save_dir, "delta_1p_$(datestr)_frame_$(lpad(frame, 3, '0')).txt")
            open(filename_w, "w") do file
                print(file, blanked)
            end        
        end

        begin
                shockplot = scatter(
                [p[1] for p in shocklist], 
                [p[2] for p in shocklist], 
                color=:green,
                legend=false, 
                title = "Shocks with directions step $frame",
                xlabel="X", ylabel="Y")
            
            
            """
            Used in the subsequent quiver plot to determine how few arrows are plotted. higher density -> lower amount of arrows, contraintuitively.
            Set to 1 to see all arrows. 
            """
            arrow_density = 10
            
            if arrow_density < 1
                arrow_density = 1
            end
            quiver!(
                [p[1] for p in shocklist[1:arrow_density:end]], 
                [p[2] for p in shocklist[1:arrow_density:end]], 
                quiver=( [10 .* n[1] for n in shock_dir[1:arrow_density:end]], [10 .* n[2] for n in shock_dir[1:arrow_density:end]]), 
                color=:red, 
                legend=:false,  # Add legend to the top right corner
                title= "Suggested Shocks with directions step $frame",
                xlabel="X", ylabel="Y", 
                size = (1000,1000),
                
                arrow=Plots.arrow(:closed, :head,0.1, 0.1),
                )
            
            
            
            plot(shockplot) 

            if STORE_PLOT || STORE_RES
                filename = joinpath(save_dir, "shocks_wDirection_$(datestr)_frame_$(lpad(frame, 3, '0')).png")
                savefig(shockplot, filename)
            end

            if STORE_LOG_PLOT || STORE_RES
                shockdata = [ (p, q) for (p, q) in zip(shocklist, shock_dir) ]
                filename_w = joinpath(save_dir, "shocks_wDirection_$(datestr)_frame_$(lpad(frame, 3, '0')).txt")
                open(filename_w, "w") do file
                    print(file, shockdata)
                end        
            end

        end


    end

end

#Plots pressure and velocity for 2d grid
function plotframe2d(frame, data::EulerSim{2, 4, T}) where {T}
    (t, u_data) = nth_step(data, frame)
    xs, ys = cell_centers(data)

    # Compute velocity magnitudes
    velocity_data_magnitude = map(eachslice(u_data; dims=(2,3))) do u
        c = ConservedProps(u[1:end])
        velocity_in_ms = velocity(c, DRY_AIR)
        velocity_x = uconvert(u"m/s", velocity_in_ms[1])
        velocity_y = uconvert(u"m/s", velocity_in_ms[2])
        return sqrt(velocity_x^2 + velocity_y^2)  # Combine x and y components into a magnitude
    end

    # Compute pressure data
    pressure_data = map(eachslice(u_data; dims=(2,3))) do u
        c = ConservedProps(u[1:end])
        gas_in_pa = pressure(c, DRY_AIR)
        return uconvert(u"Pa", gas_in_pa)
    end

    # Plotting
    pressure_plot = heatmap(xs, ys, pressure_data, aspect_ratio=:equal, title="Pressure (Pa)", color=:viridis)
    velocity_plot = heatmap(xs, ys, velocity_data_magnitude, aspect_ratio=:equal, title="Velocity Magnitude (m/s)", color=:plasma)

    # Combine plots into a layout
    combined_plot = plot(pressure_plot, velocity_plot, layout = (1, 2))

    # Display and save the plot
    #debug display(combined_plot)
    savefig(combined_plot, "plot_frame2d.png")
end


#Returns matrix with pressure data
function compute_pressure_data(frame, data::EulerSim{2, 4, T}) where {T}
    (t, u_data) = nth_step(data, frame)
    pressure_data = map(eachslice(u_data; dims=(2,3))) do u
        c = ConservedProps(u[1:end])
        return uconvert(u"Pa", pressure(c, DRY_AIR))
        
    end

    return pressure_data
end


#Returns matrix with velocity data
function compute_velocity_data(frame, data::EulerSim{2, 4, T}) where {T}
    (t, u_data) = nth_step(data, frame)
    velocity_data = map(eachslice(u_data; dims=(2,3))) do u
        c = ConservedProps(u[1:end])
        #print(velocity(c, DRY_AIR))
        return velocity(c, DRY_AIR)
    end

    return velocity_data
end

#Returns matrix with density data
function compute_density_data(frame, data::EulerSim{2,4, T}) where {T}
    (t, u_data) = nth_step(data, frame)
    # Compute density data from the conserved properties
    density_data = map(eachslice(u_data; dims=(2,3))) do u
        c = ConservedProps(u[1:end])
        return c.ρ
    end

    return density_data
end


#Return matrix of normalized velocity vectors
function normalized_velocity(frame, data::EulerSim{2, 4, T}) where {T}
    velocity_xy = compute_velocity_data(frame, data)
    for i in 1:size(velocity_xy, 1)  # Iterate over rows
        for j in 1:size(velocity_xy, 2)  # Iterate over columns
            element = velocity_xy[i, j]
            #print(element)
            magnitude = sqrt(element[1]^2 + element[2]^2)
            if magnitude != 0u"m/s"  # Compare magnitude to zero with units
                velocity_xy[i, j] = (element / (magnitude * u"1"))u"m/s"  # Normalize with units
            end
        end
    end
    return velocity_xy
end


#=Returns matrix containing a vector (∂ρ/∂x, ∂ρ/∂y) at each point
 Function to compute the gradient, accounting for mesh grid sizes hx and hy
=#
 function gradient_2d(data_no_units, hx, hy)
    rows, cols = size(data_no_units)

    #debug
    #display(data_no_units)

    # Compute the gradients in both x and y directions
    gradients = imgradients(data_no_units, KernelFactors.ando3)

    #debug
    #display(gradients)

    # Allocate result array of SVectors (2-element statically sized vectors)
    gradient_2d_result = Array{SVector{2, Float64}, 2}(undef, rows, cols)

    # Fill the result array with scaled gradients
    for i in 1:rows
        for j in 1:cols
            gradient_2d_result[i, j] = @SVector [gradients[1][i, j] / hx, gradients[2][i, j] / hy]
        end
    end

    return gradient_2d_result
end

#Calls gradient_2d after formatting data accordingly
function gradient_2d(frame, data::EulerSim{2,4,T}, compute_data_function) where {T}

    # Check if the data is an EulerSim or a regular Float64 array
    if typeof(data) <: EulerSim
        # If it's an EulerSim, extract the density data at the specified frame
        if frame === nothing
            error("Frame number must be provided when using EulerSim data")
        end
        data_units = compute_data_function(frame, data)
        data_no_units = ustrip.(data_units)  # Strip units for computation
    elseif typeof(data) <: AbstractArray{<:Real, 2}
        # If it's a Float64 array, use it directly
        data_no_units = data
    else
        error("Unsupported data type. Input must be an EulerSim or a 2D Float64 array.")
    end

    #h := next cell.
    #assuming equidistant grid
    h_x = 1
    h_y = 1
    try
        h_x = cell_centers(data)[1][2] - cell_centers(data)[1][1]
        h_y =  cell_centers(data)[2][2] - cell_centers(data)[2][1]
    catch y
        warn("h was not computed correctly. assuming h=1: ", y)

    end
    return gradient_2d(data_no_units, h_x, h_y)

end

#Given a matrix A with elements of type [x,y] and a matrix B of type float, it returns A/B
function divide_matrices(matrix1, matrix2)
 
    if size(matrix1) != size(matrix2)
        throw(ArgumentError("Matrices must have the same dimensions"))
    end

    # Create a new matrix with the same dimensions
    result = similar(matrix1, eltype(matrix1))

    # Iterate through the matrices and perform the division
    for i in eachindex(matrix1)
        for j in eachindex(matrix1, 2)
            # Extract the values from matrix2 (divisor) and matrix1 (dividend)
            (x1, y1) = matrix1[i, j]
            q2 = matrix2[i, j]

            # Divide the elements by the value of the quantity (convert to Float64)
            x2 = x1 / Float64(q2)
            y2 = y1 / Float64(q2)

            # Store the result as a tuple in the new matrix
            result[i, j] = (x2, y2)
        end
    end

    return result
end


@doc raw""" ```\delta_{1\rho} = \frac{\mathbf{v}}{\|\mathbf{v}\|} \cdot \frac{\nabla \rho}{\rho_\infty} L \quad \text{(1a)}
 \delta_{2\rho} = \frac{\mathbf{v}}{\|\mathbf{v}\|} \cdot \nabla(\delta_{1\rho}) L \quad \text{(1b)}```

 ```\frac{v}{\|v\|}``` is the normalized velocity vector, ```\nabla \rho``` is the gradient of the density data, and ```L``` is the area of computation. The dot product of the normalized velocity and the gradient of the density data is calculated piecewise. The result is then multiplied by the area of computation to obtain the final value. The second equation calculates the dot product of the normalized velocity and the gradient of the first equation. The result is then multiplied by the area of computation to obtain the final value. The functions ```delta_1p``` and ```delta_2p``` implement these equations.
 ∇\delta_{1\rho} seems to be the gradient of \delta_{1ρ} with respect to x and y. This implies a 2D vector. 


 Derivation of the formulae from this (old) paper https://vis.cs.ucdavis.edu/papers/p87-ma.pdf .

 """
 #=
    Calculate the piecewise dot-product of the normalized velocity and the gradient of the density data. According to the paper "Accurate detection of shock waves and shock interactions in two-dimensional shock-capturing solutions.pdf"
=#
function delta_1p(frame, data::EulerSim{2,4,T}) where {T}

    x_width = last(cell_centers(data)[1]) - cell_centers(data)[1][1]
    y_width = last(cell_centers(data)[2]) - cell_centers(data)[2][1]

    #factor l as the area of computation / computational domain in 2D

    #because the other option (x*y) turns out to be quite large, we will try the grid size instead.
    #state of 10.09.2024: It seems to work.
    l = cell_centers(data)[2][2] - cell_centers(data)[2][1]

    ρ = compute_density_data(frame, data)
    ρ = ustrip(ρ)

    dRho = gradient_2d(frame, data, compute_density_data)
    
    dRho_normalized = divide_matrices(dRho, ρ)

    v = ustrip.(normalized_velocity(frame, data))

    # Convert each Tuple to an SVector
    sdRho = map(x -> SVector{2}(x...), dRho_normalized)
    sdRho = ustrip.(sdRho)
    
    
    #piecewise dot-product of v and \delta Rho
    d1p = map(dot, v, sdRho)
    #factor l, approx. the computational domain size (linearly)
    d1p *= l

    return d1p
end

"""Serves the purpose of finding zeros in discretized data such as d1p and d2p through sign changes. 
    Wherever a sign change occurs, the values are replaced with 0.
"""
function find_zeros!(discret)
    if ndims(discret) == 1
        
        for i in 2:length(discret)

            #signbit: True if negative. False if positive.
            if signbit(discret[i]) != signbit(discret[i-1])
                if discret[i] != 0 && discret[i-1] != 0
                    discret[i] = 0
                    discret[i-1] = 0
                end
            end
        end
    elseif ndims(discret) == 2
        rows, cols = size(discret)
        
        # Efficient handling for 2D array
        for i in 1:rows
            for j in 2:cols
                if signbit(discret[i, j]) != signbit(discret[i, j-1])
                    if discret[i, j] != 0 && discret[i, j-1] != 0
                        discret[i, j] = 0
                        discret[i, j-1] = 0
                    end
                end
            end
        end

        for j in 1:cols
            for i in 2:rows
                if signbit(discret[i, j]) != signbit(discret[i-1, j])
                    if discret[i, j] != 0 && discret[i-1, j] != 0
                        discret[i, j] = 0
                        discret[i-1, j] = 0
                    end
                end
            end
        end
    else
        error("Input must be either a 1D or 2D array.")
    end
    return 
end





"""
Turns 0 for shocks. Presumably, when the density gradient is also not zero.

    Calculate the piecewise dot-product of the normalized velocity and the gradient of the density data. According to the paper "Accurate detection of shock waves and shock interactions in two-dimensional shock-capturing solutions.pdf"
"""
function delta_2p(frame, data::EulerSim{2,4,T}) where {T}
 

    hx = cell_centers(data)[1][2] - cell_centers(data)[1][1]
    hy = cell_centers(data)[2][2] - cell_centers(data)[2][1]


    d1p = delta_1p(frame, data)


    #formerly problematic. switching to Images for gradient2d. the source of problem is not determined yet.
    dd1p = gradient_2d(d1p, hx, hy) 


    # Convert each Tuple to an SVector for compatibility reasons
    dd1p= map(x -> SVector{2}(x...), dd1p)
    #dd1p = dd1p * u"kg/m^3"
    
    d2p = map(dot, normalized_velocity(frame,data), dd1p)
    return ustrip.(d2p)
end

"""
Faster when d1p is precomputed, taking d1p already into account. 
optional argument d1p shall be a Matrix{T}

Turns 0 for shocks. Presumably, when the density gradient is also not zero.

    Calculate the piecewise dot-product of the normalized velocity and the gradient of the density data. According to the paper "Accurate detection of shock waves and shock interactions in two-dimensional shock-capturing solutions.pdf"
"""
function delta_2p(frame, data::EulerSim{2,4,T}, d1p::Matrix{T}) where {T}
 

    hx = cell_centers(data)[1][2] - cell_centers(data)[1][1]
    hy = cell_centers(data)[2][2] - cell_centers(data)[2][1]


    #formerly problematic. solved by 1. recursion 2. switching to Images for gradient2d
    dd1p = gradient_2d(d1p, hx, hy) 


    # Convert each Tuple to an SVector for compatibility reasons
    dd1p= map(x -> SVector{2}(x...), dd1p)
    #dd1p = dd1p * u"kg/m^3"
    
    d2p = map(dot, normalized_velocity(frame,data), dd1p)
    return ustrip.(d2p)
end


"""for 1D/scalar heatmaps (e.g. δ_1ρ  and δ_2ρ )"""	
function plot_1d_heatmap(magnitude, filename::String)
 
    heatmap_plot = heatmap(
    magnitude, 
    title="Heatmap", 
    xlabel="X-axis", 
    ylabel="Y-axis", 
    color=:viridis,
    aspect_ratio=1  # square plot
)
    if filename == ""
        filename = "heatmap.png"
    end
    
    savefig(heatmap_plot, filename)
end


"""blanking part:
wherever d2p == 0 and d1p != 0. Works abit besides of initial false-positive detection of expansion waves.

Needs d1p and d2p as arguments and finds the nul-points of d2p as possible extremals of d1p, and thus shockwaves. To eliminate the rest, only candidates with a d1p value above a certain threshold remain considered.

TODO :
Find eps1, eps2 so that it works. Possible, eps1 will have to depend on the intensity of shockwave. eps2 is set to take into account floating point arithmetic errors.

"""
function blank(d1p::Matrix{T}, d2p::Matrix{T}, eps1::T = 0.1, eps2::T = 10e-4) where {T<:Number}
    # Create a blank matrix of the same size as the input matrices
    blanked = zeros(T, size(d1p)) 

    #play around with these two parameters. Something is definitely off here.


    #Decreasing eps1 increases the amount of candidates. We set eps1 a bit higher. I am sure there shall be a way to adaptively calculate eps1 to decide what is a significant density gradient.


    #Increasing eps2 increases the amount of candidates, but might lead to the detection of standard propagation waves as shocks.


    #=
    d1p = ustrip.(d1p)
    d2p = ustrip.(d2p)
    =#

    find_zeros!(d2p)

    shock_counter = 0


    # Iterate over each element in the input matrices
    for i in 1:size(d1p, 1)
        for j in 1:size(d1p, 2)
            # If both d1p and d2p are zero, set the corresponding element in the blanked matrix to zero
            if abs(d1p[i,j]) > eps1
                
                if abs(d2p[i, j]) < eps2
                    blanked[i, j] = 1
                    
                    shock_counter += 1
                    #println("Shock with d1p $(d1p[i,j]) and d2p $(d2p[i,j]) at $i $j")
                else
                    #println("Potential Shock with d1p $(d1p[i,j]) and d2p $(d2p[i,j]) at $i $j") 
                end
            else
                blanked[i, j] = 0
            end
        end
    end

    println("Number of shockwaves detected: $shock_counter")
    return blanked
end


"""for 2D gradients (tuples)"""
function plot_gradient_heatmap(gradients, filename::String)
    # Calculate the magnitude of the gradient at each point
    magnitude = [sqrt(dx^2 + dy^2) for (dx, dy) in gradients]

    # Create the heatmap plot
    heatmap_plot = heatmap(
    magnitude, 
    title="Gradient Magnitude Heatmap", 
    xlabel="X-axis", 
    ylabel="Y-axis", 
    color=:viridis,
    aspect_ratio=1  # Ensures the heatmap is square
)

    # Save the plot to the specified filename
    savefig(heatmap_plot, filename)
end

function divide_matrices(matrix1, matrix2)
    # Ensure matrices are of the same size
    if size(matrix1) != size(matrix2)
        throw(ArgumentError("Matrices must have the same dimensions"))
    end

    # Create a new matrix with the same dimensions
    result = similar(matrix1, eltype(matrix1))

    # Iterate through the matrices and perform the division
    for i in 1:size(matrix1, 1)
        for j in 1:size(matrix1, 2)
            # Extract the values from matrix2 (divisor) and matrix1 (dividend)
            (x1, y1) = matrix1[i, j]
            q2 = matrix2[i, j]

            # Divide the elements by the value of the quantity (convert to Float64)
            x2 = x1 / Float64(q2)
            y2 = y1 / Float64(q2)

            # Store the result as a tuple in the new matrix
            result[i, j] = (x2, y2)
        end
    end

    return result
end

 


function test(filename::String = "dataSim/circular_obstacle_radius_1.celltape")
    # Load the simulation data
    FILENAME = filename
    DATA = load_sim_data(FILENAME)
    #boundary = plot_bounds(DATA)
    #For what is plot_bounds??!?
    # Generate the current date and time in the desired format
    datestr = Dates.format(now(), "mm-dd-HH-MM-SS")

    save_dir = "frames/$(datestr)"
    if !isdir(save_dir)
        mkdir(save_dir)
    end

    if verbose > 1
        @info "Start postprocessing $filename"
    end
    #function where main part happens. driver function to detect shock data over all frames of a Euler Sim.
    process(DATA, save_dir, 42:1:43)

    #plotFull(save_dir, DATA)

    if verbose > 1
        @info "Script terminated succesfuly with output to $save_dir"
    end

end 

#equivalent to main() in Python
if abspath(PROGRAM_FILE) == @__FILE__
    @info "calling test()"
    test()
end