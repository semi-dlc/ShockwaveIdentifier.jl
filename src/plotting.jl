
"""
Returns plot boundaries of the simulation object sim for 1D case. (Used in plotframe)
"""
function plot_bounds(sim::EulerSim{N,NAXES,T}) where {N,NAXES,T}
	bounds = [(minimum(u), maximum(u)) for u ∈ eachslice(sim.u; dims=1)]
	return map(bounds) do (low, up)
		diffhalf = abs(up - low)/2
		mid = (up + low) / 2
		return (mid-1.1*diffhalf, mid+1.1*diffhalf)
	end
end



""" 
Plots physical property data of 1d case such as velocity, density, pressure without shockwave detection.
    Takes as arguments:
    - frame: the frame-th step that of the simulation object that shall be processed.
    - data: EulerSim{1, 3, T} object generated by Euler2D.
"""
function plotframe1D(frame, data::EulerSim{1, 3, T}) where {T}
	(t, u_data) = nth_step(data, frame)
	xs = cell_centers(data, 1)
	ylabels=[L"ρ", L"ρv", L"ρE"]
    bounds = plot_bounds(data)
	ps = [
		plot(xs, u_data[i, :], legend=(i==1), label=false, ylabel=ylabels[i], xticks=(i==3), xgrid=true, ylims=bounds[i], dpi=600) 
		for i=1:3]
	v_data = map(eachcol(u_data)) do u
		c = ConservedProps(u)
		v = velocity(c)[1]
	end
	p_data = map(eachcol(u_data)) do u
		c = ConservedProps(u)
		return uconvert(u"Pa", pressure(c, DRY_AIR))
	end
	pressure_plot = plot(xs, p_data, ylabel=L"P", legend=false)
	velocity_plot = plot(xs, v_data, ylabel=L"v", legend=false)
	titlestr = @sprintf "n=%d t=%.4e" frame t
	plot!(ps[1], ps[2], velocity_plot, pressure_plot, suptitle=titlestr, titlefontface="Computer Modern")
	savefig("plot1d_$(frame).png")
end

"""
	Plot the frame with shockwave detection
	Using only the velocity data yields good results.
    Takes as arguments:
    - frame: the frame-th step that of the simulation object that shall be processed.
    - data: EulerSim{1, 3, T} object generated by Euler2D.
    - shockwave_algorithm: function to determine shocks and takes frame, data as input arguments.
    - save: Flag whether the figures shall be stored directly.
    - debug: All plots or only density + ∇ density plots, which show shockpoints sufficiently.
    - threshold: value passed onto shockwave_algorithm as argument
"""
function plotframe1D(frame, data::EulerSim{1, 3, T},shockwave_algorithm; save = false, debug = false, threshold = eps_1d) where {T}
	(t, u_data) = nth_step(data, frame)
	xs = cell_centers(data, 1)
	ylabels=[L"ρ", L"ρv", L"ρE"]
    ps = []
    bounds = plot_bounds(data)
    x_shock = []
    # Detect the shockwave position using the provided algorithm
    try
        x_shock = shockwave_algorithm(frame, data, threshold = threshold)
    catch 
        x_shock = shockwave_algorithm(frame, data)
    end
    #x_shock = shockwave_algorithm(v_data)

    # density, momentum, and energy
    for i = 1:1
        p = plot(xs, u_data[i, :], legend=(i==1), label=false, ylabel=ylabels[i],
                 xticks=(i==3), xgrid=true, ylims=bounds[i], dpi=600)
        push!(ps, p)
    end

    #Get from data at frame density, presure, velocity
    density_data = vec(density_field(data, frame))
    v_data = vec(velocity_field(data, frame))
    p_data = vec(pressure_field(data, frame, DRY_AIR))

    # Pressure
    pressure_plot = plot(xs, p_data, ylabel=L"P", legend=false)
    scatter!(pressure_plot, [xs[x_shock]], [p_data[x_shock]], label="Shockwave",markersize=1, color="orange")

    # Velocity
    velocity_plot = plot(xs, v_data, ylabel=L"v", legend=false)
    scatter!(velocity_plot, [xs[x_shock]], [v_data[x_shock]], label="Shockwave",markersize=1, color="orange")


    # Gradient Pressure
    pressure_gradient = diff(p_data)
    push!(pressure_gradient, 0)

    pressure_gradient_plot = plot(xs, pressure_gradient, ylabel=L"\nabla P", legend=false)


    # Gradient Density
    density_gradient = diff(density_data)
    push!(density_gradient,0)

    density_gradient_plot = plot(xs, density_gradient, ylabel=L"\nabla ρ",  legend=false)
   
    # Gradient velocity
    velocity_gradient = diff(v_data)
    push!(velocity_gradient,0)

    velocity_gradient_plot = plot(xs, velocity_gradient, ylabel=L"\nabla v",  legend=false)

    # d1p: density * velocity_norm
    d1p = density_gradient .* ustrip.(v_data)[1:end]

    d1p_plot = plot(xs, d1p, ylabel=L"δ_1_ρ", legend=false)


    mach_data = vec(mach_number_field(data, frame, DRY_AIR))
    dmach = diff(mach_data)
    push!(dmach, 0)

    mach_plot = plot(xs, mach_data, ylabel=L"Mach", legend=false)
    mach_gradient_plot = plot(xs, dmach, ylabel=L"\nabla Mach",  legend=false)

    # Plotting
    scatter!(velocity_plot, [xs[x_shock]], [v_data[x_shock]], markersize=1, label="Shockwave", color="orange")
    scatter!(d1p_plot, [xs[x_shock]], [d1p[x_shock]], markersize=1,label="Shockwave", color="orange")
    scatter!(density_gradient_plot, [xs[x_shock]], [density_gradient[x_shock]], markersize=1,label="Shockwave", color="orange")
    scatter!(pressure_gradient_plot, [xs[x_shock]], [pressure_gradient[x_shock]], markersize=1,label="Shockwave", color="orange")
    scatter!(velocity_gradient_plot, [xs[x_shock]], [velocity_gradient[x_shock]], markersize=1,label="Shockwave", color="orange")        
    scatter!(mach_plot, [xs[x_shock]], [mach_data[x_shock]], markersize=1,label="Shockwave", color="orange")        

    #@show size.([v_data, d1p, mach_data, dmach, velocity_gradient, pressure_gradient, density_gradient])

    titlestr = @sprintf "n=%d t=%.4e" frame t

    """Plot the plots to a fig object. If debug, plot all. if no debug, only plot density + gradient. """
    if debug
        fig =  plot(ps[1], density_gradient_plot, d1p_plot, pressure_plot, pressure_gradient_plot, velocity_plot, velocity_gradient_plot, mach_plot,
            suptitle=titlestr, titlefontface="Computer Modern")
    else
        fig = plot(ps[1], density_gradient_plot, d1p_plot, velocity_plot, velocity_gradient_plot, mach_plot, mach_gradient_plot, suptitle=titlestr, titlefontface="Computer Modern")
    end
    if save == true
        savefig(fig, "plot1d_shock_$(frame)")
    end
    return fig
end

"""
Driver function to detect and plot shock wave points in 1D.

Input arguments:
- data: EulerSim object
Optional:
- save_dir: Directory where the figures shall be stored.
- shockwave_algorithm: The function that detects the shock points. E.g. findShock1D Shall take the arguments:
- threshold: value passed on to shockwave_algorithm
    Shall return a list of indices where shockpoints are assumed.
"""
function generate_shock_plots1D(data::EulerSim{1, 3, T}; save_dir::String = "frames", shockwave_algorithm = findShock1D, html = false, threshold = eps_1d) where {T}

    @info "Generating shock plots in 1D"

    # Generate the current date and time in the desired format
    datestr = Dates.format(now(), "mm-dd-HH-MM-SS")

    # Create general output directory if it doesn't exist
    if !isdir(save_dir)
        mkdir(save_dir)
    end
    # Create time-stamped directory if outputting to frames folder
    if save_dir == "frames"
        save_dir = "frames/$datestr"
    end
    if !isdir(save_dir)
        mkdir(save_dir)
        @info "Created folder $save_dir"
    end

    # Generate PNG files sequentially
    for i = 1:data.nsteps
        p = plotframe1D(i, data, shockwave_algorithm; threshold = threshold)
        filename = joinpath(save_dir, "output_$(datestr)_frame_$(lpad(i, 3, '0')).png")
        savefig(p, filename)
        if html
            @info "HTML NOT SUPPORTED YET. Write your own addon, please"
        end
        @info "Saved frame $i as $filename"
    end
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

#Plots for 2d grid with compute_data_function at data[frame] for a quantity, e.g. pressure.
function plotframe2D(frame, data::EulerSim{2, 4, T}, compute_data_function) where {T}
    (t, u_data) = nth_step(data, frame)
    xs, ys = cell_centers(data)
    plot_data = compute_data_function(frame, data)

    # Determine the header based on the data's units
    header = ""
    # Use eltype to check if the elements are Unitful.Quantity
    if eltype(plot_data) <: Unitful.Quantity
        unit_type = unit(plot_data[1,1])

        # Match against known units for pressure, velocity, or density
        if unit_type == u"Pa"  # Pressure (Pascals)
            header = "Pressure (Pa)"
        elseif unit_type == u"m/s"  # Velocity (meters per second)
            header = "Velocity (m/s)"
        elseif unit_type == u"kg/m^3"  # Density (kilograms per cubic meter)
            header = "Density (kg/m³)"
        else
            header = "Unknown Data"
        end
    else
        header = "Unknown Data"
    end

    #Plotting
    heatmap_plot = heatmap(xs, ys, plot_data, aspect_ratio=:1, size= (1000,1000), title=header, color=:viridis, xlabel="X", ylabel="Y")
    final_plot_layout = plot(heatmap_plot)
    return final_plot_layout
    #THIS RETURNS A PLOT OBJECT; THE PLOT IS NOT SAVED IN A FILE
end

#Plots heatmap of d1p. Mainly for debug purposes. Returns nothing!"
function plot_d1p(frame, data::Union{EulerSim{2,4,T}, CellBasedEulerSim{T}}, save_dir::AbstractString) where {T}
    datestr = Dates.format(now(), "mm-dd-HH-MM-SS")
    d1p = delta_1p(frame, data)
    #Plotting
    delta_1rho_plot = heatmap(
        d1p, 
        title="δ_1ρ step $frame", 
        xlabel="X-axis", 
        ylabel="Y-axis", 
        color=:viridis,
        aspect_ratio=1,  # Square heatmap
        size= (5000,5000) #5k resolution is too much for the disk.
        )
    plot(delta_1rho_plot)
    filename = joinpath(save_dir, "delta_1p_$(datestr)_frame_$(lpad(frame, 3, '0')).png")
    savefig(delta_1rho_plot, filename)

    return 1
end

#Plots heatmap of d2p. Mainly for debug purposes. Returns nothing!"
function plot_d2p(frame, data::Union{EulerSim{2,4,T}, CellBasedEulerSim{T}}, save_dir::AbstractString) where {T}
    d2p = delta_2p(frame, data)
    datestr = Dates.format(now(), "mm-dd-HH-MM-SS")
    #Plotting
    delta_2rho_plot = heatmap(
        d2p, 
        title="δ_2ρ step $frame", 
        xlabel="X-axis", 
        ylabel="Y-axis", 
        color=:viridis,
        aspect_ratio=1,  # Ensures the heatmap is square
        size= (5000,5000)
        )
        
    plot(delta_2rho_plot)
    filename = joinpath(save_dir, "delta_2p_$(datestr)_frame_$(lpad(frame, 3, '0')).png")
    savefig(delta_2rho_plot, filename)

end

"""
plotframe2D function to plot 2d frames including shock points and possibly normal vectors of the shock.

- frame, data: as usual.
- vectors, threshold: passed on through shockwave_algorithm.
- compute_data_function: over which data (e.g. density) the shock detection results shall be plotted.
- level: level of detection. 
    - 1: Coarse: usual
    - 2: improved: double (see findShock2D)
"""
function plotframe2D(frame, data::Union{EulerSim{2,4,T}, CellBasedEulerSim{T}}, compute_data_function, shockwave_algorithm; shockline = false, vectors = false, threshold = eps1_euler, level=1) where {T}
    (t, u_data) = nth_step(data, frame)
    xs, ys = cell_centers(data)
    shocklist = shockwave_algorithm(frame, data; threshold=threshold, level=level)
    plot_data = compute_data_function(frame, data)

    # Determine the header based on the data's units
    header = ""
    # Use eltype to check if the elements are Unitful.Quantity
    if eltype(plot_data) <: Unitful.Quantity
        unit_type = unit(plot_data[1,1])

        # Match against known units for pressure, velocity, or density
        if unit_type == u"Pa"  # Pressure (Pascals)
            header = "Pressure (Pa)"
        elseif unit_type == u"m/s"  # Velocity (meters per second)
            header = "Velocity (m/s)"
        elseif unit_type == u"kg/m^3"  # Density (kilograms per cubic meter)
            header = "Density (kg/m³)"
        else
            header = ""
        end
    else
        header = ""
    end

    header = header * "\n Shock points at frame $frame"
    if vectors
        header = header * " with normal shock directions"
    end

    #rotate matrix for correct plots
    plot_data_no_units = [x === nothing ? 0.0 : ustrip(x) for x in plot_data]   
    rows, cols = size(plot_data_no_units)
    rotated = Matrix{Float64}(undef, cols, rows) 

    for i in 1:rows 
        for j in 1:cols  
            rotated[cols - j + 1, i] = plot_data_no_units[i, j]  
        end
    end

    # Plotting heatmaps
    #Plotting
    heatmap_plot = heatmap(xs, ys, rotated, aspect_ratio=:1, size= (2000,2000), title=header, color=:viridis, xlabel="X", ylabel="Y",legendfontsize =20, ytickfontsize = 16)
    

    # Regrouped shock point and direction vectors
    shock_xs = [xs[i] for (i, j) in shocklist]
    shock_ys = [ys[j] for (i, j) in shocklist]
       

    # Overlay shock points on both plots
    #scatter!(heatmap_plot, shock_xs, shock_ys, color=:red,label="Shock Points", markersize=1, marker=:+)

    if shockline
        @info "Plot shockwave"
        rows, cols = data.ncells
        float_matrix = zeros(Float64, rows, cols)
        rotated_float_matrix = Matrix{Float64}(undef, cols, rows) 
        for (i, j) in shocklist
            float_matrix[i, j] = 1.0
        end
        for i in 1:rows 
            for j in 1:cols  
                rotated_float_matrix[cols - j + 1, i] = float_matrix[i, j]  
            end
        end
        contour!(xs, ys, rotated_float_matrix, levels=[0.5], linecolor=:red, linewidth=2)
    end
    
    
    
    if vectors
        @info "Plot normal vectors"


        """
        Used in the subsequent quiver plot to determine how few arrows are plotted. higher density -> lower amount of arrows, contraintuitively.
        Set to 1 to see all arrows. 
        """
        arrow_density = 10        
        
        """
        Used in the subsequent quiver plot to determine how many points are plotted per arrow  higher direction_density -> higher amount of pixels.
        Set to 1 to see all arrows. 
        """
        direction_density = 10
        if arrow_density < 1
            arrow_density = 1
        end

        # Regrouped shock point and direction vectors

        shock_xs = [xs[i] for (i, j) in shocklist[1:arrow_density:end]]
        shock_ys = [ys[j] for (i, j) in shocklist[1:arrow_density:end]]
        shock_dir = normalVectors(frame, data, shocklist[1:arrow_density:end])
        shockdir_xs = [dir[1] for dir in shock_dir]
        shockdir_ys = [dir[2] for dir in shock_dir]
         

        for i in range(0, 0.1, length=direction_density)
            scatter!(
                heatmap_plot,
                shock_xs .+ i .* shockdir_xs,  # Adjust points along direction
                shock_ys .+ i .* shockdir_ys,
                color=:green,
                label=false, 
                markersize=0.6, 
                marker=:+
            )
        end
        scatter!(
                heatmap_plot,
                shock_xs .+ 0.1 .* shockdir_xs,  # Adjust points along direction
                shock_ys .+ 0.1 .* shockdir_ys,
                color=:green,
                label="Shock directions", 
                markersize=0.6, 
                marker=:x
            )

    end
    
    final_plot_layout = plot(heatmap_plot)
    return final_plot_layout
        #THIS RETURNS A PLOT OBJECT; THE PLOT IS NOT SAVED IN A FILE
end

""" 
Analogue to 1D function
"""
function generate_shock_plots2D(data::Union{EulerSim{2,4,T}, CellBasedEulerSim{T}}; save_dir::String = "frames", shockwave_algorithm = findShock2D, compute_data_func = compute_density_data, html = false, vectors = false, threshold = 0.133, level = 1) where {T}
    #=
    deviating threshold here so i can detect when threshold was not changed externally by a kwarg. The value here is not of significance, as we see in "    
    if threshold == 0.133
        if data isa EulerSim{2,4,T}
            threshold = eps1_euler
        elseif data isa CellBasedEulerSim{T}
            threshold = eps1_cell
        end
    end"
    =#

    @info "Generating shock plots in 2D"

    # Generate the current date and time in the desired format
    datestr = Dates.format(now(), "mm-dd-HH-MM-SS")

    # Create general output directory if it doesn't exist
    if !isdir(save_dir)
        mkdir(save_dir)
    end
    # Create time-stamped directory if outputting to frames folder
    if save_dir == "frames"
        save_dir = "frames/$datestr"
    end
    if !isdir(save_dir)
        mkdir(save_dir)
    end

    if threshold == 0.133
        if data isa EulerSim{2,4,T}
            threshold = eps1_euler
        elseif data isa CellBasedEulerSim{T}
            threshold = eps1_cell
        end
    end

    @info "2d threshold value " threshold

    for step in 1:data.nsteps

        final_plot_layout = plotframe2D(step,data,compute_data_func,findShock2D; vectors=vectors, threshold = threshold, level = level)

        filename = joinpath(save_dir, "output_$(datestr)_frame_$(lpad(step, 3, '0'))")

        savefig(final_plot_layout, "$(filename).png")
        if html
            savefig(final_plot_layout, "$(filename)_zoomable.html")
        end
        @info "Saved frame $step as $filename"
    end
end


""" 
General-case wrapper discriminating 1D and 2D
"""
function generateShock(data::Union{EulerSim{1,3,T},EulerSim{2,4,T}, CellBasedEulerSim{T}}; save_dir::String = "frames", html = false, vectors = false, threshold = 0.133, level = 1) where {T}

    if data isa EulerSim{1,3,T}
        generate_shock_plots1D(data, save_dir = save_dir, shockwave_algorithm = findShock1D; html=html, threshold = threshold)
    elseif data isa EulerSim{2,4,T} || data isa CellBasedEulerSim{T}
        generate_shock_plots2D(data, save_dir = save_dir, shockwave_algorithm = findShock2D; html=html, threshold = threshold)
    else
        @error "Data argument is a " typeof(data) " and failed to be read."
    end
end