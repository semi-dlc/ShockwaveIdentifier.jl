#Returns plot boundaries for 1D case. (Used in plotframe)
function plot_bounds(sim::EulerSim{N,NAXES,T}) where {N,NAXES,T}
	bounds = [(minimum(u), maximum(u)) for u ∈ eachslice(sim.u; dims=1)]
	return map(bounds) do (low, up)
		diffhalf = abs(up - low)/2
		mid = (up + low) / 2
		return (mid-1.1*diffhalf, mid+1.1*diffhalf)
	end
end


# Plots data of 1d case without shockwave
function plotframe1D(frame, data::EulerSim{1, 3, T}, bounds) where {T}
	(t, u_data) = nth_step(data, frame)
	xs = cell_centers(data, 1)
	ylabels=[L"ρ", L"ρv", L"ρE"]
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
	savefig("plot1d.png")
    gui()
end

#=
	Plot the frame with shockwave detection
	Using only the velocity data yields good results.
=#
function plotframe1D(frame, data::EulerSim{1, 3, T}, bounds, shockwave_algorithm) where {T}
	(t, u_data) = nth_step(data, frame)
	xs = cell_centers(data, 1)
	ylabels=[L"ρ", L"ρv", L"ρE"]
    ps = []


    # Detect the shockwave position using the provided algorithm
    x_shock = shockwave_algorithm(frame, data)
    #x_shock = shockwave_algorithm(v_data)

    # density, momentum, and energy
    for i = 1:3
        p = plot(xs, u_data[i, :], legend=(i==1), label=false, ylabel=ylabels[i],
                 xticks=(i==3), xgrid=true, ylims=bounds[i], dpi=600)
        scatter!(p, [xs[x_shock]], [u_data[i, x_shock]], label="Shockwave", color="orange")
        push!(ps, p)
    end

    # Get Velocity out of ConservedProps
    v_data = map(eachcol(u_data)) do u
        c = ConservedProps(u)
        v = velocity(c)[1]
    end

    # Get Pressure out of ConservedProps
    p_data = map(eachcol(u_data)) do u
        c = ConservedProps(u)
        return uconvert(u"Pa", pressure(c, DRY_AIR))
    end

    # Pressure
    pressure_plot = plot(xs, p_data, ylabel=L"P", legend=false)
    scatter!(pressure_plot, [xs[x_shock]], [p_data[x_shock]], label="Shockwave", color="orange")

    # Velocity
    velocity_plot = plot(xs, v_data, ylabel=L"v", legend=false)
    scatter!(velocity_plot, [xs[x_shock]], [v_data[x_shock]], label="Shockwave", color="orange")

    # Gradient Pressure
    pressure_gradient = diff(p_data)
    gradient_xs = xs[1:end-1] + diff(xs)/2 # Adjust xs for gradient plot
    pressure_gradient_plot = plot(gradient_xs, pressure_gradient, ylabel=L"\nabla P", legend=false)

    # Gradient Density
    density_gradient = diff(u_data[1, :])
    density_gradient_plot = plot(gradient_xs, density_gradient, ylabel=L"\nabla ρ", legend=false)

    # Plotting
    scatter!(pressure_gradient_plot, [xs[x_shock]], [pressure_gradient[x_shock]], label="Shockwave", color="orange")
    scatter!(density_gradient_plot, [xs[x_shock]], [density_gradient[x_shock]], label="Shockwave", color="orange")

   
    titlestr = @sprintf "n=%d t=%.4e" frame t

    # Plot of a plot
    plot(ps[1], density_gradient_plot, ps[2], pressure_plot, pressure_gradient_plot, velocity_plot, 
         suptitle=titlestr, titlefontface="Computer Modern")
end

function generate_shock_plots1D(filename::String; save_dir::String = "frames", shockwave_algorithm = findShock1D)
    # Load simulation data
    DATA = load_sim_data(filename)
    boundary = plot_bounds(DATA)

    # Generate the current date and time in the desired format
    datestr = Dates.format(now(), "mm-dd-HH-MM-SS")

    # Create directory if it doesn't exist
    if !isdir(save_dir)
        mkdir(save_dir)
    end

    # Generate PNG files sequentially
    for i = 1:DATA.nsteps
        p = plotframe1D(i, DATA, boundary, shockwave_algorithm)
        filename = joinpath(save_dir, "output_$(datestr)_frame_$(lpad(i, 3, '0')).png")
        savefig(p, filename)
        println("Saved frame $i as $filename")
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

#Plots pressure and velocity for 2d grid. For debugging purposes mainly!
function plotframe2D(frame, data::EulerSim{2, 4, T}) where {T}
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

#Plots heatmap of d1p 
function plot_d1p(frame, data::EulerSim{2,4,T}, save_dir::AbstractString) where {T}
    datestr = Dates.format(now(), "mm-dd-HH-MM-SS")
    d1p = delta_1p(frame, data)
    #Plotting
    delta_1rho_plot = heatmap(
        d1p, 
        title="δ_1ρ step $frame", 
        xlabel="X-axis", 
        ylabel="Y-axis", 
        color=:viridis,
        aspect_ratio=1,  # Ensures the heatmap is square
        #size= (5000,5000)
        )
    plot(delta_1rho_plot)
    filename = joinpath(save_dir, "delta_1p_$(datestr)_frame_$(lpad(frame, 3, '0')).png")
    savefig(delta_1rho_plot, filename)


end

#Plots heatmap of d2p 
function plot_d2p(frame, data::EulerSim{2,4,T}, save_dir::AbstractString) where {T}
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
        #size= (5000,5000)
        )
        
    plot(delta_2rho_plot)
    filename = joinpath(save_dir, "delta_2p_$(datestr)_frame_$(lpad(frame, 3, '0')).png")
    savefig(delta_2rho_plot, filename)

end

"""
Plots detected shock points not including normal vectors.
Use STORE_PLOT because of compatibility reasons. Should be set to true when a plot is desired.
blanked: returned values from blank()
"""
function plot_shock_points(frame, data::EulerSim{2,4,T}, save_dir::AbstractString, blanked, STORE_PLOT::Bool=True) where {T}
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
    if STORE_PLOT
        filename = joinpath(save_dir, "shockpoints_$(datestr)_frame_$(lpad(frame, 3, '0')).png")
        savefig(shockpoint_plot, filename)
    end
end

"""
Plots detected shock points including normal vectors.
Use STORE_PLOT because of compatibility reasons. Should be set to true when a plot is desired.

shocklist: List of shock points. Might make sense to incorporate a matrix in the form of blanked later.

"""
function plot_shock(frame, data::EulerSim{2,4,T}, save_dir::AbstractString, shocklist, STORE_PLOT::Bool=True) where {T}
    shock_dir = [normalize(gradient_2d(frame,data,compute_pressure_data)[i,j]) for (i,j) in shocklist]
    
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
        #higher arrow density -> more space between each arrow
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

    if STORE_PLOT 
        filename = joinpath(save_dir, "shocks_wDirection_$(datestr)_frame_$(lpad(frame, 3, '0')).png")
        savefig(shockplot, filename)
    end
end
