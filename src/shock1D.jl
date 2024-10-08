# At 300K
const DRY_AIR = CaloricallyPerfectGas(1004.9u"J/kg/K", 717.8u"J/kg/K", 0.0289647u"kg/mol")

#= 
	Computes the average mean difference across a dataset data (1D!).
	Used for determining shockwaves by comparing local gradient against average gradient.
=#
function averageGradient(data)
	
	unit = u"1"
	try
		unit = Unitful.unit(data[1])
	catch 
		println("Dimensionless data, or corruptes data? ")

	end

	divergence = 0unit


	for i in eachindex(data)[1:end-1]
		divergence += abs(data[i+1] - data[i])
	end
	
	#print("Div:", divergence)

	#ignore points where data value is very small, decreasing chance of false positives
	filtered_data = filter(x -> abs(x) > 10e-3unit, data)
	if isempty(filtered_data)
		return 0unit
	end

	return divergence / length(filtered_data)
end

#Returns the maximum value of gradient evalueated at all points (1d). 
function maxGradient(data)
	unit = u"1"
	try
		unit = Unitful.unit(data[1])
	catch 
		println("Dimensionless data, or corruptes data? ")

	end
	maximum_gradient = 0unit
	for i in eachindex(data)[1:end-1]
		if abs(data[i+1] - data[i]) > maximum_gradient
			maximum_gradient = abs(data[i+1] - data[i])
			#cprint(" Shockwave detected at ", i)
		end
	end
	return maximum_gradient
end

#=
	Find all points in data where the gradient exceeds the threshold.
=#
function discontinuities(data, threshold)
	indices = []
	gradient = diff(data)
    # Iterate over each gradient with its index
    for (i, difference) in enumerate(gradient)
        if abs(difference) >= threshold
            push!(indices, i)
            # println("Shockwave detected at index ", i)
        end
    end

    return indices
end

function findShock1D(frame, data::EulerSim{1, 3, T}) where{T}
    (t, u_data) = nth_step(data, frame)

    # Get Velocity out of ConservedProps
    v_data = map(eachcol(u_data)) do u
        c = ConservedProps(u)
        v = velocity(c)[1]
    end

    threshold = 0.5 * (averageGradient(v_data) + maxGradient(v_data))
	
	return discontinuities(v_data, threshold)
end