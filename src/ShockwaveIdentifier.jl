module ShockwaveIdentifier

using Euler2D
using ShockwaveProperties
using Tullio
using Unitful
using LinearAlgebra
using Base.Iterators
using StaticArrays
using Printf
using LaTeXStrings
using Plots
using Dates
using Images


#GLOBAL THRESHOLD DEFAULT VALUES
#1D value for factor of maximum gradient
global const eps_1d = 0.5
#2D value for minimum d_1ρ cutoff
global const eps1_euler = 0.15
global const eps1_cell = 0.1

include("shock1D.jl")
include("shock2D.jl")

include("plotting.jl")
include("data_utils.jl")

export load_sim_data
export load_data

#=
Our developed approach to detect shock wave points in 1D, based on a simple approach with a gradient threshold.
The gradient threshold is the maximum gradient value multiplied by a constant around 0.5, which manages to detect all shocks in the test cases that we have received and to ignore high gradients caused by expansion waves, which can be visualized through the debug flag of generate_shock_plots1D.
We compare the weighted density gradient (density gradient multiplied by velocity) and velocity gradient. Both need to be above the threshold to suffice our shock condition. 

Input arguments:
	- frame: the frame-th step that of the simulation object that shall be processed.
    - data: EulerSim{1, 3, T} object generated by Euler2D.
    Shall return a list of indices where shockpoints are assumed.
=#
export findShock1D

#=
    This function iterates over all frames in the given data and finds shock points
    for each frame of the simulation object data::EulerSim{1, 3, T} using the findShock1D function.
=#
export findAllShocks1D

#=
Finds all shockpoints from the dataset data at the frame-th timestep and returns a list of their coordinates. Takes as inputs:
- frame: the frame-th step that of the simulation object that shall be processed.
- data: EulerSim{2, 4, T} object generated by Euler2D.
=#
export findShock2D

#=
findAllShocks2D takes as inputs:
- data::EulerSim{2, 4, T} object generated by Euler2D.
For all frames it detects the shock points.
WHy would we need this?
=#
export findAllShocks2D

#=
normalVectors takes as inputs:
- frame: the frame-th step that of the simulation object that shall be processed.
- data: EulerSim{2, 4, T} object generated by Euler2D.
- shocklist: List (or vector) of 2D points where a shock is detected.
For all points in the shocklist, normalVectors detects the direction of the shock by the pressure gradient's direction, and returns a vector of 2D directions,
=#
export normalVectors

export plotframe1D
export generate_shock_plots1D
export generate_shock_plots2D
export plotframe2D
export plotShock #wrapper for generate_shock_plots1D generate_shock_plots2D


end
