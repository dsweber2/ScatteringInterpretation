module ScatteringInterpretation
using ScatteringTransform
using Random
using Statistics
using CUDA
using FFTW
using Wavelets, ContinuousWavelets
using LinearAlgebra, AbstractFFTs
using FourierFilterFlux
using Adapt
using RecipesBase
using Plots # who cares about weight really?
using Dates
using FileIO
using Serialization

import Adapt: adapt
import ChainRules: rrule
import Zygote: has_chain_rrule, rrule
import Wavelets: eltypes

export makeObjFun, saveSerial
export getPopulation, adjustPopulation!, plotFirstXEx, plotOverTitle
export perturbWorst, pinkStart
include("noiseMethods.jl")
include("storageUtils.jl")
include("objectiveFunctions.jl")
include("blackBoxBased.jl")
include("plotting.jl")
end
# actually used in the dissertation: saveSerial, adjustPopulation!, perturbWorst, pinkStart,
# in dissertation, codebase not used in the final results: plotFirstXEx, plotOverTitle
# so it looks like I should just dump a bunch of this
