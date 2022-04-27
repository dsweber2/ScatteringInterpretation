module ScatteringInterpretation
using ScatteringTransform
using Statistics, LineSearches, Optim
using ChainRules
using CUDA
using Flux: mse, update!
using FFTW
using Wavelets, ContinuousWavelets
using Zygote, Flux, Shearlab, LinearAlgebra, AbstractFFTs
using Flux
using FourierFilterFlux
using Adapt
using RecipesBase
using Base: tail
using ChainRulesCore
using Plots # who cares about weight really?
using Statistics, LineSearches, Optim
using Dates
using BlackBoxOptim
using FileIO
using Serialization

import Adapt: adapt
import ChainRules: rrule
import Zygote: has_chain_rrule, rrule
import Wavelets: eltypes

export buildRecord, expandRecord!, addCurrent!, makeObjFun, makeNormMatchObjFun, fitReverseSt, justTrain, maximizeSingleCoordinate, fitByPerturbing, perturb, genNoise, chooseLargest, continueTrain, fitUsingOptim
export saveSerial
export getPopulation, adjustPopulation!, plotFirstXEx, plotOverTitle
export perturbWorst, pinkStart, pullAndTrain
include("noiseMethods.jl")
include("storageUtils.jl")
include("objectiveFunctions.jl")
include("gradientBased.jl")
include("blackBoxBased.jl")
include("plotting.jl")


function pullAndTrain(setProbW, objSPC; normDiscount=0.01, N=1000)
    pop = setProbW.runcontrollers[end].optimizer.population.individuals
    trainI = rand(1:size(pop, 2))
    init = idct(pop[:, trainI], 1) # bring the one we're training back to the space domain
    opt = Nesterov(normDiscount * norm(init) / norm(gradient(objSPC, init)[1]), 0.9) # initial was 10
    objSPC(init)
    pathOfDescent, err = fitReverseSt(N, init, opt=opt, obj=objSPC, ifGpu=identity, adjust=true, relGradNorm=1e-18)
    pop[:, trainI] = dct(pathOfDescent[:, 1, 1, end])
    adjustPopulation!(setProbW, pop)
    return pathOfDescent, err
end
end
