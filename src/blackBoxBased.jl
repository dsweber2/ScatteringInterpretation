
"""
    adjustPopulation!(setProbW, popMat)
given a BlackBoxOptim `OptController` with some kind of population based method, set the population to be the matrix given in `popMat`; each column of `popMat` is a different individual.
"""
function adjustPopulation!(setProbW, popMat)
    for i = 1:size(popMat, 2)
        setProbW.optimizer.population[i] = copy(popMat[:, i])
    end
end

getPopulation(setProbW) = setProbW.runcontrollers[end].optimizer.population.individuals


function perturbRand(setProbW, scores; noiseFun=pinkNoise, frac=1 / 5, transform=x -> dct(x, 1), boundLevel=1e5)
    pop = setProbW.runcontrollers[end].optimizer.population.individuals
    Ndims = size(pop, 1)
    nPert = round(Int, size(pop, 2) * frac)
    theChosenPert = rand(1:size(pop, 2), nPert)
    # add pink noise with .1 of the norm to 100 randomly selected vectors
    perturbing = transform(noiseFun(Ndims, nPert))
    perturbing ./= [norm(x) for x in eachslice(perturbing, dims=2)]'
    pop[:, theChosenPert] += perturbing .* [norm(x) for x in eachslice(pop[:, theChosenPert], dims=2)]'
    adjustPopulation!(setProbW, pop)
end

function perturbWorst(setProbW, scores; noiseFun=pinkNoise, frac=1 / 5, transform=x -> dct(x, 1), boundLevel=1e5)
    pop = setProbW.runcontrollers[end].optimizer.population.individuals
    Ndims = size(pop, 1)
    popSize = size(pop, 2)
    nPert = round(Int, size(pop, 2) * frac)
    theChosenPert = sortperm(scores)[(end+1-nPert):end]
    # add pink noise
    perturbing = transform(noiseFun(Ndims, nPert))
    # with .1 of the norm to the worst frac vectors
    targValue = 0.3 * [norm(x) for x in eachslice(pop[:, theChosenPert], dims=2)]' ./ [norm(x) for x in eachslice(perturbing, dims=2)]'
    maxValue = minimum(boundLevel .- abs.(pop[:, theChosenPert]), dims=1) ./ maximum(abs.(perturbing), dims=1)

    perturbing .*= min.(targValue, maxValue)
    pop[:, theChosenPert] += perturbing
    adjustPopulation!(setProbW, pop)
end
"""
    fittingSinglePath(N, pathTuple, l=-1, CW=dog2, namedCW="default"; makeObjective=makeCoordMaxObj, popSize=500, totalMins=60, nMinsBBO=3, perturbRate=1 / 5, extraName="", reprFun=(x -> (dct(x, 1)), x̂ -> idct(x̂, 1)), normFun=norm, λ=1e-5, searchRange=1000, relScoreConvThresh=0.01, saveDir="", randFun=pinkNoise, perturbFun=perturbWorst, β=[2, 2, 1], averagingLength=[-1, -1, 2], outputPool=8, normalize=false, poolBy=3 // 2, pNorm=2, extraOctaves=0, kwargs...) # for St, but also useful for naming
- `N`: the length of the target
- `pathTuple`: the target path, e.g. (5,3) is the 5th frequency in the second layer and the 3rd frequency in the first layer
- `l=-1`: the target location. The default automatically caluclates the center of the signal in the output location.
- `CW=dog2`: the target wavelet type
- `namedCW="default"`: a short name for the target wavelet for printing purposes
- `makeObjective=makeCoordMaxObj`: the function used to construct the objective. By default, it maximizes the target location, and penalizes using the 2 norm.
- `popSize=500`: the number of candidate solutions used
- `totalMins=60`: the total training time allowed before cutting off the run
- `nMinsBBO=3`: the amount of time to run the evolution before perturbing
- `perturbRate=1 / 5`: the fraction of the population to perturb
- `extraName=""`: anything extra to put at the end of filename to distinguish from previous runs (which are otherwise overwritten)
- `reprFun=(x -> (dct(x,1)), x̂ -> idct(x̂,1))`: a pair of functions determining the domain in which the optimization takes place
- `normFun=norm`: the norm used in the objective fucntion. By default the 2 norm
- `λ=1e-5`: the penalty in the objective function
- `searchRange=1000`: the maximum magnitude of any particular coefficient
- `relScoreConvThresh=0.01`: the relative score convergence threshold. If the relative difference between the best and worst error is below this, return early
- `randFun=pinkNoise`: the function to create noise. By default it creates colored noise with exponent uniformly between .5 and 3.
- `perturbFun=perturbWorst`: the function to perturb out of premature convergence. By default it disturbs the worst examples using `randFun`.
- `saveDir=""`: where to save. If unspecified, most of the variables below are used in the name
- `β=[2,2,1]`:
- `averagingLength=[-1,-1,2]`:
- `outputPool=8`:
- `normalize=false`:
- `poolBy=3 // 2`:
- `pNorm=2`:
- `extraOctaves=0`:
- `kwargs`: any additional keyword arguments given to the `scatteringTransform` constructor.
"""
function fittingSinglePath(N, pathTuple; loc=-1, CW=dog2, namedCW="default", makeObjective=makeCoordMaxObj, popSize=500, totalMins=60, nMinsBBO=3, perturbRate=1 / 5, extraName="", reprFun=(x -> (dct(x, 1)), x̂ -> idct(x̂, 1)), normFun=x -> norm(x)^2, λ=1.0f-5, searchRange=1000, relScoreConvThresh=0.01, saveDir="", randFun=pinkNoise, perturbFun=perturbWorst, # unique to this function
    β=[2, 2, 1], averagingLength=[-1, -1, 2], outputPool=8, normalize=false, poolBy=3 // 2, pNorm=2, extraOctaves=0, kwargs...) # for St, but also useful for naming
    layer = length(pathTuple)
    St = scatteringTransform((N, 1, 1), 2; cw=CW, poolBy=poolBy, β=β, averagingLength=averagingLength, outputPool=outputPool, normalize=normalize, σ=abs, p=pNorm, extraOctaves=extraOctaves, kwargs...)
    x = randn(N, 1, 1)
    Sx = St(x)
    if loc <= 0
        loc = ceil(Int, 1 / 2 * size(Sx[layer], 1))
    end
    path = pathLocs(layer, (loc, pathTuple...))
    aveLenStr = mapreduce(x -> "$x", *, averagingLength)
    if saveDir == ""
        saveDir = joinpath("..", "results", "singleFits", "$(namedCW)", "poolBy$(round(poolBy, sigdigits=3))_outPool$(outputPool)_pNorm$(pNorm)_aveLen$(aveLenStr)_extraOct$(extraOctaves)", "lay$(layer)", "")
    end
    println("saving to $(saveDir)")
    if !isdir(saveDir)
        mkpath(saveDir)
    end
    locDir = joinpath(saveDir, string(["l$(layer-ii+1)=" * string(x) * "_" for (ii, x) in enumerate(pathTuple)]...))
    saveName = joinpath(locDir, "data$(extraName).jld2")
    saveSerialName = joinpath(locDir, "data$(extraName)")
    if !isdir(locDir)
        mkpath(locDir)
    end
    objective = makeObjective(path, St, pretransform=reprFun[2], normFun=normFun, λ=λ)
    fitnessCallsHistory = Array{Int,1}()
    fitnessHistory = Array{Float64,1}()
    callback = function rec(oc)
        push!(fitnessCallsHistory, num_func_evals(oc))
        push!(fitnessHistory, best_fitness(oc))
    end
    setProb = bbsetup(objective; SearchRange=(-searchRange, searchRange), NumDimensions=N, PopulationSize=popSize, MaxTime=1.0, TraceInterval=5, CallbackFunction=callback, CallbackInterval=0.5, TraceMode=:silent)
    println("pretraining done")
    bboptimize(setProb)
    pop = reprFun[1](randFun(N, popSize))
    adjustPopulation!(setProb, pop)
    nIters = ceil(Int, totalMins / nMinsBBO)
    for ii in 1:nIters
        println("----------------------------------------------------------")
        println("round $ii: GO")
        println("----------------------------------------------------------")
        bboptimize(setProb, MaxTime=nMinsBBO * 60.0) # take some black box steps for the whole pop
        pop = setProb.runcontrollers[end].optimizer.population.individuals
        scores = [objective(x) for x in eachslice(pop, dims=2)]
        iScores = sortperm(scores)

        relScoreDiff = abs(log(scores[iScores[1]]) - log(scores[iScores[round(Int, 4 * popSize / 5 - 1)]])) / abs(log(scores[iScores[1]]))
        if ii > 3 && relScoreDiff .< relScoreConvThresh
            println("finished after $(ii) rounds, relative Score diff= $(relScoreDiff)")
            break
        end
        # perturb a fifth to sample the full space
        perturbFun(setProb, scores; frac=perturbRate, transform=reprFun[1], noiseFun=randFun)
    end
    pop = setProb.runcontrollers[end].optimizer.population.individuals
    scores = [objective(x) for x in eachslice(pop, dims=2)]
    save(saveName, "pop", reprFun[2](pop), "scores", scores, "fitnessCallsHistory", fitnessCallsHistory, "fitnessHistory", fitnessHistory)
    saveSerial(saveSerialName, setProb)
end
