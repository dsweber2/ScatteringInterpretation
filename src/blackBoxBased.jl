
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


"""
    bandpassed,objValue, freqInd = BandpassStayInFourier(x,FourierObj)
given a dct representation of a signal `x` and an objective function `FourierObj` which operates on the dct representation of a signal, find the frequency index a super naive way of finding a bandpass threshold so that x (which is represented as dct of the target)  FourierObj
"""
function BandpassStayInFourier(x, FourierObj)
    flattenFrom(x, i) = cat(x[1:i, axes(x)[2:end]...], zeros(size(x, 1) - i), dims=1)
    evalFilters = [FourierObj(flattenFrom(x, ncut)) for ncut = 2:size(x, 1)]
    cutFreq = argmin(evalFilters)
    return flattenFrom(x, cutFreq + 1), evalFilters[cutFreq], cutFreq + 1
end



"""
if sortIndArray isn't defined, it comes from calling sortperm on the list of
scores, e.g.
    scores = [FourierObj(x) for x in eachslice(spacePop, dims=2)]
    sortIndArray = sortperm(scores)
"""
function dampenAboveBestHardbandpass(freqPop, sortIndArray, FourierObj, dampenAmount=0.1)
    flattened, val, freqI = BandpassStayInFourier(freqPop[:, sortIndArray[1]], FourierObj)
    newFreqPop = copy(freqPop)
    newFreqPop[freqI+1:end, :] .*= dampenAmount
    return newFreqPop
end

function perturbRand(setProbW, frac=1 / 5)
    pop = setProbW.runcontrollers[end].optimizer.population.individuals
    Ndims = size(pop, 1)
    popSize = size(pop, 2)
    theChosenPert = rand(1:size(pop, 2), round(Int, popSize * frac))
    # add pink noise with .1 of the norm to 100 randomly selected vectors
    perturbing = dct(pinkStart(Ndims, -1, nPert), 1)
    perturbing ./= [norm(x) for x in eachslice(perturbing, dims=2)]'
    pop[:, theChosenPert] += perturbing .* [norm(x) for x in eachslice(pop[:, theChosenPert], dims=2)]'
    adjustPopulation!(setProbW, pop)
end

function perturbWorst(setProbW, scores, frac=1 / 5, transform=x -> dct(x, 1), boundLevel=1e5, noiseColor=-1)
    pop = setProbW.runcontrollers[end].optimizer.population.individuals
    Ndims = size(pop, 1)
    popSize = size(pop, 2)
    nPert = round(Int, size(pop, 2) * frac)
    theChosenPert = sortperm(scores)[(end+1-nPert):end]
    # add pink noise
    perturbing = transform(pinkStart(Ndims, noiseColor, nPert))
    # with .1 of the norm to the worst frac vectors
    targValue = 0.3 * [norm(x) for x in eachslice(pop[:, theChosenPert], dims=2)]' ./ [norm(x) for x in eachslice(perturbing, dims=2)]'
    maxValue = minimum(boundLevel .- abs.(pop[:, theChosenPert]), dims=1) ./ maximum(abs.(perturbing), dims=1)

    perturbing .*= min.(targValue, maxValue)
    pop[:, theChosenPert] += perturbing
    adjustPopulation!(setProbW, pop)
end
