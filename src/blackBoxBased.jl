
"""
    adjustPopulation!(setProbW, popMat)
given a BlackBoxOptim `OptController` with some kind of population based method, set the population to be the matrix given in `popMat`; each column of `popMat` is a different individual.
"""
function adjustPopulation!(setProbW, popMat)
    for i = 1:size(popMat, 2)
        setProbW.optimizer.population[i] = copy(popMat[:,i])
    end
end

getPopulation(setProbW) = setProbW.runcontrollers[end].optimizer.population.individuals


"""
    bandpassed,objValue, freqInd = BandpassStayInFourier(x,FourierObj)
given a dct representation of a signal `x` and an objective function `FourierObj` which operates on the dct representation of a signal, find the frequency index a super naive way of finding a bandpass threshold so that x (which is represented as dct of the target)  FourierObj
"""
function BandpassStayInFourier(x, FourierObj)
    flattenFrom(x, i) = cat(x[1:i,axes(x)[2:end]...], zeros(size(x, 1) - i), dims=1)
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
function dampenAboveBestHardbandpass(freqPop, sortIndArray, FourierObj, dampenAmount=.1)
    flattened, val, freqI = BandpassStayInFourier(freqPop[:,sortIndArray[1]], FourierObj)
    newFreqPop = copy(freqPop)
    newFreqPop[freqI + 1:end,:] .*= dampenAmount
    return newFreqPop
end

function perturbWorst(setProbW)
    pop = setProbW.runcontrollers[end].optimizer.population.individuals
    theChosenPert = rand(1:size(pop, 2), 100)
            # add pink noise with .1 of the norm to 100 randomly selected vectors
    perturbing = dct(cat([pinkStart(128, c) for c = range(1, 3, length=100)]..., dims=2), 1)
    perturbing ./= [norm(x) for x in eachslice(perturbing, dims=2)]'
    pop[:, theChosenPert] += perturbing .* [norm(x) for x in eachslice(pop[:,theChosenPert], dims=2)]'
    adjustPopulation!(setProbW, pop)
end
