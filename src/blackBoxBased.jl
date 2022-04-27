
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


function perturbRand(setProbW, frac=1 / 5, transform=x -> dct(x, 1), boundLevel=1e5, noiseColor=-1)
    pop = setProbW.runcontrollers[end].optimizer.population.individuals
    Ndims = size(pop, 1)
    nPert = round(Int, size(pop, 2) * frac)
    theChosenPert = rand(1:size(pop, 2), nPert)
    # add pink noise with .1 of the norm to 100 randomly selected vectors
    perturbing = transform(pinkStart(Ndims, noiseColor, nPert))
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
