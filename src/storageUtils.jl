function saveSerial(name, object...)
    fh = open(name, "w")
    serialize(fh, object...)
    close(fh)
end

function loadData(N, pathTuple; loc=-1, namedCW="default", saveDir="", saveFigDir="", CW=dog2, β=[2, 2, 1], averagingLength=[-1, -1, 2], outputPool=8, normalize=false, poolBy=3 // 2, pNorm=2, extraOctaves=0, extraName="")
    layer = length(pathTuple)
    St = scatteringTransform((N, 1, 1), cw=CW, 2, poolBy=poolBy, β=β, averagingLength=averagingLength, outputPool=outputPool, normalize=normalize, σ=abs, p=pNorm, extraOctaves=extraOctaves)
    Sx = St(randn(N))
    aveLenStr = mapreduce(x -> "$x", *, averagingLength)
    includeLoc = (loc <= 0)
    if includeLoc
        loc = ceil(Int, 1 / 2 * size(Sx[layer], 1))
    end
    if saveDir == ""
        saveDir = joinpath("..", "results", "singleFits", "$(namedCW)", "poolBy$(round(poolBy, sigdigits=3))_outPool$(outputPool)_pNorm$(pNorm)_aveLen$(aveLenStr)_extraOct$(extraOctaves)", "lay$(layer)")
    end
    saveDir = joinpath(saveDir, string(["l$(layer-ii+1)=" * string(x) * "_" for (ii, x) in enumerate(pathTuple)]...))
    if includeLoc
        saveDir = joinpath(saveDir, "location$(loc)")
        saveFigDir = joinpath(saveFigDir, "location$(loc)")
    end

    saveName = joinpath(saveDir, "data$(extraName).jld2")
    saveNameDict = joinpath(saveDir, "dictionary")
    saveSerialName = joinpath(saveDir, "data")
    tmpDictionary = FileIO.load(saveName)
    scores = tmpDictionary["scores"]
    pop = tmpDictionary["pop"]
    iScores = sortperm(scores)
    bestEx = pop[:, iScores[1]]
    sortedPop = pop[:, iScores]

    return bestEx, sortedPop, scores
end

function loadData(N, layer::Integer; loc=-1, namedCW="default", saveDir="", saveFigDir="", CW=dog2, β=[2, 2, 1], averagingLength=[-1, -1, 2], outputPool=8, normalize=false, poolBy=3 // 2, pNorm=2, extraOctaves=0, extraName="", popSize=500)
    St = scatteringTransform((N, 1, 1), cw=CW, 2, poolBy=poolBy, β=β, averagingLength=averagingLength, outputPool=outputPool, normalize=normalize, σ=abs, p=pNorm, extraOctaves=extraOctaves)
    Sx = St(randn(N))
    includeLoc = (loc <= 0)
    if includeLoc
        loc = ceil(Int, 1 / 2 * size(Sx[layer], 1))
    end
    freqs = getMeanFreq(St)
    waves = getWavelets(St, spaceDomain=true)
    exactOutputLocations = [(range(1, stop=size(waves[l+1], 1) / 2, length=size(Sx[layer], 1))...,) for l = 0:2]
    wavShift = round.(Int, [rang[loc] for rang in exactOutputLocations])
    waves = [circshift(wave, (wavShift[ii]))[1:Int(size(wave, 1) / 2), :] for (ii, wave) in enumerate(waves)]
    aveLenStr = mapreduce(x -> "$x", *, averagingLength)
    if saveFigDir == ""
        saveFigDir = joinpath("..", "figures", "singleFits", "$(namedCW)", "poolBy$(round(poolBy, sigdigits=3))_outPool$(outputPool)_pNorm$(pNorm)_aveLen$(aveLenStr)_extraOct$(extraOctaves)", "lay$(layer)")
    end
    if !isdir(saveFigDir)
        mkpath(saveFigDir)
    end
    if layer == 0
        println("return 0")
        bestEx, sortedPop, scores = loadData(N, tuple(); loc=includeLoc * loc, namedCW=namedCW, saveDir=saveDir, saveFigDir=saveFigDir, CW=CW, β=β, averagingLength=averagingLength, outputPool=outputPool, normalize=normalize, poolBy=poolBy, pNorm=pNorm, extraOctaves=extraOctaves, extraName=extraName)
        return bestEx, waves, sortedPop, scores, wavShift, freqs, saveFigDir
    end
    nFreqs = reverse([1:x for x in length.(freqs) .- 1]) # get the indexes for the frequencies in reverse order, so e.g. (second layer index, first layer index)

    examples = zeros(N, size(product(nFreqs[2:(layer+1)]...))...)
    sortedPops = zeros(N, popSize, size(product(nFreqs[2:(layer+1)]...))...)
    allScores = zeros(popSize, size(product(nFreqs[2:(layer+1)]...))...)
    for pathTuple in product(nFreqs[2:(layer+1)]...)
        bestEx, sortedPop, scores = loadData(N, pathTuple; loc=loc, namedCW=namedCW, saveDir=saveDir, saveFigDir=saveFigDir, CW=CW, β=β, averagingLength=averagingLength, outputPool=outputPool, normalize=normalize, poolBy=poolBy, pNorm=pNorm, extraOctaves=extraOctaves, extraName=extraName)
        examples[:, pathTuple...] = bestEx
        sortedPops[:, :, pathTuple...] = sortedPop
        allScores[:, pathTuple...] = scores
    end
    return examples, waves, sortedPops, allScores, wavShift, freqs, saveFigDir
end
