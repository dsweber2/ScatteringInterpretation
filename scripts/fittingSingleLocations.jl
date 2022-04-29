using Distributed
addprocs(12)
@everywhere using ScatteringTransform, ScatteringInterpretation, ContinuousWavelets

N = 128
popSize = 500
totalMins = 1 * 60
nMinsBBO = 3
perturbRate = 1 / 5
β = [2, 2, 1]
averagingLength = [-1, -1, 2]
outputPool = 8
normalize = false
poolBy = 3 // 2
pNorm = 2
extraOctaves = 0
waveletExamples = (dog2, dog1, Morlet(π), Morlet(2π))
shortNames = ["dog2", "dog1", "morl", "morl2Pi"] # set the names for saving


println("----------------------------------------------------------------------------------------------")
println("----------------------------------------------------------------------------------------------")
println("----------------------------------------------------------------------------")
println("----------------------------------------------------------------------------")
println("----------------------------------------------------------------------------")
println("Generating the data for the first layer")
println("----------------------------------------------------------------------------")
println("----------------------------------------------------------------------------")
println("----------------------------------------------------------------------------")
println("----------------------------------------------------------------------------------------------")
println("----------------------------------------------------------------------------------------------")
for (ii, CW) in enumerate(waveletExamples)
    println("----------------------------------------------------------------------------------------------")
    println("-------------------------------starting $(CW)-------------------------------")
    println("----------------------------------------------------------------------------------------------")
    St = scatteringTransform((N, 1, 1), cw=CW, 2, poolBy=poolBy, β=β, averagingLength=averagingLength, outputPool=outputPool, normalize=normalize, σ=abs, p=pNorm, extraOctaves=extraOctaves)
    J1, J2 = map(x -> length(x), getWavelets(St))
    @sync @distributed for w1 in 1:J1-1
        fittingSinglePath(N, (w1,), totalMins=totalMins, cw=CW, namedCW=shortNames[ii])
        fittingSinglePath(N, (w1,), reprFun=(identity, identity), extraName="SPACE", cw=CW, namedCW=shortNames[ii], totalMins=totalMins)
    end
end




println("----------------------------------------------------------------------------------------------")
println("----------------------------------------------------------------------------------------------")
println("----------------------------------------------------------------------------")
println("----------------------------------------------------------------------------")
println("----------------------------------------------------------------------------")
println("Generating the data for the second layer")
println("----------------------------------------------------------------------------")
println("----------------------------------------------------------------------------")
println("----------------------------------------------------------------------------")
println("----------------------------------------------------------------------------------------------")
println("----------------------------------------------------------------------------------------------")


extraName = ""
for (ii, CW) = enumerate(waveletExamples)
    namedCW = shortNames[ii]
    St = scatteringTransform((N, 1, 1), cw=CW, 2, poolBy=poolBy, β=β, averagingLength=averagingLength, outputPool=outputPool, normalize=normalize, σ=abs, p=pNorm, extraOctaves=extraOctaves)
    J1, J2 = map(x -> length(x[1]), getWavelets(St))
    listOfLocations = ([(j2, j1) for j1 = 1:(J1-1), j2 = 1:(J2-1)]...,)
    println("doing $CW, there are $(length(listOfLocations)) examples to do")
    # second layer setup begin
    # (w2,w1) = listOfLocations[15]
    @sync @distributed for (w2, w1) in listOfLocations
        println("doing ($w2,$w1)")
        fittingSinglePath(N, (w2, w1), totalMins=totalMins, cw=CW, namedCW=shortNames[ii])
        fittingSinglePath(N, (w2, w1,), reprFun=(identity, identity), extraName=extraName * "SPACE", cw=CW, namedCW=shortNames[ii], totalMins=totalMins)
    end
end

println("----------------------------------------------------------------------------------------------")
println("----------------------------------------------------------------------------------------------")
println("----------------------------------------------------------------------------")
println("----------------------------------------------------------------------------")
println("----------------------------------------------------------------------------")
println("Generating the data for the zeroth layer")
println("----------------------------------------------------------------------------")
println("----------------------------------------------------------------------------")
println("----------------------------------------------------------------------------")
println("----------------------------------------------------------------------------------------------")
println("----------------------------------------------------------------------------------------------")
for (ii, CW) = enumerate(waveletExamples)
    fittingSinglePath(N, tuple(), totalMins=totalMins, cw=CW, namedCW=shortNames[ii])
    fittingSinglePath(N, tuple(), reprFun=(identity, identity), extraName="SPACE", cw=CW, namedCW=shortNames[ii], totalMins=totalMins, randFun=(N, popSize) -> pinkNoise(N, popSize, domain=:rfft))
end
