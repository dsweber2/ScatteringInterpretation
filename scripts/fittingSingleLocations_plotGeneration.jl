# this script strictly generates the relevant figures; the data is created in section2Data.jl
using Revise
using ScatteringTransform, ContinuousWavelets, Plots, JLD, JLD2, FileIO, FFTW, LinearAlgebra, FourierFilterFlux
using BlackBoxOptim
using ScatteringInterpretation
gr(dpi=200)

N = 128
popSize = 500
waveletsFit = (dog2, dog1, Morlet(π), Morlet(2π))
longNames = ("DoG2", "DoG1", "Morlet π", "Morlet 2π")
shortNames = ["dog2", "dog1", "morl", "morl2Pi"] # set the names for saving

β = [2, 2, 1]
averagingLength = [-1, -1, 2]
outputPool = 8
normalize = false
poolBy = 3 // 2
NDescents = 100
descentLength = 1000
pNorm = 2
NTrain = 5000
extraOctaves = 0

# zeroth layer joint plots, which in the final draft is just figure 4.13
solutions = zeros(N, length(waveletsFit))
wavs = zeros(N, length(waveletsFit))
wavShifts = zeros(3, length(waveletsFit))
for (ii, CW) in enumerate(waveletsFit)
    namedCW = shortNames[ii]
    bestEx, waves, sortedPop, scores, wavShift, freqs = loadData(N, 0, CW=CW, namedCW=namedCW, β=β, averagingLength=averagingLength, extraName=extraName, poolBy=poolBy, extraOctaves=extraOctaves, pNorm=pNorm, outputPool=outputPool)
    solutions[:, ii] = bestEx
    wavs[:, ii] = real.(waves[1][:, end])
    wavShifts[:, ii] = wavShift
end
renormedWavs = cat([real.(wavs)[:, jj] * norm(solutions[:, jj]) / norm(wavs[:, jj]) for jj = 1:size(solutions, 2)]..., dims=2)
peakDiff = argmax(renormedWavs, dims=1) - argmax(solutions, dims=1)
println("the distance from the peak is $(peakDiff)")
tt = [(0:N-1) .- wavShift[1, 1] for x in eachslice(wavs, dims=2)]
ytickSizes = [(range(min(minimum(solutions[:, jj]), minimum(real.(renormedWavs)[:, jj])), max(maximum(solutions[:, jj]), maximum(real.(renormedWavs)[:, jj])), length=5), "") for jj = 1:size(solutions, 2)]
sharedArgs = (gridalpha=0.3, frame=:box, linewidth=1.5, guidefontrotation=-0, left_margin=-10Plots.px, guidefontvalign=:center, yguidefontsize=7, guidefonthalign=:center, tickfontsize=5,)
# first plot needs more vertical space
plt1 = plot(tt[end-1], [renormedWavs[:, end] solutions[:, end]]; xticks=(range(tt[end-1][1], tt[end-1][end], length=7), ""), bottom_margin=-12Plots.px, top_margin=0Plots.px, yticks=ytickSizes[end], ylabel="$(longNames[end])", xlims=(tt[end-1][1], tt[end-1][end]), legend=false, sharedArgs...)
# last plot needs horizontal ticks
pltLast = plot(tt[1], [renormedWavs[:, 1] solutions[:, 1]]; xticks=(range(tt[1][1], tt[1][end], length=7), round.(Int, range(1, 128, length=7))), top_margin=-12Plots.px, yticks=ytickSizes[1], ylabel="$(longNames[1])", xlabel="Time (ms)", xlims=(tt[1][1], tt[1][end]), legend=false, sharedArgs...)
# every other plot
plts = [plot(tt[jj], [renormedWavs[:, jj] solutions[:, jj]]; xticks=(range(tt[end][1], tt[end][end], length=7), ""), legend=false, bottom_margin=-12Plots.px, top_margin=-12Plots.px, frame=:box, yticks=ytickSizes[jj], ylabel="$(longNames[jj])", xlims=(tt[jj][1], tt[jj][end]), sharedArgs...) for jj = (size(solutions, 2)-1):-1:2]
title = "Fitting Zeroth Layer"
yLabel = "Frequency (Hz)"
gridSizes = (size(solutions, 2), 1)
setOfPlots = [plt1, plts..., pltLast]
overTitle = plot(randn(1, 2), xlims=(1, 1.1), xshowaxis=false, yshowaxis=false, label="", colorbar=false, grid=false, framestyle=:none, title=title, top_margin=-25Plots.px, bottom_margin=-13Plots.px, legend=:outertopright, labels=["original wavelet" "coordinate maximizer"], legendfontsize=4,)
fauxYLabel = Plots.scatter([0, 0], [0, 1], xlims=(1, 1.1), xshowaxis=false, label="", colorbar=false, grid=false, ylabel="Wavelet type", xticks=false, yticks=false, framestyle=:grid, left_margin=12Plots.px, right_margin=-12Plots.px)
lay = @layout [a{0.01h}; [b{0.00001w} grid(gridSizes...)]]
plot(overTitle, fauxYLabel, setOfPlots..., layout=lay)
mkpath(joinpath("..", "figures", "interpret", "singleFits"))
savefig(joinpath("..", "figures", "interpret", "singleFits", "zerothMultiBestPlot.pdf"))
savefig(joinpath("..", "figures", "interpret", "singleFits", "zerothMultiBestPlot.png"))





println("first layer joint comparative plots")
for (ii, CW) in enumerate(waveletsFit)
    println(ii)
    namedCW = shortNames[ii]
    solutions, waves, sortedPops, allScores, wavShift, freqs, saveFigDir = loadData(N, 1, CW=CW, namedCW=namedCW, β=β, averagingLength=averagingLength, extraName=extraName, poolBy=poolBy, extraOctaves=extraOctaves, pNorm=pNorm, outputPool=outputPool)
    firstLayerFrequencies = map(f -> round(f, sigdigits=3), freqs[1])[1:end-1]
    secondLayerFrequencies = map(f -> round(f, sigdigits=3), freqs[2])[1:end-1]
    firstWaves = waves[1]
    plot(firstWaves)
    # check whether or not solutions needs to be flipped to match the sign of the original wavelets
    signs = [(sign(solutions[argmax(abs.(firstWaves[:, jj])), jj]) * sign(real.(firstWaves)[argmax(abs.(firstWaves[:, jj])), jj])) for jj = 1:size(solutions, 2)]
    signedSols = signs' .* real.(solutions)
    renormedWavs = cat([real.(firstWaves)[:, jj] * norm(solutions[:, jj]) / norm(firstWaves[:, jj]) for jj = 1:size(solutions, 2)]..., dims=2)
    ytickSizes = [(range(min(minimum(signedSols[:, jj]), minimum(real.(renormedWavs)[:, jj])), max(maximum(signedSols[:, jj]), maximum(real.(renormedWavs)[:, jj])), length=5), "") for jj = 1:size(solutions, 2)]
    kurtFirstLayFreq = round.(Int, firstLayerFrequencies)

    sharedArgs = (gridalpha=0.3, frame=:box, linewidth=1.5, guidefontrotation=-0, left_margin=-10Plots.px, guidefontvalign=:center, yguidefontsize=5, guidefonthalign=:center, tickfontsize=5,)
    tt = [(0:N-1) .- wavShift[1] for x in eachslice(firstWaves, dims=2)]
    # first plot needs more vertical space
    plt1 = plot(tt[end-1], [renormedWavs[:, end] signedSols[:, end]]; xticks=(range(tt[end-1][1], tt[end-1][end], length=7), ""), bottom_margin=-12Plots.px, top_margin=0Plots.px, yticks=ytickSizes[end], ylabel="$(kurtFirstLayFreq[end])", xlims=(tt[end-1][1], tt[end-1][end]), legend=false, sharedArgs...)
    # last plot needs horizontal ticks
    pltLast = plot(tt[1], [renormedWavs[:, 1] signedSols[:, 1]]; xticks=(range(tt[1][1], tt[1][end], length=7), round.(Int, range(1, 128, length=7))), top_margin=-12Plots.px, yticks=ytickSizes[1], ylabel="$(kurtFirstLayFreq[1])", xlabel="Time (ms)", xlims=(tt[1][1], tt[1][end]), legend=false, sharedArgs...)
    # every other plot
    plts = [plot(tt[jj], [renormedWavs[:, jj] signedSols[:, jj]]; xticks=(range(tt[end][1], tt[end][end], length=7), ""), legend=false, bottom_margin=-12Plots.px, top_margin=-12Plots.px, frame=:box, yticks=ytickSizes[jj], ylabel="$(kurtFirstLayFreq[jj])", xlims=(tt[jj][1], tt[jj][end]), sharedArgs...) for jj = (size(solutions, 2)-1):-1:2]
    l = @layout grid(1, 16)
    title = "Fitting First Layer, $(longNames[ii])"
    yLabel = "Frequency (Hz)"
    gridSizes = (size(solutions, 2), 1)
    setOfPlots = [plt1, plts..., pltLast]
    overTitle = plot(randn(1, 2), xlims=(1, 1.1), xshowaxis=false, yshowaxis=false, label="", colorbar=false, grid=false, framestyle=:none, title=title, top_margin=-25Plots.px, bottom_margin=-13Plots.px, legend=:outertopright, labels=["original wavelet" "coordinate maximizer"], legendfontsize=4,)
    fauxYLabel = Plots.scatter([0, 0], [0, 1], xlims=(1, 1.1), xshowaxis=false, label="", colorbar=false, grid=false, ylabel=yLabel, xticks=false, yticks=false, framestyle=:grid, left_margin=12Plots.px, right_margin=-12Plots.px)
    lay = @layout [a{0.01h}; [b{0.00001w} grid(gridSizes...)]]
    plot(overTitle, fauxYLabel, setOfPlots..., layout=lay)
    savefig(joinpath(saveFigDir, "multiBestPlot.pdf"))


    numPlts = size(signedSols, 2)
    height = round(Int, sqrt(numPlts))
    width = ceil(Int, numPlts / height)
    tmpPlt = heatmap(1:128, f1[1:end-1], abs.(circshift(St.mainChain[1](signedSols[:, 1])[:, 1:(end-1), 1, 1]', (0, 0))) .^ 2, c=cgrad(:viridis, scale=:exp), framestyle=:box, title="$(firstLayerFrequencies[1]) Hz", titlefontsize=7, guidefontsize=5, tickfontsize=4, colorbar=false, link=:all)
    listOfPlots = Array{typeof(tmpPlt),1}(undef, size(solutions, 2))
    for w1 = 1:size(solutions, 2)
        if w1 % width == 1
            if w1 >= (height - 1) * width + 1
                # on the left edge and the bottom
                extraArgs = (ylabel="Frequency (Hz)", xlabel="time (ms)")
            else
                # on the left edge but not the bottom
                extraArgs = (ylabel="Frequency (Hz)", xticks=false)
            end
        else
            if w1 >= (height - 1) * width + 1
                # on the bottom and not the left edge
                extraArgs = (xlabel="time (ms)", yticks=false)
            else
                # in the middle
                extraArgs = (ticks=false,)
            end
        end
        listOfPlots[w1] = heatmap(1:128, f1[1:end-1], abs.(circshift(St.mainChain[1](signedSols[:, w1])[:, 1:(end-1), 1, 1]', (0, 0))) .^ 2; c=cgrad(:viridis, scale=:exp), framestyle=:box, title="$(firstLayerFrequencies[w1]) Hz", titlefontsize=5, guidefontsize=5, tickfontsize=4, colorbar=false, link=:all, bottom_margin=-4Plots.px, extraArgs...)
    end
    plt = plot(listOfPlots..., link=:all)
    title = "Scalograms of First Layer Fits, $(longNames[ii])"
    overTitle = Plots.scatter([0, 0], [0, 1], xlims=(1, 1.1), xshowaxis=false, yshowaxis=false, label="", colorbar=false, grid=false, framestyle=:none, title=title, bottom_margin=-30Plots.px)
    l = @layout [a{0.0001h}; b]
    plot(overTitle, plt, layout=l)
    savefig(joinpath(saveFigDir, "allScalograms.pdf"))
    savefig(joinpath(saveFigDir, "allScalograms.png"))
end


println("Second layer joint comparative plots. This is figures 4.18-22")
# second layer
for (ii, CW) in enumerate(waveletsFit)
    println("Doing $CW")
    longName = longNames[ii]
    namedCW = shortNames[ii]
    solutions, waves, sortedPops, allScores, wavShift, freqs, saveFigDir = loadData(N, 2; CW=CW, namedCW=namedCW, β=β, averagingLength=averagingLength, extraName=extraName, poolBy=poolBy, extraOctaves=extraOctaves, pNorm=pNorm, outputPool=outputPool)
    firstLayerFrequencies = map(f -> round(f, sigdigits=3), freqs[1])[1:end-1]
    secondLayerFrequencies = map(f -> round(f, sigdigits=3), freqs[2])[1:end-1]
    nFreqs = reverse([1:x for x in length.(freqs) .- 1]) # get the indexes for the frequencies in reverse order, so e.g. (second layer index, first layer index)
    size(solutions)
    targets = Dict(map(x -> (x => solutions[:, x...]), product(nFreqs[2:3]...)))
    f1, f2, f3 = freqs
    lenFreq2 = length(secondLayerFrequencies)
    lenFreq1 = length(firstLayerFrequencies)
    exWidth = 1 / lenFreq2
    exHeight = 1 / lenFreq1
    pltt = scatter(0, 0, tickdirection=:out, rotation=30, xlims=(0.5, lenFreq2 + 0.5), xticks=(1:lenFreq2, secondLayerFrequencies), xlabel="Second layer frequency (Hz)", ylims=(0.5, lenFreq1 + 0.5), yticks=(1:lenFreq1, firstLayerFrequencies), ylabel="First layer frequency (Hz)", legend=false, tickfontsize=4, guidefontsize=7)
    plot!(title="Best Fit Solutions for $(longNames[ii]) Second Layer", titlefontsize=12)
    for (plotNumber, key) in enumerate(keys(targets))
        i2, i1 = key
        pltt = plot!(targets[key], inset=(1, bbox((i2 - 1) * exWidth, (i1 - 1) * exHeight, exWidth, exHeight, :bottom, :left)), legend=false, framestyle=:box, subplot=plotNumber + 1, ticks=false)
    end
    pltt
    savefig(joinpath(saveFigDir, "multiBestPlot.pdf"))
    savefig(joinpath(saveFigDir, "multiBestPlot.png"))

    w2 = 1
    w1 = lenFreq1 - 1
    plot(range(1, stop=N, length=size(waves[2], 1)), norm(targets[(w2, w1)], Inf) / norm(waves[2][:, w2], Inf) * waves[2][:, w2], label="Second Layer Wavelet")
    plot!(targets[(w2, w1)], label="Result")
    title!("$((secondLayerFrequencies[w2], firstLayerFrequencies[w1]))")


    pltt = scatter(0, 0, tickdirection=:out, rotation=30, xlims=(0.5, lenFreq2 + 0.5), xticks=(1:lenFreq2, secondLayerFrequencies), xlabel="Second layer frequency (Hz)", ylims=(0.5, lenFreq1 + 0.5), yticks=(1:lenFreq1, firstLayerFrequencies), ylabel="First layer frequency (Hz)", legend=false,)
    plot!(title="Scalograms of Best Fit Solutions for $(longNames[ii]) Second Layer", titlefontsize=12)
    for (plotNumber, key) in enumerate(keys(targets))
        i2, i1 = key
        pltt = heatmap!(abs.(circshift(St.mainChain[1](targets[key])[:, 1:(end-1), 1, 1]', (0, 0))) .^ 2, c=cgrad(:viridis, scale=:exp), colorbar=false, inset=(1, bbox((i2 - 1) * exWidth + 0.001, (i1 - 1) * exHeight + 0.001, exWidth, exHeight, :bottom, :left)), legend=false, subplot=plotNumber + 1, ticks=false, framestyle=:none)
    end
    pltt
    savefig(joinpath(saveFigDir, "multiBestScalogramPlot.pdf"))
    savefig(joinpath(saveFigDir, "multiBestScalogramPlot.png"))
    # run(`convert -density 100 $(saveFigDir)multiBestScalogramPlot.pdf -quality 70 $(saveFigDir)multiBestScalogramPlot.png`)
end
