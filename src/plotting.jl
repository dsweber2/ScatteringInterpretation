
function plotOverTitle(setOfPlots, title, gridSizes)
    overTitle = Plots.scatter([0, 0], [0, 1], xlims=(1, 1.1), xshowaxis=false, yshowaxis=false, label="", colorbar=false, grid=false, framestyle=:none, title=title, bottom_margin=-20Plots.px,)
    lay = Plots.@layout [a{0.00001h}; grid(gridSizes...)]
    plot(overTitle, setOfPlots..., layout=lay)
end

function plotFirstXEx(sorted, scores, title="Over title", apFun=identity, gridSizes=(4, 5); varargs...)
    plotOverTitle([plot(apFun(x), legend=false, title="$(round(log(scores[ix]), sigdigits=5))", titlefontsize=8, xticks=false, c=:black, varargs...) for (ix, x) in enumerate(eachcol(sorted[:, 1:prod(gridSizes)]))], title, gridSizes)
end
