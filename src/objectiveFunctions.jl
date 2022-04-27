
"""
    obj = makeObjFun(target, path::pathLocs, st, normalize=st.normalize,
                     λ=1e-10, jointObj=false)

Make an objective function `obj(x)` that evaluates the difference between a
target stOutput and the input `x`.

    makeObjFun(target::AbstractArray, st, jointObj=false, kwargs...)
jointObj makes a function `obj(x,y)` where `x` is as otherwise and `y` has the
shape of target.

Note that the joint case is defined for y as an array. In that case, the path is
ignored and the difference across the whole output is evaluted.
"""
function makeObjFun(target::ScatteredOut, path::pathLocs, st,
    normalize=st.normalize, λ=1e-10, jointObj=false, innerProd=false)
    if jointObj # return a function of both x and y?
        if normalize # normalize the input
            if innerProd # just maximize the dot product?
                jointNormIPObjxy(x, y::ScatteredOut) = sum(mse(a[1], a[2]) for a in zip(st(x)[path], y[path])) + λ * norm(x)
                function jointNormIPObjxy(x, y::AbstractArray)
                    -sum(y' * ScatteringTransform.flatten(st(x))) + λ * norm(x)
                end
                return jointNormIPObjxy
            else
                jointNormObjxy(x, y::ScatteredOut) = sum(mse(a[1], a[2]) for a in zip(st(x)[path], y[path])) + λ * norm(x)
                function jointNormObjxy(x, y::AbstractArray)
                    sum((ScatteringTransform.flatten(st(x)) .- y) .^ 2) + λ * norm(x)
                end
                return jointNormObjxy
            end
        else
            jointObjxy(x, y::ScatteredOut) = sum(mse(a[1], a[2]) for a in
                                                 zip(st(x)[path], y[path]))
            function jointObjxy(x, y::AbstractArray)
                sum((ScatteringTransform.flatten(st(x)) .- y) .^ 2)
            end
            return jointObjxy
        end
    else
        if normalize
            return normObjx(x) = sum(mse(a[1], a[2]) for a in zip(st(x)[path], target[path])) + λ * norm(x)
        else
            return unnormObjx(x) = sum(mse(a[1], a[2]) for a in zip(st(x)[path], target[path]))
        end
    end
end

"""
    obj(x) = makeNormMatchObjFun(target, st, normTarget, λ)
fit the target in the scattering domain and try to keep the input domain norm near `normTarget`.
"""
function makeNormMatchObjFun(target, st, normTarget, λ)
    jointNormObj(x) = sum(mse(a[1], a[2]) for a in zip(st(x), target)) + λ * (normTarget - norm(x))^2
    function jointNormObjxy(x)
        sum((ScatteringTransform.flatten(st(x)) .- target) .^ 2) + λ * (normTarget - norm(x))^2
    end
end
"""
    makeObjFun(target, path::pathLocs, st, normalize=st.normalize)
if we want to fit the whole path for the target
"""
function makeObjFun(target::ScatteredOut, st, kwargs...)
    path = pathLocs(0, :, 1, :, 2, :)
    makeObjFun(target, path, st, kwargs...)
end


function makeObjFun(target::AbstractArray, st, kwargs...)
    makeObjFun(roll(target, st), st, kwargs...)
end


"""
Given a collection of vectors β used in multinomial regression, fit the positive
and negative parts separately
"""
function fitPositiveNegativeSeperately(β, inputSize; Niter=100, nExEach=1,
    initGen=randn, trainThese=1:size(β, 2),
    varargs...)
    βplus = max.(β, 0)
    βminus = max.(-β, 0)
    initPlus = initGen(inputSize, 1, nExEach, length(trainThese))
    initMinus = initGen(inputSize, 1, nExEach, length(trainThese))
    for i = trainThese
        plusObj = makeObjFun(βplus)
    end

end

distFromI(ii::CartesianIndex, jj, normType) = norm(ii.I .- jj, normType)
distFromI(ii, jj, normType) = norm(ii - jj, normType)
distFromI(ii, jj::Tuple{<:Integer}, normType) = norm(ii - jj[1], normType)
function computeSingleWeightMatrix(target, p::pathLocs, Pnorm, Nd=1)
    outDims = map(x -> x[1+Nd:end], size(target))
    weights = map(x -> fill(Inf, x...), outDims)
    n = ndims(target)
    for (layerI, jj) in enumerate(p.indices)
        if jj !== nothing
            layInds = eachindex(view(weights[layerI], axes(weights[layerI])...))
            weights[layerI][:] = map(ii -> distFromI(ii, jj[1+n:end-1], Pnorm), layInds)
        end
    end
    return weights
end
function computeWeightMatrices(target, pathCollection, distanceNorm)
    allWeights = map(p -> computeSingleWeightMatrix(target, p, distanceNorm), pathCollection)
    # find the minimal distance to each location
    netWeight = map(x -> zeros(size(x)), allWeights[1]) # set weights to zero initally
    for layI in 1:length(netWeight) # iterate over layers
        nDimsWeight = ndims(allWeights[1][layI]) # the ndims of this layer
        joined = cat([allWeights[pathInd][layI] for pathInd = 1:length(allWeights)]..., dims=nDimsWeight + 1)
        shortestDistance = minimum(joined, dims=nDimsWeight + 1) # closest pathLoc in this layer
        netWeight[layI][:] = shortestDistance
    end
    return netWeight
end
function weightByDistance(target, pathCollection, St, distanceNorm=1.0, elementNorm=2.0)
    netWeight = computeWeightMatrices(target, pathCollection, distanceNorm)
    to = target.output
    usedLayers = [i - 1 for i = 1:length(to) if netWeight[i][1] != Inf]
    function objFun(x)
        Stx = St(x)
        total = 0.0
        # minimize the irrelevant locations
        for ii in usedLayers
            w = netWeight[ii+1] # just a matrix
            total += sum(reshape(w, (1, size(w)...)) .* (Stx[ii] .- target[ii]) .^ elementNorm)
        end
        # maximize the relevant ones
        onPath = mapreduce(p -> sum((Stx[p] .- target[p]) .^ elementNorm), +, pathCollection)
        return total + onPath
    end
end


function allBut(l, w, x)
    ii = LinearIndices(x)[l, w, 1]
    x[[1:(ii-1)... (ii+1):end...]]
end
