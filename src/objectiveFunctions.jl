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
function makeObjFun(target::ScatteredOut, path::pathLocs, st, normalize=st.normalize, λ=1e-5, innerProd=false, pretransform=idct)
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
            if innerProd
                function normObjx(x̂)
                    x = pretransform(x̂)
                    t = st(x)
                    1.1f0^()
                end
                return normObjx
            end
        end
    end
end
function makeCoordMaxObj(path, st; normFun=norm, λ=1e-5, pretransform=x̂ -> idct(x̂, 1))
    function normObjx(x̂)
        x = pretransform(x̂)
        t = st(x)
        1.1f0^(-t[path][1] + λ * normFun(x))
    end
    return normObjx
end
