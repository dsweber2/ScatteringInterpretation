function genNoise(sz, K, noiseType)
    if noiseType == :white
        tmp = randn(Float32, sz..., K)
        return tmp / norm(tmp)
    elseif noiseType == :pink
        N = sz[1]
        pink = 1 ./ Float32.(1 .+ (0:N >> 1)) # note: not 1:N>>1+1
        phase = exp.(Float32(2π) * im) .* rand(Float32, N >> 1 + 1, sz[2:end]..., K)
        res = irfft(pink .* phase, N, 1)
        return res ./ norm(res)
    end
end

"""
    pinkStart(n, c=2, m=1)
generate `m` pink noise samples with exponent `c` and length `n`
"""
function pinkStart(n, c=2, m=1)
    if c > 0
        return irfft(randn(ComplexF64, n >> 1 + 1, m) .* 1 ./ (1:(n >> 1 + 1)).^c, n, 1)
    else
        # otherwise, choose c randomly for each location uniformly over the interval [.5,3]
        c = (.5 .+ 2.5 * rand(m))'
        return irfft(randn(ComplexF64, n >> 1 + 1, m) .* 1 ./ (1:(n >> 1 + 1)).^c, n, 1)
    end
end
