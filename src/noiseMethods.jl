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

pinkStart(n, c=2) = irfft(randn(ComplexF64, n >> 1 + 1) .* 1 ./ (1:(n >> 1 + 1)).^c, n)
