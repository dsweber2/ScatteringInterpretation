"""
    pinkStart(n, c=2, m=1)
generate `m` pink noise samples with exponent `c` and length `n`. Does not scale frequencies
"""
function pinkNoise(n, m=1, c=-1; domain=:dct, rng=MersenneTwister())
    if c <= 0
        # if c isn't a reasonable value, choose c randomly for each location uniformly over the interval [.5,3]
        c = (0.5 .+ 2.5 * rand(rng, m))'
    end

    if domain == :rfft
        fourierDomain = randn(rng, ComplexF64, n >> 1 + 1, m) .* 1 ./ (1:(n>>1+1)) .^ c
        return irfft(fourierDomain, n, 1)
    elseif domain == :dct
        fourierDomain = randn(rng, n, m) .* 1 ./ (1:(n)) .^ c
        return idct(fourierDomain, 1)
    end
end
