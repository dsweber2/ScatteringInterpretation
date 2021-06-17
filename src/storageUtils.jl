function buildRecord(stack::stFlux, nEpochs)
    p = params(stack).order
    return [zeros(Float32, size(x)..., nEpochs + 1) for x in p]
end

function buildRecord(stack, nEpochs)
    zeros(eltype(stack), size(stack)..., nEpochs)
end

function expandRecord!(record::Array{Array}, nEpochs)
    for ii in 1:length(record)
        adding = zeros(Float32, size(record[ii])[1:end - 1]..., nEpochs)
        record[ii] = cat(record[ii], adding, dims=ndims(record[ii]))
        size(record[ii])
    end
end

function expandRecord!(record::Array{<:Number}, nEpochs)
    adding = zeros(eltype(record), size(record)[1:end - 1]..., nEpochs)
    record[:] = cat(record, adding, dims=ndims(record))
end

function addCurrent!(record, stack::stFlux, ii)
    p = params(stack).order

    for (kk, x) in enumerate(p)
        ax = axes(x)
        record[kk][ax..., ii + 1] = x
    end
end

function addCurrent!(record, result, ii)
    p = params(stack).order
    result = 3
    record[]
    for (kk, x) in enumerate(p)
        ax = axes(x)
        record[kk][ax..., ii + 1] = x
    end
end




function saveSerial(name, object...)
    fh = open(name, "w")
    serialize(fh, object...)
    close(fh)
end
