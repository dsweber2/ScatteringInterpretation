function saveSerial(name, object...)
    fh = open(name, "w")
    serialize(fh, object...)
    close(fh)
end
