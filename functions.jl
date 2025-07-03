## data reading functions. --------------------------------------------------------------------------

# training data
function read_training(p)

    function flocal(v, sd)::Vector{Bool}
        out = zeros(Bool, length(sd))
        for x in v
            out[sd[x]] = true
        end
        return out
    end

    # read input file
    data_raw = split.(readlines(p), '\t')

    # extract header
    hdr = popfirst!(data_raw)
    emtype = parse.(Float64, hdr[2:end])

    # strains
    sraw = split.(first.(data_raw), '|')
    strains = reduce(union, sraw) |> sort
    sd = Dict(k => i for (i, k) in enumerate(strains))
    strain_mat = reduce(hcat, [flocal(x, sd) for x in sraw])' |> collect

    # data
    data = reduce(hcat, [parse.(Float64, x[2:3]) for x in data_raw])' |> collect

    # return
    return strains, strain_mat, data, emtype

end;

# verification data
function read_verification(p2)

    function process_block(x)
        xs = split(x, '\n')
        return split.(xs, '\t')
    end

    raw = read(p2) |> String
    rs1 = split(raw, "\n//\n")

    t0 = process_block(rs1[2])
    t0d = Dict{String,Dict{String,Vector{Float64}}}()

    for x in t0
        vs = parse.(Float64, x[3:4])
        if haskey(t0d, x[1])
            t0d[x[1]][x[2]] = vs
        else
            t0d[x[1]] = Dict(x[2] => vs)
        end
    end
    tend = process_block(rs1[3])[1:end-1]

    return t0d, tend

end;

## fitting functions. -------------------------------------------------------------------------------

hill(y_raw, km, hill_exp) = @. (y_raw^hill_exp) / (km^hill_exp + y_raw^hill_exp)

function monofit(y, p)

    # y is a vector that contains the column indexed strain mat followed by the hill function
    # parameters from fitting hill1. p is the nxk matrix of C,R prefences of each strain to be fit

    # hill response function 

    # extract data
    x = reshape(max.(Ref(0), p[1:end-2]), :, 2)
    A = reshape(max.(Ref(0),y), :, size(x, 1),) |> Matrix{Int}
    hill1p = p[end-1:end]

    # calculate input community degradation - A*x
    Ax = A * x

    # get final degradation of each monomer type after transformation with hill function
    c = hill(Ax[:, 1], hill1p...)
    r = hill(Ax[:, 2], hill1p...)

    return [c; r]

end;

function consume_leftover(x)
    curr = maximum(x)
    if length(x)>1
        for v in sort(x, rev=true)[2:end]
            curr = 1-(1-curr)*(1-v)
        end
    end
    return curr
end


function mm(strain_mat, mono)
    o = zeros(Float64, size(strain_mat, 1), 2)
    cnt = 1
    for i in eachrow(strain_mat)
        o[cnt, 1] = consume_leftover(mono[i,1])
        o[cnt, 2] = consume_leftover(mono[i,2])
        cnt += 1
    end
    return o
end

function multifit(y, p)
    # y is a vector that contains the column indexed strain mat  
    # p is the nxk matrix of C,R prefences of each strain to be fit
    x = reshape(p, :, 2)
    A = reshape(y, :, size(x, 1),) |> Matrix{Bool}
    return mm(A, x)[:]
end

function monofit_total(y, p)
    # y is a vector that contains the column indexed strain mat followed by the hill function parameters from fitting hill1 
    # p is the nx1 matrix of total degradation ability of each strain to be fit
    
    x = p[1:end-2]
    hill1p = p[end-1:end]

    A = reshape(y, :, length(x),) |> Matrix{Int}
    Ax = A * x

    return hill(max.(0, Ax), hill1p...)

end

function predict(strain_mat, monoculture_params, km, hill_exp) 
    # ŷ is the vector/matrix of observed values
    y_raw = strain_mat * monoculture_params
    y = hill(y_raw[:], km, hill_exp)
    return y
end

sum_square(y, ŷ) = sum((y .- ŷ).^2)

function fvu(strain_mat, p_mono, p_hill, ŷ)::Float64
    y = predict(strain_mat, p_mono, p_hill...)
    ȳ = mean(y)
    sse = sum_square(y, ŷ[:])
    ss_tot = sum_square(y[:], ȳ)
    return sse / ss_tot
end
fvu(y::Vector{Float64}, ŷ::Vector{Float64}) = sum_square(y, ŷ) / sum_square(y, mean(y))

rsquare(strain_mat, p_mono, p_hill, obs) = 1.0 - fvu(strain_mat, p_mono, p_hill, obs)
rsquare(y::Vector{Float64}, ŷ::Vector{Float64}) = 1 - fvu(y,ŷ)



function main(strain_mat, y, data, nstrains)

    # init model parameters
    par0 = [ones(length(mono[:])); 1.0; 1.0]
    par0t = [m1sm; 1; 5]
    lb = zeros(length(par0)) # lower parameter limits
    hb = ones(length(par0))  # upper parameter limits
    hb[(end - 1):end] .= Inf

    # find relevant co-cultures
    mfi = sum(strain_mat, dims = 2)[:] .<= nstrains

    # fit degradation to non-linear additive model:
    # for each monomer type j -
    #   deg_tot(j) = sum(strain_monomer_decomposition)
    #   deg(j) = hill(deg_tot(j), km, hill_exponent)
    # where
    #   hill(x, km, hill_exponent) = x^hill_exponent / (km^hill_exponent + x^hill-exponent)
    # This function is fit using least squares
    sol2 = curve_fit(
        monofit,
        strain_mat[mfi, :][:],
        y[mfi, :][:],
        par0,
        lower = lb,
        upper = hb
    )
    par1 = coef(sol2)[(end - 1):end] # km and hill_exponent
    par2 = reshape(coef(sol2)[1:(end - 2)], :, 2) # fit monomer decomposition per strain

    # multiplicative model.
    # Each strain decomposes a fixed proportion of what is left due to the previous strain's actvity
    sol3 = curve_fit(
        multifit,
        strain_mat[mfi, :][:],
        y[mfi, :][:], mono[:],
        lower = lb[1:(end - 2)],
        upper = hb[1:(end - 2)]
    )
    par3 = reshape(coef(sol3), :, 2) # fit monomer activity per strain

    # least squares fit directly on the total degradation,
    # instead of independently on each monomer type
    sol4 = curve_fit(
        monofit_total,
        strain_mat[mfi, :][:],
        (sum(data, dims = 2) ./ sum(emtype))[mfi],
        par0t,
        lower = zeros(length(par0t)),
        upper = hb[(end - length(par0t) + 1):end]
    )
    par4 = coef(sol4)[1:(end - 2)] # fit proportion of total C degraded by each strain
    par4mm = coef(sol4)[(end - 1):end] # fit hill parameters: km, exponent

    begin
        ss = [join(strains[x], '|') for x in eachrow(strain_mat)]
        pt = predict(strain_mat, par2, par1...) |> x -> reshape(x, :, 2)
        pt[:, 1] = pt[:, 1] .* emtype[1]
        pt[:, 2] = pt[:, 2] .* emtype[2]
        pt2 = predict(strain_mat, par4, par4mm...) .* sum(emtype)
        odf1 = DataFrame(
            type = "training", polymer = "Sigma FV", strain = ss,
            obsC = data[:, 1], obsR = data[:, 2], obsTot = sum(data, dims = 2)[:],
            predC = pt[:, 1], predR = pt[:, 2], predTotOnly = pt2,
            training = mfi
        )
    end


    ## r^2. -----------------------------------------------------------------------------------------

    # total deg
    obs = sum(data, dims = 2)[:]

    # multiplicative
    pred_mult_each = mm(strain_mat, par3)
    pred_mult = pred_mult_each * emtype

    # monomer model
    pred_each = predict(strain_mat, par2, par1...) |> x -> reshape(x, :, 2)
    pred_mono = pred_each * emtype

    # total degradation model
    pred_tot = predict(strain_mat, par4, par4mm...) .* emtot

    # report
    r2 = [rsquare(pred_mult, obs), rsquare(pred_mono, obs), rsquare(pred_tot, obs)]
    println("multiplicative model", '\t', r2[1])
    println("monomer model", '\t', r2[2])
    println("total degradation model:", '\t', r2[3])

    return r2, [sol3, par3], [sol2, par2, par1], [sol4, par4, par4mm] 

end
