using GLMakie
using Statistics
using LsqFit
using Random: seed!
using GLMakie.Colors
using DataFrames
"""
    fx(br, x)
# model (2nd order)
 yc = Σ(bc_i*x_i) + Σ(bc_ij*x_i*x_j)
 yr = Σ(br_i*x_i) + Σ(br_ij*x_i*x_j)
# parameters
 yc - deg of common
 yr - deg of rare
 b_i - first order terms
 b_ij - second order terms
 x_i - presense/absense of strain i
"""
function fx(br, x)
    b = Float64.(collect(br))
    b_i = b[1:length(x)] # first order terms
    b_ij = b[length(x)+1:end] # second order terms
    x_ij = [x[i] * x[j] for i in 1:(length(x)-1) for j in (i+1):length(x)]
    o = 0.0
    @inbounds for i in eachindex(b_i) # first order terms sum
        o += @views b_i[i] * x[i]
    end
    @inbounds for i in eachindex(b_ij)
        o += @views x_ij[i] * b_ij[i]
    end
    return o
end
function read_training(p)
    function flocal(v, sd)
        out = zeros(Bool, length(sd))
        for x in v
            out[sd[x]] = true
        end
        return out
    end
    data_raw = split.(readlines(p), '\t')
    hdr = popfirst!(data_raw)
    emtype = parse.(Float64, hdr[2:end])
    sraw = split.(first.(data_raw), '|')
    strains = reduce(union, sraw) |> sort
    sd = Dict(k => i for (i, k) in enumerate(strains))
    strain_mat = reduce(hcat, [flocal(x, sd) for x in sraw])' |> collect
    data = reduce(hcat, [parse.(Float64, x[2:3]) for x in data_raw])' |> collect
    return strains, strain_mat, data, emtype
end
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
end
"""
    mm1(Ax,p)
A is a kxn matrix signifying which strains are found in each of the k communities
x is a nx2 matrix where each column is the C,R monoculture degradation of each isolate
the dot product Ax is therefor the monoculture data summed over the relevant strains in the community 
"""
@. hill1(Ax, p) = Ax^p[2] / (p[1]^p[2] + Ax^p[2]);
@. ncdf(Ax, p) = cdf(Normal(p...), Ax)
"""
    monofit(y, p)
TBW
"""
function monofit(y, p)
    # y is a vector that contains the column indexed strain mat followed by the hill function parameters from fitting hill1 
    # p is the nxk matrix of C,R prefences of each strain to be fit
    x = reshape(max.(Ref(0), p[1:end-2]), :, 2)
    A = reshape(max.(Ref(0),y), :, size(x, 1),) |> Matrix{Int}
    hill1p = p[end-1:end]
    Ax = A * x
    c = hill1(Ax[:, 1], hill1p)
    r = hill1(Ax[:, 2], hill1p)
    return [c; r]
end
function monofit_total(y, p)
    # y is a vector that contains the column indexed strain mat followed by the hill function parameters from fitting hill1 
    # p is the nx1 matrix of total degradation ability of each strain to be fit
    x = p[1:end-2]
    hill1p = p[end-1:end]
    A = reshape(y, :, length(x),) |> Matrix{Int}
    Ax = A * x
    return hill1(max.(0, Ax), hill1p)
end
# multiplicative model
function consume_leftover(x)
    curr = maximum(x)
    if length(x)>1
        for v in sort(x, rev=true)[2:end]
            curr = 1-(1-curr)*(1-v)
        end
        # curr = 1-curr
    end
    return curr
end
function mm(strain_mat, mono)
    # x = 1 .- mono
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
function y2bf(y)
    o = zeros(Float64, size(y, 1), 3)
    @inbounds for i in axes(y, 1)
        yi = y[i, :]
        o[i, :] .= @views [yi[1]^2, 2 * yi[1] * yi[2], yi[2]^2]
    end
    return o
end
function bf2y(bf)
    o = zeros(Float64, size(bf, 1), 2)
    @inbounds for i in axes(bf, 1)
        bfi = bf[i, :]
        o[i, :] .= @views sqrt.([bfi[1], bfi[3]])
    end
    return o
end
function fit_to_nstrains(nstrains, mono, m1sm, strain_mat, y)
    par0 = [ones(length(mono[:])); 1.0; 1.0]
    par0t = [m1sm; 1; 5]
    lb = zeros(length(par0))
    hb = ones(length(par0))
    hb[end-1:end] .= Inf
    mfi = sum(strain_mat, dims=2)[:] .<= nstrains
    sol = curve_fit(monofit, strain_mat[mfi, :][:], y[mfi, :][:], par0, lower=lb, upper=hb)
    par2 = reshape(coef(sol)[1:end-2], :, 2)
    par1 = coef(sol)[end-1:end]
    return sol, par1, par2
end
const clrs = parse.(Colorant, ["#003a7d", "#008dff", "#ff73b6", "#c701ff", "#4ecb8d", "#ff9d3a", "#f9e858", "#d83034", :transparent])

p = "training.tsv";
strains, strain_mat, data, emtype = read_training(p);
emtot = sum(emtype)
s2i = Dict(k => i for (i, k) in enumerate(strains))
i = sum(strain_mat, dims=2)[:] .== 1;
begin
    mono1 = data[i, :]
    mono = reduce(vcat, [mean(mono1[i:i+2, :], dims=1) for i in 1:3:size(mono1, 1)])
    @. mono[:, 1] = mono[:, 1] / emtype[1]
    @. mono[:, 2] = mono[:, 2] / emtype[2]
    x = strain_mat * mono
end;
m1s = sum(mono1,dims=2)[:]
m1s = m1s ./ sum(emtype)
m1sm = [mean(m1s[i:i+2]) for i in 1:3:length(m1s)]
y = reduce(hcat, [x ./ emtype for x in eachrow(data)])' |> collect

# fit
begin
    nstrains = 10
    par0 = [ones(length(mono[:])); 1.0; 1.0]
    par0t = [m1sm; 1; 5]
    lb = zeros(length(par0))
    hb = ones(length(par0))
    hb[end-1:end] .= Inf
    mfi = sum(strain_mat, dims=2)[:] .<= nstrains
    # mfi = y[:,1].<=.75
    sol2 = curve_fit(monofit, strain_mat[mfi, :][:], y[mfi, :][:], par0, lower=lb, upper=hb)
    par2 = reshape(coef(sol2)[1:end-2], :, 2)
    par1 = coef(sol2)[end-1:end]
    sol3 = curve_fit(multifit, strain_mat[mfi, :][:], y[mfi, :][:], mono[:], lower=lb[1:end-2], upper=hb[1:end-2])
    par3 = reshape(coef(sol3), :, 2)
    sol4 = curve_fit(monofit_total, strain_mat[mfi, :][:], (sum(data, dims=2)./sum(emtype))[mfi], par0t,
        lower=zeros(length(par0t)), upper=hb[end-length(par0t)+1:end])
    par4 = coef(sol4)[1:end-2]
    par4mm = coef(sol4)[end-1:end]
end;

# prepare output dataframe
begin
    local ss = [join(strains[x], '|') for x ∈ eachrow(strain_mat)]
    local fp = strain_mat * par2
    local pt = hill1(fp, par1)
    local fp2 = strain_mat * par4
    local pt2 = hill1(fp2, par4mm) .* sum(emtype)
    pt[:, 1] = pt[:, 1] .* emtype[1]
    pt[:, 2] = pt[:, 2] .* emtype[2]
    odf1 = DataFrame(type="training", polymer="Sigma FV", strain=ss, 
        obsC=data[:, 1], obsR=data[:, 2], obsTot=sum(data,dims=2)[:], 
        predC=pt[:, 1], predR=pt[:, 2], predTotOnly=pt2,
        training=mfi)
end

# verification data
p2 = "verification.tsv";
t0d, vt1 = read_verification(p2);
obsv = [sum(parse.(Float64, x[3:4])) for x ∈ vt1];
obsvC = [parse(Float64, x[3]) for x ∈ vt1];
obsvR = [parse(Float64, x[4]) for x ∈ vt1];
vS = [x[2] for x ∈ vt1];
vP = [x[1] for x ∈ vt1];
predv = Float64[];
predvto = Float64[];
pvC = Float64[];
pvR = Float64[];
clrd = Dict(k => clrs[i] for (i, k) in enumerate(keys(t0d)));
clrv = Colorant[];
xtemp = zeros(Bool, length(strains));
for i in eachindex(vt1)
    v = vt1[i]
    etv = t0d[v[1]][v[2]]
    xi = [s2i[k] for k in split(v[2], '|')]
    xtemp[xi] .= true
    prparts = hill1(xtemp' * par2, par1)[:] .* etv
    pr = prparts |> sum
    pr2 = hill1(transpose(xtemp) * par4, par4mm) * sum(etv)
    push!(pvC, prparts[1])
    push!(pvR, prparts[2])
    push!(predv, pr)
    push!(predvto, pr2)
    push!(clrv, clrd[v[1]])
    xtemp .= false
end;
odf2 = DataFrame(type="verification", polymer=vP, strain=vS, 
    obsC=obsvC, obsR=obsvR, obsTot=obsv,
    predC=pvC, predR=pvR, predTotOnly=predvto,
    training=false);


# r2
ver_strain_mat = zeros(Bool, length(vt1), 7)
ver_msize = Vector{Vector{Float64}}(undef, length(vt1))
for i in eachindex(vt1)
    v = vt1[i]
    etv = t0d[v[1]]
    mFuc = mean(first.(values(etv)))
    mRare = mean(getindex.(values(etv),2))
    ver_msize[i] = [mFuc, mRare]
    xi = [s2i[k] for k in split(v[2], '|')]
    ver_strain_mat[i,xi] .= true
    xtemp .= false
end
d = sum(data,dims=2)[:]
r2 = zeros(Float64,7, 2)
function calc_r2(obs,pred)
    ȳ = mean(obs)
    sstot = sum((obs .- ȳ).^2)
    ssres = sum((pred .- obs).^2)
    return 1 - (ssres/sstot)
end
for n in 1:7
    sol, sp1, sp2 =  fit_to_nstrains(n, mono, m1sm, strain_mat, y)
    tpred = hill1(strain_mat * sp2, sp1) * emtype
    vpred_raw = hill1(ver_strain_mat * sp2, sp1)
    vpred = [sum(vpred_raw[i] * ver_msize[i]) for i in eachindex(ver_msize)]
    r2[n, 1] = calc_r2(d,tpred)
    r2[n, 2] = calc_r2(obsv,vpred)
end


# first part of plot
begin

    f = Figure(size=(1500, 400), fontsize=20)
    gl1 = GridLayout(f[1, 1])
    gl2 = GridLayout(f[1, 2])
    gl3 = GridLayout(f[1, 3])
    gl4 = GridLayout(f[1, 4])
    ax1 = Axis(gl1[1, 1], xlabel="C [%]", ylabel="R [%]", title="Strain\nparameter optimization")
    ax2 = Axis(gl2[1, 1], xlabel="Sum of individual activities", ylabel="Observed proportion degraded", title="Monoculture\nparameters")
    ax3 = Axis(gl3[1, 1], xlabel="Sum of individual activities", title="Optimized\nparameters")
    ax4 = Axis(gl4[1, 1], xlabel="maximal community size", ylabel=rich("R",superscript("2")))
    # parameter change
    pnts = [(Point2f(mono[i, :] .* 100), Point2f(par2[i, :] .* 100)) for i in axes(mono, 1)]
    linesegments!(ax1, pnts, color=:lightgrey, linewidth=3)
    scatter!(ax1, mono .* 100, color=:transparent, strokewidth=3, markersize=10, label="Monoculture\nparameter\n")
    scatter!(ax1, par2 .* 100, color=:black, strokewidth=3, markersize=10, label="Optimized\nparameter")
    limits!(ax1, 0, 100, 0, 100)
    Legend(gl1[1,2], ax1; rowgap=30)
    colgap!(f.layout,1,30)
    colsize!(f.layout, 1, 410)
    # influence on fit
    clr = Vector{Symbol}(undef, length(x[:]))
    clr[axes(x, 1)] .= :grey
    clr[size(x, 1)+1:end] .= :orange
    x2 = strain_mat * par2
    x3 = strain_mat * mono
    xt = 0:0.01:maximum(x2[:])
    yt = hill1(xt, par1)
    scatter!(ax3, x2[:], y[:], color=clr)
    scatter!(ax2, x3[:], y[:], color=clr)
    lines!(ax3, xt, yt, color=:black, linewidth=5)
    lines!(ax2, xt, yt, color=:black, linewidth=5)
    limits!(ax2, 0, nothing, 0, nothing)
    limits!(ax3, 0, nothing, 0, nothing)
    elem_1 = MarkerElement(color = :grey, marker = :circle, markersize = 15)
    elem_2 = MarkerElement(color = :orange, marker = :circle, markersize = 15)
    Legend(gl3[1,2], [elem_1, elem_2], ["Fucose", "Rare"])
    colgap!(f.layout,3,30)
    colsize!(f.layout,3,320)
    # D
    scatterlines!(ax4, 1:7, r2[:, 1], label="training", color="grey", linewidth=5, markersize=20)
    scatterlines!(ax4, 1:7, r2[:, 2], label="verification", color="blue", linewidth=5, markersize=20)
    limits!(ax4, 0, 8, 0, 1)
    axislegend(ax4; position=:rb)

    for (label, layout) in zip(["A", "B", "C", "D"], [gl1, gl2, gl3, gl4])
        Label(layout[1, 1, TopLeft()], label,
            fontsize = 26,
            font = :bold,
            padding = (0, 50, 5, 0),
            halign = :right)
    end

end;

using CairoMakie
CairoMakie.activate!()
save("model.pdf",f; size=(1500, 400))

