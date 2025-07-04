# using WGLMakie
using Statistics
using LsqFit
using Random: seed!
using Colors
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
"""
    obj(b, p)
b - taylor-series coefficient vector
b is split evenly into bc and br representing common and rare model coefficients
bc[1:nstrains] - first order terms
bc[nstrains+1:end] - second order terms
p[1] = strain_mat
p[2] = emtype
p[3] = observed data
"""
# function obj(b, p)
#     xmat = p[1] # strain mat
#     emtype = p[2] # polymer bond frequencies: [C, R]
#     obs = p[3] # observed degradation
#     blen = sum(1:size(xmat, 2))
#     bc = b[1:blen]
#     br = b[blen+1:end]
#     pred = zeros(Float64, size(xmat, 1), 2)
#     for i in axes(xmat, 1)
#         pred[i, 1] = fx(bc, xmat[i, :]) * emtype[1]
#         pred[i, 2] = fx(br, xmat[i, :]) * emtype[2]
#     end
#     fvu1 = fvu(obs[:, 1], pred[:, 1])
#     fvu2 = fvu(obs[:, 2], pred[:, 2])
#     return sqrt((fvu1^2 + fvu2^2)/2)
# end;
# function fit_model(data, strain_mat, emtype)
#     _p = (strain_mat, emtype, data)
#     Random.seed!(1234)
#     vlen = sum(1:size(strain_mat, 2)) * 2
#     # x0 = rand(Float64, vlen)
#     x0 = zeros(Float64, vlen)
#     optf = OptimizationFunction(obj, Optimization.AutoFiniteDiff())
#     prob = OptimizationProblem(optf, x0, _p)
#     sol = solve(prob, BFGS())
#     return sol
# end
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

# # write output
# odf = vcat(odf1, odf2);
# od = "output_final_may2024"
# !isdir(od) && mkdir(od)
# open("$od/obs_pred_train_and_verify_combined.tsv", "w") do io
#     println(io, join(names(odf), '\t'))
#     for i ∈ axes(odf, 1)
#         println(io, join(odf[i, :], '\t'))
#     end
# end;
# open("$od/w_n3.mat", "w") do iow
#     println(iow, "model: (Σx)^n / (km^n + (Σx)^n)")
#     println(iow, "n = $(par1[2])")
#     println(iow, "km = $(par1[1])")
#     println(iow, "---")
#     println(iow, "strain\tC\tR")
#     for i ∈ axes(par2, 1)
#         println(iow, strains[i], '\t', par2[i, 1], '\t', par2[i, 2])
#     end
# end

# first part of plot
browser_display()
begin
    f = Figure(size=(1000, 700))
    gl1 = GridLayout(f[1:2, 1])
    # gl2 = GridLayout(f[2, 1])
    gl3 = GridLayout(f[3, 1])
    ax0 = Axis(gl1[1, 2], xlabel="C [%]", ylabel="R [%]")
    ax4 = Axis(gl1[1, 3], xlabel="Predicted", ylabel="Observed", title="Multiplicative\ninit")
    ax5 = Axis(gl1[1, 4], xlabel="Predicted", title="multiplicative\noptimized")
    ax6 = Axis(gl1[2, 2], xlabel="C [%]", ylabel="R [%]")
    ax7 = Axis(gl1[2, 3], xlabel="Sum of individual activities", ylabel="Observed", title="Additive\ninit")
    ax = Axis(gl1[2, 4], xlabel="Sum of individual activities", title="Additive\noptimized")
    ax2 = Axis(gl3[1, 1], xlabel="Observed", ylabel="Predicted", title="Training")
    ax3 = Axis(gl3[1, 2], xlabel="Observed", ylabel="Predicted", title="Verification")
    hideydecorations!(ax5, ticks=false, grid=false)
    hideydecorations!(ax, ticks=false, grid=false)
    pnts = [(Point2f(mono[i, :] .* 100), Point2f(par2[i, :] .* 100)) for i in axes(mono, 1)]
    pnts2 = [(Point2f(mono[i, :] .* 100), Point2f(par3[i, :] .* 100)) for i in axes(mono, 1)]
    linesegments!(ax6, pnts, color=:lightgrey, linewidth=5)
    linesegments!(ax0, pnts2, color=:lightgrey, linewidth=5)
    scatter!(ax0, par3 .* 100, color=:black, strokewidth=3, markersize=10)
    scatter!(ax6, par2 .* 100, color=:black, strokewidth=3, markersize=10)
    limits!(ax0, 0, 100, 0, 100)
    limits!(ax6, 0, 100, 0, 100)
    clr = Vector{Symbol}(undef, length(x[:]))
    clr[axes(x, 1)] .= :grey
    clr[size(x, 1)+1:end] .= :orange
    # multiplicative
    ym = mm(strain_mat, mono)
    scatter!(ax4, ym[:], y[:], color=clr)
    ablines!(ax4, 0, 1)
    scatter!(ax5, mm(strain_mat, par3)[:], y[:], color=clr)
    ablines!(ax5, 0, 1)
    x2 = strain_mat * par2
    x3 = strain_mat * mono
    xt = 0:0.01:maximum(x2[:])
    yt = hill1(xt, par1)
    scatter!(ax, x2[:], y[:], color=clr)
    scatter!(ax7, x3[:], y[:], color=clr)
    lines!(ax, xt, yt, color=:black, linewidth=5)
    lines!(ax7, xt, yt, color=:black, linewidth=5)
    linkaxes!(ax0, ax6)
    obs = sum(data, dims=2)[:]
    pt = hill1(x2, par1)
    pt[:, 1] = pt[:, 1] .* emtype[1]
    pt[:, 2] = pt[:, 2] .* emtype[2]
    pred = sum(pt, dims=2)[:]
    scatter!(ax2, obs, pred,
        color=:black,
        markersize=15,
        strokecolor=:white,
        strokewidth=1)
    ablines!(ax2, 0, 1,
        color=:black,
        linewidth=5)
    # verification
    scatter!(ax3, obsv, predv,
        color=clrv,
        markersize=15,
        strokecolor=:black,
        strokewidth=1)
    ablines!(ax3, 0, 1,
        color=:black,
        linewidth=5)
    ablines!(ax,0,1)
    ablines!(ax7,0,1)
end
f


sm2 = strain_mat[sum(strain_mat,dims=2)[:] .> 1, :]
o = [Matrix{Float64}(undef,0,2) for _ in axes(par2,1)]
for i in axes(sm2,1)
    js = findall(sm2[i,:])
    # c1 = hill1(sum(par2[js,1]), par1)
    # r1 = hill1(sum(par2[js,2]), par1)
    c1 = sum(par2[js,1])
    r1 = sum(par2[js,2])
        for j1 in js
            # c2 = hill1(sum(par2[setdiff(js,j1),1]), par1)
            # r2 = hill1(sum(par2[setdiff(js,j1),2]), par1)
            c2 = par2[j1,1]
            r2 = par2[j1,2]
            # v1 = c1>0 ? 1 - (c2/c1) : 0
            # v2 = r1>0 ? 1 - (r2/r1) : 0
            v1 = c1>0 ? c2/c1 : 0.0
            v2 = r1>0 ? r2/r1 : 0.0
            o[j1] = [o[j1]; v1 v2]
        end
end

begin
    f = Figure(size=(400,1000))
    axs = [Axis(f[i,1], aspect=4, title=strains[i]) for i in eachindex(o)]
    ss = .02
    for i in eachindex(o)
        od = o[i]
        if length(unique(od[:,1]))>1
            hist!(axs[i],od[:,1], color=:black,bins=-ss/2:ss:1)
        end
        if length(unique(od[:,2]))>1
            hist!(axs[i],od[:,2], color=(:orange,.85),bins=-ss/2:ss:1)
        end
    end
end
f

x = sum(strain_mat,dims=2)[:]
xt = strain_mat * par2
xth = hill1(xt,par1)
i = sum(strain_mat,dims=2)[:].<=10
scatter(xth[i,1],y[i,1],strokewidth=1)
scatter!(xth[i,2],y[i,2],strokewidth=1)

scatter(x,y[:,1], strokewidth=1)
