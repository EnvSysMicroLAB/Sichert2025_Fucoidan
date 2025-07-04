using Statistics
using LsqFit
using DataFrames
# source function definitions
include("./functions.jl")


### fit. --------------------------------------------------------------------------------------------

## read training data. ------------------------------------------------------------------------------

# Path to the training data file
p = "training.tsv";

# Read the training data from the file. The function `read_training` is assumed to return:
# - `strains`: A list of strain identifiers
# - `strain_mat`: A matrix indicating the presence of strains
# - `data`: The main community fucoidan decomposition matrix
# - `emtype`: A vector of total carbon per monomer type
strains, strain_mat, data, emtype = read_training(p);

# Calculate the total amount of carbon
emtot = sum(emtype)

# Create a dictionary mapping strain names to their indices
s2i = Dict(k => i for (i, k) in enumerate(strains))

# Identify rows in `strain_mat` where the sum across columns is exactly 1
# In other words, find monocultures
i = sum(strain_mat, dims = 2)[:] .== 1;

# Extract rows from `data` corresponding to monocultures
mono1 = data[i, :]

# Get mean degradation across replicates.
# (Reduce the data by averaging every three consecutive rows in `mono1`)
mono = reduce(
    vcat,
    [
        mean(mono1[i:(i + 2), :], dims = 1) for
            i in 1:3:size(mono1, 1)
    ]
)

# Normalize the first and second columns of `mono` by dividing by
# the corresponding total of each monomer type
mono[:, 1] = @. mono[:, 1] / emtype[1]
mono[:, 2] = @. mono[:, 2] / emtype[2]

# get total potential degradation by summing individual community member contribution
x = strain_mat * mono

# Compute total degradation for each monoculture and normalize each entry by the total available C
m1s = sum(mono1, dims = 2)[:]
m1s = m1s ./ sum(emtype)

# Compute the mean total degradation across triplicates
m1sm = [mean(m1s[i:(i + 2)]) for i in 1:3:length(m1s)]

# Normalize degradation of each coculture as proportion of available monomer
xn = [x ./ emtype for x in eachrow(data)]
y = reduce(hcat, xn)' |> collect

# only train on communities of size <= nstrains
for i in 1:7
    println("maximal community size: ", i)
    r2, m_mult, m_mono, m_tot = main(strain_mat, y, data, emtot, i)
    println("")
end

# prepare output dataframe
nstrains = 3
r2, m_mult, m_mono, m_tot = main(strain_mat, y, data, emtot, nstrains)

begin
    local ss = [join(strains[x], '|') for x in eachrow(strain_mat)]
    local fp = strain_mat * m_mono[2]
    local pt = hill(fp, m_mono[3]...)
    local fp2 = strain_mat * m_tot[2]
    local pt2 = hill(fp2, m_tot[3]...) .* sum(emtype)
    pt[:, 1] = pt[:, 1] .* emtype[1]
    pt[:, 2] = pt[:, 2] .* emtype[2]
    odf1 = DataFrame(
        type = "training", polymer = "Sigma FV", strain = ss,
        obsC = data[:, 1], obsR = data[:, 2], obsTot = sum(data, dims = 2)[:],
        predC = pt[:, 1], predR = pt[:, 2], predTotOnly = pt2,
        training = (sum(strain_mat, dims = 2) .<= nstrains)[:]
    )
end


## verification. ------------------------------------------------------------------------------------

# read verification data
p2 = "verification.tsv";
t0d, vt1 = read_verification(p2);
obsv = [sum(parse.(Float64, x[3:4])) for x in vt1];
obsvC = [parse(Float64, x[3]) for x in vt1];
obsvR = [parse(Float64, x[4]) for x in vt1];
vS = [x[2] for x in vt1];
vP = [x[1] for x in vt1];
predv = Float64[];
predvto = Float64[];
pvC = Float64[];
pvR = Float64[];
xtemp = zeros(Bool, length(strains));
for i in eachindex(vt1)
    v = vt1[i]
    etv = t0d[v[1]][v[2]]
    xi = [s2i[k] for k in split(v[2], '|')]
    xtemp[xi] .= true
    prparts = hill(xtemp' * m_mono[2], m_mono[3]...)[:] .* etv
    pr = prparts |> sum
    pr2 = hill(transpose(xtemp) * m_tot[2], m_tot[3]...) * sum(etv)
    push!(pvC, prparts[1])
    push!(pvR, prparts[2])
    push!(predv, pr)
    push!(predvto, pr2)
    xtemp .= false
end;
odf2 = DataFrame(
    type = "verification", polymer = vP, strain = vS,
    obsC = obsvC, obsR = obsvR, obsTot = obsv,
    predC = pvC, predR = pvR, predTotOnly = predvto,
    training = false
);

# overall prediction data frame
odf = vcat(odf1, odf2)
i = .!odf.training .&& odf.type .== "verification"
a = odf.predC[i] + odf.predR[i]
b = odf.obsTot[i]

scatter(a, b)
