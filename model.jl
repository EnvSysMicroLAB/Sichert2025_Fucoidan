using Statistics
using LsqFit
using DataFrames


# source function definitions
include("./functions.jl")

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


## fit. ---------------------------------------------------------------------------------------------

# only train on communities of size <= nstrains
for i in 1:7
    println("maximal community size: ", i)
    main(strain_mat, y, data, i)
    println("")
end
