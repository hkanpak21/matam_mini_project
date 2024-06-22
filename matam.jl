### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 26622734-5d85-44a7-8a43-c2bca26d4916
using Pkg

# ╔═╡ 3f1abb0f-13f6-47f4-a906-9efd9c42d142
begin
	Pkg.add("CSV")
	Pkg.add("DataFrames")
	Pkg.add("MultivariateStats")
	Pkg.add("GLM")
end

# ╔═╡ 66a28952-3060-11ef-19a3-9bb037c2dc33
begin
	using Markdown
	using InteractiveUtils
end

# ╔═╡ 0bc2ebf4-67b0-4698-9ea1-fa6f15866ee3
using Statistics

# ╔═╡ 6a642b58-b508-41e2-9a57-5d72c426f88d
using CSV, DataFrames, MultivariateStats, GLM

# ╔═╡ 316173dc-d6db-4b21-8037-db9fe808f1de
begin
	train_data = CSV.read("DATA_split/DATA_4sn_trn.txt", DataFrame, delim='\t', header=false)
	val_data = CSV.read("DATA_split/DATA_4sn_val.txt", DataFrame, delim='\t', header=false)
	test_data = CSV.read("DATA_split/DATA_4sn_tst.txt", DataFrame, delim='\t', header=false)
end;

# ╔═╡ f2c736cd-0011-4fb9-9417-898c8bbd1968
function split_features_labels(data::DataFrame)
    features = Matrix{Float32}(data[:, 1:end-1])
    labels = Vector{Float32}(data[:, end])
    return features, labels
end

# ╔═╡ 1db634d8-04b9-48c0-beda-52f60c3dbaa8
combined_data = vcat(train_data, val_data);

# ╔═╡ 739bb33e-9a98-4709-ab12-6f810f59494e
combined_features, combined_labels = split_features_labels(combined_data);

# ╔═╡ 571e9ca0-e3db-499c-8690-a2dd1a000d92
means = mean(combined_features, dims=1);

# ╔═╡ 6be5fb4c-52f6-4309-8852-6201351e8701
stds = std(combined_features, dims=1);

# ╔═╡ 627a95d9-cb0d-40be-97a3-0e81c8e0d0e7
standardized_features = (combined_features .- means) ./ stds;

# ╔═╡ 2b50fd13-b7b6-4443-92f3-917e353a549e
pca_model = fit(PCA, standardized_features'; maxoutdim=20)  # Ensure maxoutdim is less than the number of features

# ╔═╡ 8240cb9b-73ea-462f-955b-e324df466c6d
combined_features_pca = MultivariateStats.transform(pca_model,standardized_features')

# ╔═╡ 90e395c7-eb40-42f6-8ded-f37af9cb1afc
combined_features_pca_convert = convert(Matrix{Float64}, combined_features_pca);

# ╔═╡ 0466e6b4-3058-45bd-bebb-f60ec836f1a6
combined_labels_convert = convert(Vector{Float64}, combined_labels);

# ╔═╡ 16e36a77-8036-4048-b155-8294b8dcbe3e
train_sample_size = size(train_data, 1);

# ╔═╡ 1725aa76-fc6e-47e0-8f7a-0fe5d420c6ae
train_features_pca = combined_features_pca'[1:train_sample_size, :];

# ╔═╡ df766027-5bb3-4b97-9e01-dde9120e8966
train_labels = combined_labels_convert[1:train_sample_size];

# ╔═╡ ebfa6160-3feb-4a67-9438-1e4ff57e7a59
val_features_pca = combined_features_pca_convert'[train_sample_size + 1:end, :];

# ╔═╡ 5b61d3d3-a25e-4adb-a4f1-c210de76130d
val_labels = combined_labels_convert[train_sample_size + 1:end, :];

# ╔═╡ ffa447d8-d4b6-46cd-9187-06bba13a3451
train_df = DataFrame([Symbol("PC$i") => train_features_pca[:, i] for i in 1:size(train_features_pca, 2)]);

# ╔═╡ 6fe672fb-27e3-4ee9-97fe-a83d5fc89731
train_df.Y = train_labels;

# ╔═╡ 0fbeaef3-da6b-463c-8904-eab4133df53f
val_df = DataFrame([Symbol("PC$i") => val_features_pca[:, i] for i in 1:size(val_features_pca, 2)]);

# ╔═╡ ba9ad3d5-701a-463c-bb02-c81a68cc60a7
val_df[!, :Y] .= val_labels;  # Correctly using broadcasting

# ╔═╡ 177b4acc-24ce-4073-8c73-31b96f71e39f
model = lm(@formula(Y ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10), train_df)

# ╔═╡ a198a33f-17e2-459d-bbcd-13622cc5a7be
predictions = predict(model, val_df)

# ╔═╡ 3410e8af-bc3a-4d2d-811e-54957d43f0d8
mse = mean((val_df.Y - predictions).^2)

# ╔═╡ Cell order:
# ╠═66a28952-3060-11ef-19a3-9bb037c2dc33
# ╠═26622734-5d85-44a7-8a43-c2bca26d4916
# ╠═0bc2ebf4-67b0-4698-9ea1-fa6f15866ee3
# ╠═3f1abb0f-13f6-47f4-a906-9efd9c42d142
# ╠═6a642b58-b508-41e2-9a57-5d72c426f88d
# ╠═316173dc-d6db-4b21-8037-db9fe808f1de
# ╠═f2c736cd-0011-4fb9-9417-898c8bbd1968
# ╠═1db634d8-04b9-48c0-beda-52f60c3dbaa8
# ╠═739bb33e-9a98-4709-ab12-6f810f59494e
# ╠═571e9ca0-e3db-499c-8690-a2dd1a000d92
# ╠═6be5fb4c-52f6-4309-8852-6201351e8701
# ╠═627a95d9-cb0d-40be-97a3-0e81c8e0d0e7
# ╠═2b50fd13-b7b6-4443-92f3-917e353a549e
# ╠═8240cb9b-73ea-462f-955b-e324df466c6d
# ╠═90e395c7-eb40-42f6-8ded-f37af9cb1afc
# ╠═0466e6b4-3058-45bd-bebb-f60ec836f1a6
# ╠═16e36a77-8036-4048-b155-8294b8dcbe3e
# ╠═1725aa76-fc6e-47e0-8f7a-0fe5d420c6ae
# ╠═df766027-5bb3-4b97-9e01-dde9120e8966
# ╠═ebfa6160-3feb-4a67-9438-1e4ff57e7a59
# ╠═5b61d3d3-a25e-4adb-a4f1-c210de76130d
# ╠═ffa447d8-d4b6-46cd-9187-06bba13a3451
# ╠═6fe672fb-27e3-4ee9-97fe-a83d5fc89731
# ╠═0fbeaef3-da6b-463c-8904-eab4133df53f
# ╠═ba9ad3d5-701a-463c-bb02-c81a68cc60a7
# ╠═177b4acc-24ce-4073-8c73-31b96f71e39f
# ╠═a198a33f-17e2-459d-bbcd-13622cc5a7be
# ╠═3410e8af-bc3a-4d2d-811e-54957d43f0d8
