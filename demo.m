%% Settings and Init
% rng(2)
addpath("funs");
addpath("data");
addpath("finchpp");

data_index = 5;

[X, y, dataset_name] = load_dataset(data_index);
X = full(X);
c = length(unique(y));
y_pred = harmonic_cut(X, 0.3, c);

ClusteringMeasure_new(y, y_pred)
% ClusteringMeasure_new(y, y1)