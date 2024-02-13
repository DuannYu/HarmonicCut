function Z= construct_bigraph(A, B, neighbors)
% each column is a data, compute sqare distance
% 构建最优二部图
% ||xij-aij||_2^2*sij + gamma * sij^2 
% A: data A
% B: data B
% gamma：正则项前朝参数
% neighbor: 近邻个数，0：二部图为全连接

n = size(A, 1);
m = size(B, 1);
distance = L2_distance_1(A',B');
% 找出距每个点最近的点聚类中心
[sorted_distance, sorted_index] = sort(distance, 2);
Z = zeros(n, m);
for i = 1:n
    Z(i, sorted_index(i, 1:neighbors)) = (sorted_distance(i,neighbors+1) - sorted_distance(i, 1:neighbors))/(neighbors*sorted_distance(i,neighbors+1) - sum(sorted_distance(i, 1:neighbors))+eps);
end

