function Z= construct_bigraph(A, B, neighbors)
% each column is a data, compute sqare distance
% �������Ŷ���ͼ
% ||xij-aij||_2^2*sij + gamma * sij^2 
% A: data A
% B: data B
% gamma��������ǰ������
% neighbor: ���ڸ�����0������ͼΪȫ����

n = size(A, 1);
m = size(B, 1);
distance = L2_distance_1(A',B');
% �ҳ���ÿ��������ĵ��������
[sorted_distance, sorted_index] = sort(distance, 2);
Z = zeros(n, m);
for i = 1:n
    Z(i, sorted_index(i, 1:neighbors)) = (sorted_distance(i,neighbors+1) - sorted_distance(i, 1:neighbors))/(neighbors*sorted_distance(i,neighbors+1) - sum(sorted_distance(i, 1:neighbors))+eps);
end

