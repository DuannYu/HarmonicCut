function [y_pred, obj, anchors] = harmonic_cut(X, ratio, c, neighbors)
    if nargin < 4
        neighbors = 10;
    end
    n = size(X, 1);
    m = floor(ratio*n);

    % Generate Anchors
    [~, anchors] = litekmeans(X, m);
    % m = floor(log2(n)-2); anchors = bkhk(X, m); m = 2^m;
    % [anchors, ~, ~] = graphgen_anchor(X, m);
    % Construct Bipartite
    B = construct_bigraph(X, anchors, neighbors);
    % [y1,m_vec,~,~,~,~] = coclustering_bipartite_fast1(B, c);
    % A0=sparse(n+m,n+m); A0(1:n,n+1:end)=B; A0(n+1:end,1:n)=B';
    % y_init = finchpp(A0, c);

    m_vec = litekmeans(anchors, c);
    % ClusteringMeasure_new(y, y1)
%     disp('Init Finished')


    %% CD for Harmonic Cut (Slow Version)

    % Z = load('Z');
    % Z = Z.Z;
    % Z_init = Z;
    % m_vec = vec2ind(Z');
    Z = ind2vec(m_vec')';
    n_iter = 10;
    % y'*B'*B*y

    bz = B*Z;
    links = diag(bz'*bz);

    BB = diag(B'*B);

    obj = zeros(n_iter, 1);
    for iter = 1:n_iter
        obj(iter) = sum(1./links);
        if iter > 2 && abs((obj(iter) - obj(iter - 1)) / obj(iter - 1)) < 1e-10
            break;
        end

        for i = 1:m
            col = m_vec(i);
            % j ~= m
            links_new = links + 2*bz'*B(:, i) + BB(i);
            links_new(col) = links(col);

            % j == m
            links_0 = links;
            links_0(col) = links(col) - 2*bz(:, col)'*B(:, i) + BB(i);

            delta = 1./links_new - 1./links_0;

            [~, p] = min(delta);

            if p ~= col
                links(col) = links_0(col);
                links(p) = links_new(p);

                bz(:, col) = bz(:, col) - B(:, i);
                bz(:, p) = bz(:, p) + B(:, i);


                Z(i, p) = 1;
                Z(i, col) = 0;
                m_vec(i) = p;
            end
        end
    end


    Y = B*Z;
    [~, y_pred] = max(Y, [], 2);
end