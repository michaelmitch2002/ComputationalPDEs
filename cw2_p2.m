%% cw2_p1.m

clear all;
close all;
clc;

%% Iterative
err = 10^(-8);

%% Functions
% Iterative Methods
function [U, iterhis] = iterative(A, M, N, b, U, n, err)
    resid = 1;
    iterhis = [];
    iter = 0;
    Tw = M\N;
    cw = M\b;
    while resid > err && iter < 5000
        iter = iter + 1;
        resid = norm(b - A*U)/(n - 2)^2;
        U = Tw*U + cw;
        iterhis = [iterhis, resid];
    end
end

% Jacobi
function [U, iterhis] = Jacobi(iterative, A, b, U, N, err)
    D = spdiags(spdiags(A, 0), 0, (N-2)^2, (N-2)^2);
    C = spdiags(zeros((N - 2)^2, 1),0, A);
    [U, iterhis] = iterative(A, D, -C, b, U, N, err);
end

% Gauss-Seidel
function [U, iterhis] = GS(iterative, A, b, U, N, err)
    D = spdiags(spdiags(A, 0), 0, (N-2)^2, (N-2)^2);
    C = spdiags(zeros((N - 2)^2, 1),0, A);
    V = triu(C);
    L = tril(C); 
    [U, iterhis] = iterative(A, D + L, -V, b, U, N, err);
end

% SOR
function [U, iterhis] = sor(iterative, A, b, U, N, err, omega)
    D = spdiags(spdiags(A, 0), 0, (N-2)^2, (N-2)^2);
    C = spdiags(zeros((N - 2)^2, 1),0, A);
    V = triu(C); L = tril(C); B = D + L;
    M = 1/omega*(D + omega*L); Never = 1/omega*((1 - omega)*D - omega*V);
    [U, iterhis] = iterative(A, M, Never, b, U, N, err);
end

%% Grid

% Construct Grid


N = 11; M = 7;

function [x, y, U] = gridcreate(N, M)
    x_0 = 0; x_N = 5; y_0 = 0; y_N = 3;
    U_0 = 0; dU_1 = 0; U_2 = 1;

    x = linspace(x_0, x_N, N); y = linspace(y_0, y_N, M);
    [x, y] = meshgrid(x, y);
    U = zeros(size(x));


    c = reshape(x(2:end-1, 2:end-1) + y(2:end-1, 2:end-1) < 6.5, (N-2)*(M-2), 1);
    ind = find(c == 1);


    % Construct Matrix
    F = (spdiags([ -1 2 -1 ], -1:1, (N - 2), (N - 2)));
    %F2 = (spdiags([ -1 2 -1 ], -1:1, (N - 2), (N - 2)));
    F3 = (spdiags([ -1 2 -1 ], [-(1) 0 (1)], (M - 2), (M - 2)));
    %A = kron(speye(M - 2), F) + kron(speye(M - 2), F2) + kron(F3, speye(N-2));
    A = kron(speye(M - 2), F) + kron(F3, speye(N-2));
    %A = A(ind, ind);
    full(A)

    b = zeros((M - 2)*(N - 2), 1);
    for k = 1:(N-2)
        % 
        if x(1, k + 1) <= 0.5 || x(1, k + 1) >= 4.5
            b(k) = 0;
        elseif x(1, k+1) < 1.5
            b(k) = x(1, k+1) - 0.5;
        elseif x(1, k+1) <= 3.5
            b(k) = 1;
        elseif x(1, k+1) < 4.5
            b(k) = 0;
        end
    end
end
   

[domainx, domainy] = gridcreate(N, M);

plot(domainx, domainy, '*')