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

%% Grid Creation
function [x2, y2, A, b, x, y, U] = gridcreate(N, M)
    % Define Boundary Conditions
    x_0 = 0; x_N = 5; y_0 = 0; y_N = 3;
    U_0 = 0; dU_1 = 0; U_2 = 1;

    % Create Plotting Mesh
    x = linspace(x_0, x_N, N); y = linspace(y_0, y_N, M);
    [x, y] = meshgrid(x, y);
    U = zeros(size(x));

    % Create Solution Points
    x2 = linspace(x_0, x_N, N); y2 = linspace(y_0, y_N, M);
    [x2, y2] = meshgrid(x2, y2);
    x2 = reshape(x2(1:end-1, 2:end-1)', (N-2)*(M-1), 1); y2 = reshape(y2(1:end-1, 2:end-1)', (N-2)*(M-1), 1);

    % Adaptive Grid
    omega = ((6.5 - y2) - x2)*(N - 1)/5;
    omega(omega > 1) = 1;
    theta = ((6.5 - x2) - y2)*(M - 1)/3;
    theta(theta > 1) = 1;

    % Reduce Solution Points to locations needed solving
    ind2 = find((x2 + y2 < (6.5)) + (y2 > 0) == 2);
    d11 = x2 > 3.5;
    d12 = x2 < 4.5;
    d13 = y2 == 0;
    ind3 = find((d11 == 1) + (d12 == 1) + (d13 == 1) == 3);
    ind2 = union(ind2, ind3);
    x2 = x2(ind2); y2 = y2(ind2);

    % Define "x" part of A
    Fx = (spdiags([ -1 2 -1 ], -1:1, (N - 2), (N - 2)));
    C = kron(speye(M - 1), Fx);
    C = spdiags(spdiags(C, -1:1).*[2./(1 + omega), 1./omega, 2./(1 + omega)], -1:1, C)';

    % Define "y" part of A
    Fy = (spdiags([ -1 2 -1 ], -1:1, (M - 1), (M - 1)));
    B = kron(Fy, speye(N - 2));
    B = spdiags(spdiags(B, [-(N-2) 0 (N-2)]).*[2./(1 + theta), 1./theta, 2./(1 + theta)], [-(N-2) 0 (N-2)], B)';
    g = spdiags(B, [-(N-2) 0 (N-2)]);
    g(N - 2: 2*(N - 2) , 3) = -2;
    B = spdiags(g, [-(N-2) 0 (N-2)], B);

    % Reduce components to locations needed solving and combine
    C = C(ind2, ind2); B = B(ind2, ind2);
    A = C + B;

    %BCs
    b = zeros(length(A), 1);
    b(1) = 1;
    for k = length(ind3): length(ind3) + (N-3)
        if x2(k) <= 0.5 || x2(k) >= 4.5
            b(k) = 0;
        elseif x2(k) <= 1.5
            b(k) = x2(k) - 0.5;
        elseif x2(k) <= 3.5
            b(k) = 1;
        elseif x2(k) < 4.5
            b(k) = 0;
        end
    end
end

% Mesh Pts.
N = 201; M = 121;

% Gaussian
[x2, y2, A, b, x, y, U] = gridcreate(N, M);
sol = A\b;

% Plotting
n = 0;
for i = 1:M
    for j = 1:N
        indx = find(x2 == x(i, j));
        indy = find(y2 == y(i, j));
        ind = intersect(indx, indy);
        if isempty(ind) == 1
            if y(i, j) == 0 && x(i,j) > 0.5 && x(i,j) < 1.5
                U(i, j) = x(i, j) - 0.5;
            elseif y(i, j) == 0 && x(i,j) >= 1.5 && x(i,j) <= 3.5
                U(i, j) = 1;
            elseif y(i,j) + x(i,j) > 6.5
                U(i, j) = NaN;
            else
                U(i,j) = 0;
            end
        else
            U(i,j) = sol(ind);
        end
    end
end

figure(2)
contourf(x, y, U, 20)

figure(1)
plot3(x, y, U, '*')

