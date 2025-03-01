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

function [x2, y2, A, b] = gridcreate(N, M)
    x_0 = 0; x_N = 5; y_0 = 0; y_N = 3;
    U_0 = 0; dU_1 = 0; U_2 = 1;

    x2 = linspace(x_0, x_N, N); y2 = linspace(y_0, y_N, M);
    [x2, y2] = meshgrid(x2, y2);

    x2 = reshape(x2(2:end-1, 2:end-1)', (N-2)*(M-2), 1); y2 = reshape(y2(2:end-1, 2:end-1)', (N-2)*(M-2), 1);

    omega = ((6.5 - y2) - x2)*(N - 1)/5;
    omega(omega > 1) = 1;
    theta = ((6.5 - x2) - y2)*(M - 1)/3;
    theta(theta > 1) = 1;

    ind2 = find(x2 + y2 < (6.5) == 1);
    x2 = x2(ind2); y2 = y2(ind2);
    d11 = x2(1:N-2) > 3.5
    d12 = x2(1:N-2) < 4.5
    test = find(d11 == d12)
    add = sum(d11 == d12)
    
    % Construct Matrix
    Fx = (spdiags([ -1 2 -1 ], -1:1, (N - 2), (N - 2)));
    C = kron(speye(M - 2), Fx);
    C = spdiags(spdiags(C, -1:1).*[2./(1 + omega), 1./omega, 2./(1 + omega)], -1:1, C)';

    Fy = (spdiags([ -1 2 -1 ], -1:1, (M - 2), (M - 2)));
    B = kron(Fy, speye(N-2));
    B = spdiags(spdiags(B, [-(N-2) 0 (N-2)]).*[2./(1 + theta), 1./theta, 2./(1 + theta)], [-(N-2) 0 (N-2)], B)';

    C = C(ind2, ind2); B = B(ind2, ind2);
    A = C + B;

    A = full(A);
    A2 = zeros(size(A) + add);
    A2(1:size(A), 1:size(A)) = A;
    A2(size(A)+1:end, size(A)+1:end) = 4*eye(add) + diag(-1*ones(add - 1, 1), 1) + diag(-1*ones(add - 1, 1), -1);
    A2(end-add + 1:end, test) = -2*eye(add);
    A2(test, end-add + 1:end) = -eye(add);

    A = A2;

    xtest = x2(1:N-2);
    xtest = xtest(d11 == d12);

    x2 = [x2; xtest];
    y2 = [y2; zeros(size(xtest))];

    b = zeros(length(A), 1);
    b(end - add + 1) = 1;
    for k = 1:(N-2)
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

% Construct Grid
N = 101; M = 61;
   
[x2, y2, A, b] = gridcreate(N, M);
sol = A\b
figure(1)
plot3(x2, y2, sol, '*')
hold on
plot3(linspace(0,5,N), linspace(0,0,N), [0; b(1:70); sol(end - 18: end); b(90:N-2); 0],  '*')

