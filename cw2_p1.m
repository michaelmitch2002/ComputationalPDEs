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

%% Setup

% Construct Grid
x_0 = 0; x_N = 1; y_0 = 0; y_N = 1;
U_x0 = 0; U_y0 = 0; U_yN = 0; U_xN = 1;
delta = 1/4;
N = 5;

function [x, y, U, A, b] = gridcreate(N, x_0, x_N, y_0, y_N, U_x0, U_y0, U_yN, U_xN, delta)
    x = linspace(x_0, x_N, N); y = linspace(y_0, y_N, N);
    domain = meshgrid(x, y);
    
    U = zeros(size(domain));
    U(:, 1) = U_y0; U(:, end) = U_yN; U(1, :) = U_x0;
    U(end, ceil((N - 1)*(1/2 - delta)) + 1 : floor((N - 1)*(1/2 + delta)) + 1) = U_xN;
    
    % Construct Matrix
    F = spdiags([-1 2 -1], -1:1, N - 2, N - 2);
    A = kron(F, speye(N - 2)) + kron(speye(N - 2), F);
    
    % Add Initial Conditions
    b = zeros(N - 2, N - 2);
    for n = 2:N - 1
        for j = 2:N - 1
            if n == 2
                b(n - 1, j - 1) = b(n - 1, j - 1) + U(n - 1, j);
            elseif n == N - 1
                b(n - 1, j - 1) = b(n - 1, j - 1) + U(n + 1, j);
            end
            if j == 2
                b(n - 1, j - 1) = b(n - 1, j - 1) +  U(n, j - 1);
            elseif j == N - 1
                b(n - 1, j - 1) = b(n - 1, j - 1) + U(n, j + 1);
            end
        end
    end
    b = reshape(b, [(N - 2)^2, 1]);
end

[x, y, U, A, b] = gridcreate(N, x_0, x_N, y_0, y_N, U_x0, U_y0, U_yN, U_xN, delta);

%% 2.1
%  Gaussian Solve
U_21 = U;
U_21(2: end - 1, 2:end - 1) = reshape(A\b, [N - 2, N - 2]);

figure(1)
tcl = tiledlayout(2,2);
xlabel(tcl,'$x$', Interpreter = 'latex')
ylabel(tcl,'$y$', Interpreter = 'latex')

%% 2.2
% Jacobi
U_22J = U;
[Upart, iterhisJ] = Jacobi(@iterative, A, b, reshape(U_22J(2: end - 1, 2:end - 1), [(N - 2)^2, 1]), N, err);
U_22J(2: end - 1, 2: end - 1) = reshape(Upart, [N - 2, N - 2]);

% Gauss-Seidel
U_22GS = U;
[Upart, iterhisGS] = GS(@iterative, A, b, reshape(U_22GS(2: end - 1, 2:end - 1), [(N - 2)^2, 1]), N, err);
U_22GS(2: end - 1, 2: end - 1) = reshape(Upart, [N - 2, N - 2]);

% SOR
% SOR B
h = 1/(N - 1);
theor = 2/(1 + sin(pi*h));
omegas = linspace(0, 2, 50);
omegas = omegas(2:49);
iters = zeros(48, 1);
for i = 1:length(omegas)
    U_22SORtest = U;
    [Upart, iterhisSOR] = sor(@iterative, A, b, reshape(U_22SORtest(2: end - 1, 2:end - 1), [(N - 2)^2, 1]), N, err, omegas(i));
    iters(i) = length(iterhisSOR);
end
[miniters, argmin] = min(iters);

% SOR A
U_22SOR = U;
omega = omegas(argmin);
[Upart, iterhisSOR] = sor(@iterative, A, b, reshape(U_22SOR(2: end - 1, 2:end - 1), [(N - 2)^2, 1]), N, err, omega);
U_22SOR(2: end - 1, 2: end - 1) = reshape(Upart, [N - 2, N - 2]);

%% 2.2
figure (1)
tcl = tiledlayout(2,2);
fontsize(18, 'points')
xlabel(tcl,'$x$', Interpreter = 'latex')
ylabel(tcl,'$y$', Interpreter = 'latex')

nexttile()
contourf(transpose(x), transpose(y), transpose(U_21))
fontsize(18, 'points')
title('Gaussian Elimination')
clim([0,1])

nexttile()
contourf(transpose(x), transpose(y), transpose(U_22J))
fontsize(18, 'points')
title('Jacobi')
clim([0,1])

nexttile()
contourf(transpose(x), transpose(y), transpose(U_22GS))
fontsize(18, 'points')
title('Gauss-Seidel')
clim([0,1])

nexttile()
contourf(transpose(x), transpose(y), transpose(U_22SOR))
fontsize(18, 'points')
title('SOR')
clim([0,1])

cb = colorbar(); 
cb.Layout.Tile = 'east'; 

%% 2.3 
errJ = norm(U_21 - U_22J);
errGS = norm(U_21 - U_22GS);
errSOR = norm(U_21 - U_22SOR);

%% 2.4
figure(2)
semilogy(iterhisJ, 'b', 'LineWidth', 1)
hold on
semilogy(iterhisGS, 'r', 'LineWidth', 1)
semilogy(iterhisSOR, 'g', 'LineWidth', 1)
fontsize(18, 'points')
xlabel('Iteration', Interpreter = 'latex')
ylabel('$||b - AU||$', Interpreter = 'latex')
legend('Jacobi', 'Gauss-Seidel', 'SOR')

%% 2.5
figure(3)
h = 1/(N - 1);
semilogy(omegas, iters, 'b*-', 'LineWidth', 1)
hold on
xline(theor, 'k--', 'LineWidth', 1)
fontsize(18, 'points')
xlabel('$\omega$', Interpreter = 'latex')
ylabel('Iterations', Interpreter = 'latex')
leg = ['Experimental'; '$2 - 2\pi h$'];
legend(leg, Interpreter= 'latex')

% %% 2.7
% Ns = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33];
% omegatest = zeros(size(Ns));
% theortest = zeros(size(Ns));
% leg = ['$N$ = 05'; '$N$ = 07'; '$N$ = 09'; '$N$ = 11'; '$N$ = 13'; '$N$ = 15'; '$N$ = 17'; '$N$ = 19'; '$N$ = 21'; '$N$ = 23'; '$N$ = 25'; '$N$ = 27'; '$N$ = 29'; '$N$ = 31'; '$N$ = 33'];
% for k = 1:length(Ns)
%     N = Ns(k);
% 
%     [x, y, U, A, b] = gridcreate(N, x_0, x_N, y_0, y_N, U_x0, U_y0, U_yN, U_xN, delta)
% 
%     % SOR
%     % SOR B
%     h = 1/(N - 1);
%     theortest(k) = 2/(1 + sin(pi*h));
%     omegas = linspace(0.1, 1.99, 50);
%     iters = zeros(50, 1);
%     for i = 1:length(omegas)
%         U_22SORtest = U;
%         [Upart, iterhisSOR] = sor(@iterative, A, b, reshape(U_22SORtest(2: end - 1, 2:end - 1), [(N - 2)^2, 1]), N, err, omegas(i));
%         iters(i) = length(iterhisSOR);
%         if iters(i) == min(iters(1:i))
%             Upartreal = Upart;
%             iterhisSORreal = iterhisSOR;
%         end
%     end
%     [miniters, argmin] = min(iters);
%     omegatest(k) = omegas(argmin);
% 
%     figure(6)
%     if k == 2
%         hold on
%         fontsize(18, 'points')
%         xlabel('Iteration', Interpreter = 'latex')
%         ylabel('$||b - AU||$', Interpreter = 'latex')
%     end
%     semilogy(iterhisSORreal, 'LineWidth', 1)
% end
% figure(6)
% legend(leg, Interpreter = 'latex')
% figure (5)
% fontsize(18, 'points')
% hold on
% plot(Ns, omegatest, 'b*-', 'LineWidth', 1)
% plot(Ns, theortest, 'r*-', 'LineWidth', 1)
% legend('Experimental', '$2 - 2\pi h$', Interpreter = 'latex')
% xlabel('$N$', Interpreter = 'latex')
% ylabel('$\omega$', Interpreter = 'latex')
% 
% U_22SOR = U;
% U_22SOR(2: end - 1, 2: end - 1) = reshape(Upartreal, [N - 2, N - 2]);

%% 2.9

% Construct Grid

N = 3001;
[x_real, y_real, U, A, b] = gridcreate(N, x_0, x_N, y_0, y_N, U_x0, U_y0, U_yN, U_xN, delta);


U_real = U;
U_real(2: end - 1, 2:end - 1) = reshape(A\b, [N - 2, N - 2]);

figure(10)
xlabel('$x$', Interpreter = 'latex')
ylabel('$y$', Interpreter = 'latex')
contourf(transpose(x_real), transpose(y_real), transpose(U_real))
fontsize(18, 'points')
title('Gaussian Elimination')
clim([0,1])

Ns = [11, 21, 41, 61, 81, 101, 201, 501, 1001, 1201, 1501, 2001];
residN = zeros(size(Ns))

vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
c = ismember(x_real, vals);
index_real = find(c == 1);


err = 10^(-7);
resid = 1;
n = 0
while n < length(Ns) && resid > err
    n = n + 1
    N = Ns(n)
    [x, y, U, A, b] = gridcreate(N, x_0, x_N, y_0, y_N, U_x0, U_y0, U_yN, U_xN, delta);
    U_test = U;
    U_test(2: end - 1, 2:end - 1) = reshape(A\b, [N - 2, N - 2]);
    c = ismember(x, vals);
    index = find(c == 1);
    resid = norm(U_test(index, index) - U_real(index_real, index_real))/81
    residN(n) = resid
end

figure(11)
semilogy(Ns, residN)