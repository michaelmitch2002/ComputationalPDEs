%% cw1.m

clear all;
close all;
clc;

%% Jacobian Function
function J = Jacobian(u, A1, A2, w2, N)
    Bout = spdiags(A1, [-3 -2 -1 0 1 2 3]) ...
        + spdiags(u(2: N - 1).*A2, [-3 -2 -1 0 1 2 3])...
        + spdiags(speye(N - 2).*((A2*u(2: N - 1) - 1) + w2(:, 1)*u(1) + w2(:, 2)*u(N)), [-3 -2 -1 0 1 2 3]);
        
    J = spdiags(Bout,[-3 -2 -1 0 1 2 3], N - 2, N - 2);
end

%% ODE Approximation Residuals

function r = resid_fd(u, A1, A2, w1, w2, N)
    r = A1*u(2: N - 1) + u(2: N - 1).*(A2*u(2: N - 1) - 1) + ...
        w1(:,1)*u(1) + w1(:,2)*u(N) + ...
        u(2: N - 1).*w2(:,1).*u(1) + u(2: N - 1).*w2(:,2).*u(N);
end

%% Complete Approximation w/ Newton's
function [u, iter] = fd(f, Jac, u, err, N)
    r = ones(N, 1);
    iter = 0;
    while norm(r)/N > err
        iter = iter + 1;
        r = f(u);
        J = Jac(u);
        u(2:N - 1) = u(2:N - 1) - J\r;
    end
end

%% Create Arrays

function [A2, A1, w1, w2] = twoorder(N, eta, h)
    A2 = spdiags([1 -2 1], -1:1, N - 2, N - 2)*eta/h^2;
    A1 = spdiags([-1 0 1], -1:1, N - 2, N - 2)/(2*h);

    w1 = zeros(N - 2, 2);
    w1(1, 1) = eta/(h^2);
    w1(N - 2, 2) = eta/h^2;
    w2 = zeros(N - 2, 2);
    w2(1, 1) = -1/(2*h);
    w2(N - 2, 2) = 1/(2*h);
end

function [A2, A1, w1, w2] = fourorder(N, eta, h)
    A2 = spdiags([-1 16 -30 16 -1], -2:2, N - 2, N - 2)*eta/(12*h^2);
    A2(1,1:4) = [-20, 6, 4, -1]*eta/(12*h^2);
    A2(N - 2, N - 5: N - 2) = [-1, 4, 6, -20]*eta/(12*h^2);
    A1 = spdiags([1 -8 0 8 -1], -2:2, N - 2, N - 2)*1/(12*h);
    A1(N - 2, N - 5: N - 2) = [-1, 6, -18, 10]*1/(12*h);
    A1(1,1:4) = [-10, 18, -6, 1]*1/(12*h);

    w1 = zeros(N - 2, 2);
    w1(1, 1) = eta/(12*h^2)*11;
    w1(2, 1) = -eta/(12*h^2);
    w1(N - 3, 2) = -eta/(12*h^2);
    w1(N - 2, 2) = eta/(12*h^2)*11;
    
    w2 = zeros(N - 2, 2);
    w2(1, 1) = -1/(12*h)*3;
    w2(2, 1) = 1/(12*h);
    w2(N - 3, 2) = -1/(12*h);
    w2(N - 2, 2) = 1/(12*h)*3;
end

%% Finality

function [x, u, iter] = sol(N, eta, err, u_0, u_1, order)
    h = 1/(N - 1);
    u = linspace(u_0, u_1, N).';
    x = linspace(0, 1, N);
    if order == 2
        [A2, A1, w1, w2] = twoorder(N, eta, h);
    elseif order == 4
        [A2, A1, w1, w2] = fourorder(N, eta, h);
    end
    Jac = @(u)Jacobian(u, A2, A1, w2, N);
    f2 = @(x)resid_fd(x, A2, A1, w1, w2, N);
    [u, iter] = fd(f2, Jac, u, err, N);
end

%% Q1: 

u_0 = -1;
u_1 = 1/2;
tol = 10^(-8);
eta_opt = [.1, 0.01, 0.001];
N_truth = 200001;
N_array = [301, 401, 501, 1001];
err = zeros(length(eta_opt), length(N_array), 2)
figure (1)
hold on
figure (2)
hold on
figure (3)
hold on

for i = 1:length(eta_opt)

    [x_truth, u_truth, iter] = sol(N_truth, eta_opt(i), tol, u_0, u_1, 4);
    ind_truth = x_truth == 0.72  | x_truth == 0.75;

    for j = 1:length(N_array)
        [x, u_2, iter] = sol(N_array(j), eta_opt(i), tol, u_0, u_1, 2);
        [x, u_4, iter] = sol(N_array(j), eta_opt(i), tol, u_0, u_1, 4);
        ind = x == 0.72  | x == 0.75;
        err(i, j, 1) = sum((u_2(ind) - u_truth(ind_truth)).^2);
        err(i, j, 2) = sum((u_4(ind) - u_truth(ind_truth)).^2);
    end

    figure (1)
    plot(x, u_2, 'LineWidth', 1)
    figure (2)
    plot(x, u_4, 'LineWidth', 1)
    figure (3)
    plot(x_truth, u_truth, 'LineWidth', 1)
end
figure (4)
hold on 
loglog(1./(N_array - 1), err(3, :, 1), 'g',  'LineWidth', 1)
figure(5)
hold on
loglog(1./(N_array - 1), err(3, :, 2), 'r', 'LineWidth', 1)



