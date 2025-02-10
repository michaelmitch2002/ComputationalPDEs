%% cw1.m

clear all;
close all;
clc;


u_0 = -1;
u_1 = 1/2;
err = 10^(-5);

%% Jacobian Function
function J = Jacobian2(f, u, N)
    err = 10.0^(-8.0);
    J = zeros(N - 2, N - 2);
    u_base = f(u);
    for i = 2:N - 1
        u(i) = u(i) + err;
        J(:, i - 1) = (f(u) - u_base)/(err);
        u(i) = u(i) - err;
    end
end

%% Jacobian Function
function J = Jacobian(u, A1, A2, w2, N, h)
    Bout = spdiags(A1, [-3 -2 -1 0 1 2 3])...
        + spdiags(A2, [-3 -2 -1 0 1 2 3]).*u(2: N - 1)...
        - spdiags(speye(N - 2), [-3 -2 -1 0 1 2 3]);
        %spdiags(speye(N - 2).*(A2*u(2: N - 1) + w2(:,1)*u(1) + w2(:,2)*u(N)), [-3 -2 -1 0 1 2 3])
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
        iter = iter + 1
        r = f(u);
        %J2 = Jacobian2(f, u, N);
        J = Jac(u);
        Jtest = full(J);
        u(2:N - 1) = u(2:N - 1) - J\r;
    end
end


%% Q1: 

eta_opt = [.1, 0.01, 0.001, 0.0001];
figure (1)
hold on
figure (2)
hold on
for i = 1:length(eta_opt)
    eta = eta_opt(i);
    N = 4001;

    h = 1/(N - 1);
    u = linspace(u_0, u_1, N).';
    x = linspace(0, 1, N);

    A2_2 = spdiags([1 -2 1],-1:1,N - 2,N - 2)*eta/h^2;
    A1_2 = spdiags([-1 0 1],-1:1,N - 2,N - 2)/(2*h);
    w1_2 = zeros(N - 2, 2);
    w1_2(1, 1) = eta/(h^2);
    w1_2(N - 2, 2) = eta/h^2;
    w2_2 = zeros(N - 2, 2);
    w2_2(1, 1) = -1/(2*h);
    w2_2(N - 2, 2) = 1/(2*h);

    A2_4 = spdiags([-1 16 -30 16 -1], -2:2, N - 2, N - 2)*eta/(12*h^2);
    A2_4(1,1:4) = [-20, 6, 4, -1]*eta/(12*h^2);
    A2_4(N - 2, N - 5: N - 2) = [-1, 4, 6, -20]*eta/(12*h^2);
    A1_4 = spdiags([1 -8 0 8 -1], -2:2, N - 2, N - 2)*1/(12*h);
    A1_4(N - 2, N - 5: N - 2) = [-1, 6, -18, 10]*1/(12*h);
    A1_4(1,1:4) = [-10, 18, -6, 1]*1/(12*h);

    w1_4 = zeros(N - 2, 2);
    w1_4(1, 1) = eta/(12*h^2)*11;
    w1_4(2, 1) = -eta/(12*h^2);
    w1_4(N - 3, 2) = -eta/(12*h^2);
    w1_4(N - 2, 2) = eta/(12*h^2)*11;
    
    w2_4 = zeros(N - 2, 2);
    w2_4(1, 1) = -1/(12*h)*3;
    w2_4(2, 1) = 1/(12*h);
    w2_4(N - 3, 2) = -1/(12*h);
    w2_4(N - 2, 2) = 1/(12*h)*3;

    Jac = @(u)Jacobian(u, A2_2, A1_2, w2_2, N, h);
    f2 = @(x)resid_fd(x, A2_2, A1_2, w1_2, w2_2, N);
    [u_2, iter] = fd(f2, Jac, u, err, N);

    Jac = @(u)Jacobian(u, A2_4, A1_4, w2_4, N, h);
    f4 = @(x)resid_fd(x, A2_4, A1_4, w1_4, w2_4, N);
    [u_4, iter] = fd(f4, Jac, u, err, N);

    figure (1)
    plot(x, u_2, 'LineWidth', 1)
    figure (2)
    plot(x, u_4, 'LineWidth', 1)
end
 


