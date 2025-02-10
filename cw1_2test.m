%% cw2test.m

clear all;
close all;
clc;

global N h J k A

u_0 = 0;
u_1 = 0;

N = 51;
p = 2;
J = 10001;
eta = 0.01;
h = 1/(N - 1);
k = 1/(J - 1);
r = k/h^2
omega = 0.5;
err = 10^(-5)

x = linspace(0, 1, N);
u = zeros(N, J);
u(1, :) = u_0;
u(N, :) = u_1;
u(:, 1) = 4.*x.*(1 - x);

A = (diag(-2*ones(N - 2, 1)) + diag(ones(N - 3, 1), 1) + diag(ones(N - 3, 1), -1));

% %% explicit
% 
% 
% figure(1)
% hold on
% for j = 2:J
%     u(2:N-1, j) = u(2:N-1, j - 1) + r*A*(u(2:N-1, j - 1).^p);
%     if mod(j, 1000) == 0
%         plot(x, u(:, j), 'LineWidth', 1)
%     end
% end

%% lagged

figure(1)
hold on
for j = 2:J
    u_guess = u(2:N - 1, j - 1)
    while norm(u(2:N - 1, j) - u_guess) > err
        u(2:N - 1, j) = u(2:N-1, j - 1) + r*omega*A*u_guess.^p + r*(1-omega)*A*u(2:N - 1, j - 1).^p;
        u_guess = u(2:N - 1, j);
    end
    if mod(j, 100) == 0
        plot(x, u(:, j), 'LineWidth', 1)
    end
end



%% fully implicit

% % Jacobian Function
% function J = Jacobian(f, u, u_past, A, r, p)
%     global N
%     err = 10.0^(-8.0);
%     J = zeros(N - 2, N - 2);
%     u_base = f(u, u_past, A, r, p);
%     for i = 1:N - 2
%         u(i) = u(i) + err;
%         J(:, i) = (f(u, u_past, A, r, p) - u_base)/(err);
%         u(i) = u(i) - err;
%     end
% end
% 
% % ODE Approximation Residuals
% function resid = resid_implicit(u, u_past, A, r, p)
%     resid = -u + u_past + r.*A*u.^p;
% end
% 
% %% Complete Approximation w/ Newton's
% function [u, iter] = fd(f, u, err, A, r, p)
%     global N eta h
%     resid = ones(N - 2, 1);
%     iter = 0;
%     u_past = u;
%     while norm(resid)/N > err
%         iter = iter + 1
%         resid = f(u, u_past, A, r, p);
%         J = Jacobian(f, u, u_past, A, r, p);
%         u = u - J\resid;
%     end
% end
% 
% figure(1)
% hold on
% for j = 2:J
%     u(2:N-1, j) = fd(@resid_implicit, u(2:N-1, j - 1), err, A, r, p);
%     if mod(j, 10) == 0
%         plot(x, u(:, j), 'LineWidth', 1)
%     end
% end
