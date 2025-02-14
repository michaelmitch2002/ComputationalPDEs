%% cw2test.m

clear all;
close all;
clc;

%% Setup
omega = 0.5;
err = 10^(-5);
u_0 = 0;
u_1 = 0;
N = 50 + 1;
p = 2;
J = 10000 + 1;
t_end = 1;
k = t_end/(J - 1);
h = 1/(N - 1);
r = k/h^2;



%% Pre-Functionals
% General Functions
function A = discret2(N, r)
    A = r*spdiags([1 -2 1], -1:1, N - 2, N - 2);
end

function u_init = init(x)
    u_init = 4.*x.*(1 - x);
end

% Implicit Functions
function resid = resid_implicit(u, u_past, A, p)
    resid = -u + u_past + A*u.^p;
end

function J = Jacobian(u, A, p, N)
    Bout = spdiags(p*A.*u.^(p - 1), [-1 0 1]) ...
        - spdiags(speye(N - 2), [-1 0 1]);
    J = spdiags(Bout,[-1 0 1], N - 2, N - 2);
end

function [u, iter] = Newtons(f, Jac, u, err, N)
    resid = ones(N - 2, 1);
    iter = 0;
    u_past = u;
    while norm(resid)/N > err
        iter = iter + 1;
        resid = f(u, u_past);
        J = Jac(u);
        u = u - J\resid;
    end
end

%% Finite Difference Methods
% Explicit
function [x, u, t_stop] = explicit(u_init, u_0, u_1, p, N, J, t_end)

    h = 1/(N - 1); k = t_end/(J - 1); r = k/h^2;
    A = discret2(N, r);

    x = linspace(0, 1, N); u = zeros(N, J);
    u(1, :) = u_0; u(N, :) = u_1; u(:, 1) = u_init(x);

    for j = 2:J
        u(2:N-1, j) = u(2:N-1, j - 1) + A*(u(2:N-1, j - 1).^p);
        if norm(u(2:N-1, j)) < 10^(-5)
            u = u(:, 1:j);
            t_stop = (j - 1)*t_end/(J - 1);
            return
        end
    end
    t_stop = (j - 1)*t_end/(J - 1);
end

% Lagged
function [x, u] = lagged(u_init, u_0, u_1, p, N, J, t_end, err, omega)

    h = 1/(N - 1); k = 1/(J - 1); r = k/h^2;
    A = discret2(N, r);

    x = linspace(0, 1, N); u = zeros(N, J);
    u(1, :) = u_0; u(N, :) = u_1; u(:, 1) = u_init(x);

    for j = 2:J
        u_guess = u(2:N - 1, j - 1);
        while norm(u(2:N - 1, j) - u_guess) > err
            u(2:N - 1, j) = u(2 :N - 1, j - 1) + omega*A*u_guess.^p...
                + (1-omega)*A*u(2:N - 1, j - 1).^p;
            u_guess = u(2:N - 1, j);
        end
    end
end

% Implicit
function [x, u] = implicit(u_init, u_0, u_1, p, N, J, t_end, err)

    h = 1/(N - 1); k = 1/(J - 1); r = k/h^2;
    A = discret2(N, r);

    x = linspace(0, 1, N); u = zeros(N, J);
    u(1, :) = u_0; u(N, :) = u_1; u(:, 1) = u_init(x);

    Jac = @(u)Jacobian(u, A, p, N);
    f = @(u, u_past)resid_implicit(u, u_past, A, p);
    Newt = @(f, Jac, u)Newtons(f, Jac, u, err, N);

    for j = 2:J
        u(2:N-1, j) = Newt(f, Jac, u(2 :N - 1, j - 1));
    end
end




%% Solution Space
function [x_truth, u_truth, t_stop] = pt1(init, u_0, u_1, p, N, J, err, omega, t_end)
    J = 10000*100 + 1;
    N = 50*10 + 1;
    N_array = [200, 100, 50, 40];
    J_array = N_array.^2/0.25 + 1;
    %[x_truth, u_truth, t_stop] = explicit(init, u_0, u_1, p, N, J, t_end);
    for i = 1:length(N_array)
        [x_truth, u_truth, t_stop] = explicit(init, u_0, u_1, p, N_array(i), J_array(i), t_end);
    end
    
end

function x = pt2(init, u_0, u_1, p, N, J, err, omega, t_end)
    [x, u_exp, t_stop] = explicit(init, u_0, u_1, p, N, J, t_end);
    [x, u_lagged] = lagged(init, u_0, u_1, p, N, J, t_end, err, omega);
    [x, u_imp] = implicit(init, u_0, u_1, p, N, J, t_end, err);
    
    figure(2)
    hold on
    plot(x,u_imp(:, 1:100:end))
end

%pt2(@init, u_0, u_1, p, N, J, err, omega, t_end)
[x_truth, u_truth, t_stop] = pt1(@init, u_0, u_1, 1, N, J, err, omega, t_end);

