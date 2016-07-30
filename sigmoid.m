% This function implements the sigmoid function. It computes and returns
% the output of the sigmoid function as well as its gradient. You do *not*
% have to modify this function.
function [f_a, grad_f_a] = sigmoid(a)

% Compute the output of the sigmoid function (i.e. f_a).
f_a = 1 ./ (1 + exp(-a));

% Use f_a to compute the gradient of the sigmoid function (i.e. grad_f_a).
grad_f_a = f_a .* (1 - f_a);

end