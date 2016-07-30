% This function implements the cross entropy function. It computes and
% returns the output of the cross entropy function as well as its gradient
% with respect to the parameters (i.e. E_theta and grad_E_theta). You do
% *not* have to modify this function.
function [E_theta, grad_E_theta] = cross_entropy(args, theta, X, T)

% Unvectorize the biases and weights.
[W_1, W_2, b_1, b_2] = cnnParamsToStack(theta, args);

% Do forward propagation.
[f_a_2, f_a_3, grad_f_a_2] = forwardprop(args, b_1, b_2, W_1, W_2, X);

% Compute the output of the cross entropy function.
E_theta = f_a_3 .* T;
E_theta = -mean(log(E_theta(E_theta ~= 0)));

% Compute the training (mini-batch) accuracy.
[~, I]     = max(T);
[~, I_hat] = max(f_a_3);

accuracy = 100 * mean(I == I_hat);

disp('---o---');
disp(['Training (mini-batch) accuracy = ' num2str(accuracy) '%']);

% Do backpropagation.
[grad_E_b_1, grad_E_b_2, grad_E_W_1, grad_E_W_2] = backprop(args, f_a_2, f_a_3, grad_f_a_2, T, W_2, X);

% Vectorize the weights and biases.
grad_E_theta = cat(1, grad_E_W_1(:), grad_E_W_2(:), grad_E_b_1(:), grad_E_b_2(:));

end

