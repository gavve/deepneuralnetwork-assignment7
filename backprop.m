% This function implements back propagation. It computes and returns the
% gradients of the cross entropy function with respect to the biases and
% weights (i.e. grad_E_b_1, grad_E_b_2, grad_E_W_1 and grad_E_W_2).
function [grad_E_b_1, grad_E_b_2, grad_E_W_1, grad_E_W_2] = backprop(args,
f_a_2, f_a_3, grad_f_a_2, T, W_2, X)
% Compute the deltas of the output units (i.e. delta_3).
%
% You do *not* have to change the following.
delta_3 = (f_a_3 - T) / size(X, 3);
% Compute the gradients of the error function with respect to the biased
% and weights in the output layer (i.e. grad_E_b_2 and grad_E_w_2).
%
% You do *not* have to change the following.
grad_E_b_2 = sum(delta_3, 2);
grad_E_W_2 = delta_3 * reshape(f_a_2, [], size(X, 3))';
% Compute the deltas of the hidden units (i.e. delta_2).
%
% You do *not* have to change the following.
delta_2 = W_2' * delta_3;
delta_2 = reshape(delta_2, (args.imageDim - args.filterDim + 1) /
args.poolDim, (args.imageDim - args.filterDim + 1) / args.poolDim,
args.numFilters, []);
for i = size(X, 3) : -1 : 1

 for j = args.numFilters : -1 : 1

 temp(:, :, j, i) = (1 / args.poolDim ^ 2) * kron(delta_2(:, :, j,
i), ones(args.poolDim));

 end

end
delta_2 = temp;
delta_2 = delta_2 .* grad_f_a_2;
% Compute the gradients of the cross entropy function with respect to the
% biases and weights in the hidden layer (i.e. grad_E_b_1 and grad_E_W_1).
%
% For each hidden unit (i.e. j), you should sum delta_2(:, :, j, :) over
% everything but j to obtain the corresponding grad_E_b_1 (i.e.
% grad_E_b_1(j)).
%
% Recall that in the case of a fully-connected hidden layer, grad_E_W_1 is
% given by delta_2 * X'. However, in the case of a convolutional hidden
% layer, grad_E_W_1 is given by convolution of delta_2 and X. For each
% image and hidden unit pair (i.e. i and j, respectively), you should
% convolve X(:, :, i) with delta_2(:, :, j, i) to obtain the corresponding
% grad_E_W_1 (i.e. grad_E_W_1(:, :, j, i)). You should then sum
% grad_E_W_1(:, :, j, i) over i.
%
% TIP: You can use the conv2 function for convolution. If you use the conv2
% function, you should use the 'valid' shape parameter. Note that the conv2
% function flips its second input horizontally and vertically, which is
% something that you do not want and should prevent by flipping the second
% input of the conv2 function horizontally and vertically before using
% it. For example, to flip a matrix A horizontally and vertically, you can
% use rot90(A, 2).
%
% A nested for loop is provided for your convenience.
%
% Write your code in the for loop:
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
grad_E_b_1 = zeros(args.numFilters, 1);
grad_E_W_1 = zeros(args.imageDim - size(delta_2, 1) + 1, args.imageDim -
size(delta_2, 2) + 1, args.numFilters);
for i = 1 : size(X, 3) % loop over images

 for j = 1 : args.numFilters % loop over hidden units

 % Write your code here
 % -----------------------------------------------------------------
 grad_E_b_1(j) = grad_E_b_1(j) + sum(sum(delta_2(:, :, j, i)));

 grad_E_W_1(:, :, j) = grad_E_W_1(:, :, j) + conv2(X(:, :, i),
rot90(delta_2(:, :, j, i), 2), 'valid');
 %grad_E_W_1(i) = sum(grad_E_W_1(:, :, j, i));

 % -----------------------------------------------------------------

 end
end
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
end