% This function implements forward propagation. It computes and returns the
% outputs of the hidden and output units as well as the gradients of the
% hidden units (i.e. f_a_2, f_a_3 and grad_f_a_2).
function [f_a_2, f_a_3, grad_f_a_2] = forwardprop(args, b_1, b_2, W_1, W_2,
X)
% Compute the inputs of the hidden units (i.e. a_2).
%
% For each image and hidden unit pair (i.e. i and j, respectively), you
% should convolve X(:, :, i) with W_1(:, :, j) and add b_1(j) to obtain the
% corresponding a_2 (i.e. a_2(:, :, j, i)).
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
a_2 = zeros(args.imageDim - args.filterDim + 1, args.imageDim -
args.filterDim + 1, args.numFilters, size(X, 3));
for i = 1 : size(X, 3) % loop over images

 for j = 1 : args.numFilters % loop over hidden units

 % Write your code here
 % -----------------------------------------------------------------

 %flip = rot90(W_1, 2);
 a_2(:, :, j, i) = conv2(X(:,:,i), rot90(W_1(:,:,j), 2), 'valid') +
b_1(j);

 % -----------------------------------------------------------------

 end

end
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% Compute the outputs of the hidden units (i.e. f_a_2) and their gradients
% (i.e. grad_f_a_2).
%
% You do *not* have to modify the following.
[f_a_2, grad_f_a_2] = sigmoid(a_2);
% Compute the mean pooled outputs of the hidden units (i.e. f_a_2).
%
% For each image and hidden unit pair (i.e. i and j, respectively), you
% should mean pool the corresponding f_a_2 (i.e. f_a_2(:, :, j, i)). That
% is, you should average all nonoverlapping (by default) 2 by 2 blocks of
% f_a_2(:, :, j, i).
%
% TIP: You can use the im2col function to rearrange blocks of
% f_a_2(:, :, j, i) into columns, average the columns and use the col2im
% function to rearrange the columns into blocks.
%
% A nested for loop is provided for your convenience.
%
% Note that the mean pooled outputs of the hidden units should be stored in
% f_a_2. You might want the store them in a temporary variable during the
% loop and assign the temporary variable to f_a_2 after the loop.
%
% Write your code in the for loop:
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
temp = zeros(size(f_a_2, 1) / args.poolDim, size(f_a_2, 2) / args.poolDim,
size(f_a_2, 3), size(f_a_2, 4));
for i = 1 : size(X, 3) % loop over images

 for j = 1 : args.numFilters % loop over hidden units

 % Write your code here
 % -----------------------------------------------------------------

 %f_a_2(:, :, j, i)
 col = im2col(f_a_2(:, :, j, i),[2 2], 'distinct');
 m = mean(col);
 temp(:, :, j, i) = col2im(m, [1 1], [10 10]);

 % -----------------------------------------------------------------

 end

end
f_a_2 = temp;
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% Compute the inputs of the third layer units (i.e. a_3).
%
% You do *not* have to modify the following.
a_3 = bsxfun(@plus, W_2 * reshape(f_a_2, [], size(f_a_2, 4)), b_2);
% Compute the outputs of the output units (i.e. f_a_3).
%
% You do *not* have to modify the following.
f_a_3 = softmax(a_3);
end
