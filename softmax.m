% This function implements the softmax function. It computes and returns
% the output of the softmax function. Unlike the sigmoid function, the
% gradient of the softmax function is not required.
function f_a = softmax(a)

% TIP: The softmax function requires exponentiating a. Exponentiating large
% numbers might result in an overflow. You can prevent an overflow by
% subtracting the maximum of a(:, i) from a(:, i) for all i before
% exponentiating a. You do *not* have to modify the following.
a = bsxfun(@minus, a, max(a));

% Compute the output of the softmax function (i.e. f_a).
% -------------------------------------------------------------------------
for i=1:size(a, 2)
 f_a(:,i) = exp(a(:,i))./sum(exp(a(:,i)));
end


% -------------------------------------------------------------------------
% -------------------------------------------------------------------------

end

