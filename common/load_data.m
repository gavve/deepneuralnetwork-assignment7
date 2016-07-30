% This function loads the MNIST database of handwritten digits.
function [T_test, T_training, X_test, X_training] = load_data(n_test, n_training)

X_test = loadMNISTImages('t10k-images');
X_test = reshape(X_test, 28, 28, []);

X_training = loadMNISTImages('train-images');
X_training = reshape(X_training, 28, 28, []);

T_test = loadMNISTLabels('t10k-labels');

T_test(T_test == 0) = 10;

T_test = full(sparse(T_test, 1 : length(T_test), 1));

T_training = loadMNISTLabels('train-labels');

T_training(T_training == 0) = 10;

T_training = full(sparse(T_training, 1 : length(T_training), 1));

if ~isinf(n_test)
    
    r      = randperm(10000);
    T_test = T_test(:, r(1 : n_test));
    X_test = X_test(:, :, r(1 : n_test));
    
end

if ~isinf(n_training)
    
    r          = randperm(60000);    
    T_training = T_training(:, r(1 : n_training));
    X_training = X_training(:, :, r(1 : n_training));
    
end

end

