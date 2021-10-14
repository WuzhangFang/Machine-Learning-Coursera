function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
% theta1: 25x401, theta2: 10x26


X = [ones(m, 1) X]; % add one, X is a row vector, dim: 5000x401
z_2 = X * Theta1'; % 5000x25
a_2 = sigmoid(z_2); 
a_2 = [ones(m, 1) a_2]; % add one, a_2: 5000*26

z_3 =  a_2 * Theta2'; % 5000x10
a_3 = sigmoid(z_3);

[M,p]= max(a_3, [], 2); %maximum index is also the predicition digit
%p: 5000x1



% =========================================================================


end
