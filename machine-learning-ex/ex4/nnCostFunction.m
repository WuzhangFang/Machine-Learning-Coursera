function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1

X = [ones(m, 1) X]; % add one, X is a row vector, dim: 5000x401
z_2 = X * Theta1'; % 5000x25
a_2 = sigmoid(z_2); % 5000x25
a_2 = [ones(m, 1) a_2]; % add one, a_2: 5000*26

z_3 =  a_2 * Theta2'; % 5000x10
a_3 = sigmoid(z_3); % 5000x10

% based on the digit, creat row vectors
%  y_mat = zeros(m, num_labels); % 5000*10
 y_mat = (1:num_labels) == y;
%  for i=1:m
%     y_mat(i,y(i)) = 1; 
%  end

% cost function for neural network
% for i=1:m
%     for k=1:num_labels
%     J = J - y_mat(i,k)*log(a_3(i,k))-(1-y_mat(i,k))*log(1-a_3(i,k));
%     end
% end
% vectorized version

J = sum(sum(-y_mat.*log(a_3)-(1-y_mat).*log(1-a_3)))/m;


% Part 2
% remove first column
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));

% % regularized part

% % Theta1
% for j=1:hidden_layer_size
%     for k=2:input_layer_size+1
%         J_r = J_r + Theta1(j,k)^2;
%     end
% end
% 
% % Theta2
% for j=1:num_labels
%     for k=2:hidden_layer_size+1
%         J_r = J_r + Theta2(j,k)^2;
%     end
% end

% vectorized version
J_r = (sum(sum(t1.^2)) + sum(sum(t2.^2))) * lambda / (2*m);

J = J + J_r;

% Part 3 
% Backpropagation

delta_3 = a_3 - y_mat; % 5000*10
% m is training example
for t=1:m
    z2 = [1 z_2(t,:)]'; % 26x1
    delta3 = delta_3(t,:)'; % 10x1
    delta2 = (Theta2')*(delta3).*sigmoidGradient(z2); %(26x10)*(10x1)=26x1
    delta2 = delta2(2:end); % 25x1
    a1 = X(t,:); %1x401
    a2 = a_2(t,:); % 1x26
    Theta1_grad = Theta1_grad + delta2*a1; % 25x401
    Theta2_grad = Theta2_grad + delta3*a2; % 10x26
end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Regularization, starting from the second column

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
