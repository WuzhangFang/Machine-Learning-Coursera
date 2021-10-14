function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = X*theta;
Jerrors = (h-y).^2;
J = sum(Jerrors)/(2*m);

% theta_0 is not regularized
J_regularized = (lambda/(2*m))*sum(theta.^2) - (lambda/(2*m))*theta(1)^2; 
J = J + J_regularized;

% gradient
original_grad = X'*(h-y)/m;

grad = original_grad + (lambda/m)*theta;

grad(1)= original_grad(1); % theta_0 is not regularized


% =========================================================================

grad = grad(:);

end
