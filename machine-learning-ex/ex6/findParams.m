cParams=[0.01 0.03 0.1 0.3 1 3 10 30];
sParams=[0.01 0.03 0.1 0.3 1 3 10 30];
errors=zeros(length(cParams),length(sParams));

for i=1:length(cParams)
   for j=1:length(sParams)
       model = svmTrain(X, y, cParams(i), @(x1, x2)gaussianKernel(x1, x2, sParams(j))); % every data point should be included
       predictions = svmPredict(model,Xval);
       errors(i,j)=mean(double(predictions~=yval));
   end
end

error_min = min(min(errors));
[row,col]= find(errors==error_min);
fprintf("C=%f, sigma=%f\n", cParams(row), sParams(col));
