function [prediction, accuracy, decv] = svm_predict_1vall(labels, features, models, opts)
  if nargin == 3
    opts = '';
  end
  classes = models{1}.classes;
	decvs = zeros(numel(classes), numel(labels));
  textprogressbar('Testing SVM: ');
	for i = 1: numel(classes)
     textprogressbar(100 * i/numel(classes));
		 [pred, acc, decvs(i, :)] = predict(double(labels==classes(i)), features, models{i}, opts);
     recall = sum(pred)/sum(double(labels==classes(i)));
	end
	[decv, I] = max(-decvs, [], 1);
	prediction = classes(I);
	accuracy = sum(prediction==labels)/numel(labels);
end
