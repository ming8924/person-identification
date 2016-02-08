function models = svm_train_1vall(labels, features, opts)
  classes = unique(labels);
 	models = cell(1, numel(classes));
	for i = 1: numel(classes)
		 fprintf('Training SVM for class %d/%d\n', i, numel(classes));
		 models{i} = train(double(labels==classes(i)), features, opts);
		 models{i}.classes = classes;
     [a,b,c] = predict(double(labels==classes(i)), features, models{i});
	end
end
