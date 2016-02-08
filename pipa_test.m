% Load extracted features
num_data = numel(data.test_idx);
fc7_feature = zeros(4096, num_data, numel(config.MODEL_PART_WEIGHT));
for i = 1: numel(config.MODEL_PART_WEIGHT)
  cur_model = sprintf('%s/%s_feat.mat', config.FEAT_CACHE, config.MODEL_PART_NAME{i});
  load(cur_model);
  fc7_feature(:, :, i) = cur_fc7;
end

% Choose feature concatenation 
feature1 = cat(1, fc7_feature(:, :, 4));
feature2 = cat(1, fc7_feature(:, :, 1), fc7_feature(:, :, 2), fc7_feature(:, :, 3));
feature3 = cat(1, fc7_feature(:, :, 1), fc7_feature(:, :, 2), fc7_feature(:, :, 3), fc7_feature(:, :, 4));
% feature1 = cat(1, fc7_feature(:, :, 1), fc7_feature(:, :, 2), fc7_feature(:, :, 3), fc7_feature(:, :, 4), fc7_feature(:, :, 5), fc7_feature(:, :, 6));
% feature1 = cat(1, fc7_feature(:, :, 1), fc7_feature(:, :, 2), fc7_feature(:, :, 3), fc7_feature(:, :, 4), fc7_feature(:, :, 6));


% Prepare test set split ids
train_id = data.test_idx(data.test_split);
train_id_test = data.test_split;
all_id = 1: num_data;
test_id = data.test_idx(all_id(~ismember(all_id, data.test_split)));
test_id_test = all_id(~ismember(all_id, data.test_split));
train_label = data.identity_ids(train_id);
test_label = data.identity_ids(test_id);

% Train linear SVM model
tic
model1 = train(train_label, sparse(feature1(:, train_id_test, 1))', '-B 1 -c 1 -q');
[predicted_label1, accuracy1, dv] = predict(test_label, sparse(feature1(:, test_id_test, 1))', model1);
fprintf('Head model accuracy: %f\n', accuracy1(1));
toc

tic
model2 = train(train_label, sparse(feature2(:, train_id_test, 1))', '-B 1 -c 1 -q');
[predicted_label2, accuracy2, dv] = predict(test_label, sparse(feature2(:, test_id_test, 1))', model2);
fprintf('Head + Full body + Upper body model accuracy: %f\n', accuracy2(1));
toc

tic
model3 = train(train_label, sparse(feature3(:, train_id_test, 1))', '-B 1 -c 1 -q');
[predicted_label13, accuracy3, dv] = predict(test_label, sparse(feature3(:, test_id_test, 1))', model3);
fprintf('Head + Full body + Upper body + Scene model accuracy: %f\n', accuracy3(1));
toc
