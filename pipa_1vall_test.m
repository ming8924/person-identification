% Load extracted features
num_data = numel(data.test_idx);
fc7_feature = zeros(4096, num_data, numel(config.MODEL_PART_WEIGHT));
for i = 1: numel(config.MODEL_PART_WEIGHT)
  cur_model = sprintf('%s/%s_test_feat.mat', config.FEAT_CACHE, config.MODEL_PART_NAME{i});
  load(cur_model);
  fc7_feature(:, :, i) = cur_fc7;
end

% Choose feature concatenation 
feature{1} = cat(1, fc7_feature(:, :, 4));
feature{2} = cat(1, fc7_feature(:, :, 1), fc7_feature(:, :, 2), fc7_feature(:, :, 3));
feature{3} = cat(1, fc7_feature(:, :, 1), fc7_feature(:, :, 2), fc7_feature(:, :, 3), fc7_feature(:, :, 4));
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
for i = 1: numel(feature)
  tic
  model{i} = svm_train_1vall(train_label, sparse(feature{i}(:, train_id_test, 1))', '-B 1 -c 1 -q');
  [predicted_label{i}, accuracy{i}, dv] = svm_predict_1vall(test_label, sparse(feature{i}(:, test_id_test, 1))', model{i});
  fprintf('Head model accuracy: %f\n', accuracy{i}(1));
  toc
end
