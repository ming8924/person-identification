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
feature{4} = cat(1, fc7_feature(:, :, 1), fc7_feature(:, :, 2), fc7_feature(:, :, 3), fc7_feature(:, :, 4), fc7_feature(:, :, 5), fc7_feature(:, :, 6));

% Prepare test set split ids
train_id = data.test_idx(data.test_split);
train_id_test = data.test_split;
all_id = 1: num_data;
test_id = data.test_idx(all_id(~ismember(all_id, data.test_split)));
test_id_test = all_id(~ismember(all_id, data.test_split));
train_label = data.identity_ids(train_id);
test_label = data.identity_ids(test_id);


for i = 1: numel(feature)
  fprintf('Querying the test image features\n');
  tic
  distance{i} = zeros(numel(train_id), numel(test_id));
%   for j = 1: numel(feature{i}(1, test_id_test))
%     fprintf('Calculating distance of %d\n', j);
%     % euclean distance
%     diff = bsxfun(@minus, feature{i}(:, train_id_test), feature{i}(:, test_id_test(j)));
%     distance{i}(:, j) = sum(diff.^2, 1);  
%   end
  distance{i} = normc(feature{i}(:, train_id_test))' * normc(feature{i}(:, test_id_test));
  toc
  [~, index{i}] = min(distance{i}, [], 1);
  label{i} = train_label(index{i});
end

for i = 1: numel(feature)
  for k = 1: 10
    predicted_label{i, k} = mode(label{i}(1:k, :))';
    accuracy(i, k) = (sum(predicted_label{i, k} == test_label))/numel(test_label);
  end
end