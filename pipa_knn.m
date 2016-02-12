% Load extracted features
num_data = numel(data.test_idx);
fc7_feature = zeros(4096, num_data, numel(config.MODEL_PART_WEIGHT));
for i = 1: numel(config.MODEL_PART_WEIGHT)
  cur_model = sprintf('%s/%s_test_feat.mat', config.FEAT_CACHE, config.MODEL_PART_NAME{i});
  load(cur_model);
  fc7_feature(:, :, i) = feat;
end

% Choose feature concatenation 
feature{1} = cat(1, fc7_feature(:, :, 1));
feature{2} = cat(1, fc7_feature(:, :, 1), fc7_feature(:, :, 2), fc7_feature(:, :, 3));
feature{3} = cat(1, fc7_feature(:, :, 1), fc7_feature(:, :, 2), fc7_feature(:, :, 3), fc7_feature(:, :, 4));
feature{4} = cat(1, fc7_feature(:, :, 1), fc7_feature(:, :, 2), fc7_feature(:, :, 3), fc7_feature(:, :, 4), fc7_feature(:, :, 5), fc7_feature(:, :, 6));
% feature1 = cat(1, fc7_feature(:, :, 1), fc7_feature(:, :, 2), fc7_feature(:, :, 3), fc7_feature(:, :, 4), fc7_feature(:, :, 6));

% % Prepare test set split ids
% train_id = data.test_idx(data.test_split);
% train_id_test = data.test_split;
% all_id = 1: num_data;
% test_id = data.test_idx(all_id(~ismember(all_id, data.test_split)));
% test_id_test = all_id(~ismember(all_id, data.test_split));
% train_label = data.identity_ids(train_id);
% test_label = data.identity_ids(test_id);
% 
% im_id = data.test_idx;
im_label = data.identity_ids(data.test_idx);
index = cell(numel(feature), 1);
distance = cell(numel(feature), 1);
kdtree = cell(numel(feature), 1);
label = cell(numel(feature), 1);
for i = 1: numel(feature)
  fprintf('Building kdtree for feature %d\n', i);
  tic
  kdtree{i} = vl_kdtreebuild(feature{i});
  toc
  fprintf('Querying the test image features\n');
  tic
  [index{i}, distance{i}] = vl_kdtreequery(kdtree{i}, feature{i}, feature{i}, 'NumNeighbors', 10);

  toc
  label{i} = im_label(index{i}(1:10, :));
end
load knn_label

for i = 1: numel(feature)
  for k = 1: 10
    predicted_label{i, k} = mode(label{i}(1:k, :))';
    accuracy(i, k) = (sum(predicted_label{i, k} == im_label))/numel(im_label);
  end
end