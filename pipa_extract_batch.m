% Extract features for each body part with pre-trained models
model = config.CNN_MODEL;
split_set = 'test';
if strcmp(split_set, 'train')
  split_index = data.train_idx;
  feat_name = config.MODEL_PART_TRAIN_FEAT;
else 
  split_index = data.test_idx;
  feat_name = config.MODEL_PART_TEST_FEAT;
end

% Generate file list of current split set
im_dir = sprintf('../data/pipa/%s', split_set);
for i = 1: numel(split_index)
  index = split_index(i);
  im_list{i} = sprintf('%s/%s_%s.jpg',im_dir, data.photoset_ids{index}, data.photo_ids{index});
end

% Extract bounding boxes for different body parts
if exist('bboxes.mat', 'file')
  load bboxes.mat
else 
  for i = 1: numel(config.MODEL_PART_WEIGHT)
    tic
    fprintf('%s\n', config.MODEL_PART_NAME{i});
    if ~exist('bboxes', 'var') || ~isfield(bboxes, config.MODEL_PART_NAME{i}) || config.OVERRIDE_EXTRACT
      cur_bboxes = zeros(numel(split_index), 4);
      parfor im_idx = 1: numel(split_index)
  %       fprintf('%s: Image %d/%d\n', config.MODEL_PART_NAME{i}, im_idx, numel(split_index));
        im = imread(im_list{im_idx});
        head_bbox = data.head_boxes(split_index(im_idx), :);
        cur_bboxes(im_idx, :) = pipa_gen_body_bbox(im, head_bbox, config.MODEL_PART_NAME{i});
      end
      bboxes.(config.MODEL_PART_NAME{i}) = cur_bboxes;
    end
    toc
  end
  clear cur_bboxes
end



% For each body part, extract features in batch mode
num_images = numel(split_index);
fc7_feature = single(zeros(4096, num_images, numel(config.MODEL_PART_WEIGHT)));
batch_size = 256;
caffe.set_mode_gpu();
caffe.set_device(0);
for i = 1: numel(config.MODEL_PART_WEIGHT)
  textprogressbar('Extracting batch features: ');
  if exist(feat_name{i}, 'file') ~= 2 || config.OVERRIDE_EXTRACT
    weights = config.MODEL_PART_WEIGHT{i};
    net = caffe.Net(model, weights, 'test'); % create net and load weights
    num_batches = ceil(num_images/batch_size);
    for bb = 1 : 2%num_batches
      textprogressbar(100 * bb/ num_batches);
      range = 1+batch_size*(bb-1):min(num_images,batch_size * bb);
      im_data = prepare_batch(im_list(range), bboxes.(config.MODEL_PART_NAME{i})(range, :), batch_size);
%       fprintf('Batch %d out of %d %.2f%% \n',bb,num_batches,bb/num_batches*100);
      res = net.forward({im_data});
      res = squeeze(res{1});
      fc7_feature(:, range, i) = res(:,mod(range-1,batch_size)+1);
    end
    textprogressbar('Done');
    caffe.reset_all();
  end
end
