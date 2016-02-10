% Extract features for each body part with pre-trained models
model = config.CNN_MODEL;
split_set = 'train';
if split_set == 'train'
  split_index = data.train_idx;
  feat_name = config.MODEL_PART_TRAIN_FEAT;
else 
  split_index = data.test_idx;
  feat_name = config.MODEL_PART_TEST_FEAT;
end

fc7_feature = single(zeros(4096, numel(split_index), numel(config.MODEL_PART_WEIGHT)));
for i = 1: numel(config.MODEL_PART_WEIGHT)
  if exist(config.MODEL_PART_FEAT{i}, 'file') ~= 2 || config.OVERRIDE_EXTRACT
    weights = config.MODEL_PART_WEIGHT{i};
    caffe.set_mode_gpu();
    caffe.set_device(0);
    net = caffe.Net(model, weights, 'test'); % create net and load weights
    for im_idx = 1: numel(split_index)
      fprintf('%s: Image %d/%d\n', config.MODEL_PART_NAME{i}, im_idx, numel(split_index));
      [im, bbox] = load_im(split_index(im_idx), split_set);
      body_part = crop_body(im, bbox, config.MODEL_PART_NAME{i});
      im_data = prepare_image(body_part);
      res = net.forward({im_data});
      fc7_feature(:, im_idx, i) = res{1};
    end
    cur_fc7 = fc7_feature(:, :, i);
    save(feat_name{i}, 'cur_fc7');
  end
end
caffe.reset_all();
