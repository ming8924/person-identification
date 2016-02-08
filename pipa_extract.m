% Extract features for each body part with pre-trained models
model = config.CNN_MODEL;
fc7_feature = single(zeros(4096, numel(data.test_idx), numel(config.MODEL_PART_WEIGHT)));
for i = 1: numel(config.MODEL_PART_WEIGHT)
  if exist(config.MODEL_PART_FEAT{i}, 'file') ~= 2 || config.OVERRIDE_EXTRACT
    weights = config.MODEL_PART_WEIGHT{i};
    caffe.set_mode_gpu();
    caffe.set_device(0);
    net = caffe.Net(model, weights, 'test'); % create net and load weights
    for im_idx = 1: numel(data.test_idx)
      fprintf('%s: Image %d/%d\n', config.MODEL_PART_NAME{i}, im_idx, numel(data.test_idx));
      [im, bbox] = load_im(data.test_idx(im_idx), 'test');
      body_part = crop_body(im, bbox, config.MODEL_PART_NAME{i});
      im_data = prepare_image(body_part);
      res = net.forward({im_data});
      fc7_feature(:, im_idx, i) = res{1};
    end
    cur_fc7 = fc7_feature(:, :, i);
    feat_name = config.MODEL_PART_FEAT{i};
    save(feat_name, 'cur_fc7');
  end
end
caffe.reset_all();
