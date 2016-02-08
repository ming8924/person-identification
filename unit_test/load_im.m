cd ..
for i = 1: 20
  index = data.test_idx(i);
  im_dir = '../data/pipa/test';
  im_name = sprintf('%s/%s_%s.jpg',test_dir, data.photoset_ids{index}, data.photo_ids{index});
  bbox = data.head_boxes(index, :);
  im = imread(im_name);
  showboxes(im, bbox);
end
