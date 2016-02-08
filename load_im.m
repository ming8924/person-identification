function [im, bbox] = load_im(index, split_set)
% return an image specified by the index and split set
% Note: index and split_set are actually redundant
  global data;
  im_dir = sprintf('../data/pipa/%s', split_set);
  im_name = sprintf('%s/%s_%s.jpg',im_dir, data.photoset_ids{index}, data.photo_ids{index});
  bbox = data.head_boxes(index, :);
  im = imread(im_name);
end
