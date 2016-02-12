function [ images ] = prepare_batch(im_name, bboxes, batch_size )
%PREPARE_BATCH Summary of this function goes here
%   Detailed explanation goes here
d = load('../caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat');
IMAGE_MEAN = d.mean_data;

num_images = numel(im_name);
if nargin < 3
    batch_size = num_images;
end

IMAGE_DIM = 256;
CROPPED_DIM = 227;
indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
center = floor(indices(2) / 2)+1;

images = zeros(CROPPED_DIM,CROPPED_DIM,3,batch_size,'single');

parfor i=1:num_images
    % read file
    try
        im = imread(im_name{i});
        % crop the body part
        im = im(bboxes(i, 2): bboxes(i, 4), bboxes(i, 1): bboxes(i, 3), :);
        % resize to fixed input size
        im = single(im);
        im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
        % Transform GRAY to RGB
        if size(im,3) == 1
            im = cat(3,im,im,im);
        end
        % permute from RGB to BGR (IMAGE_MEAN is already BGR)
        im = im(:,:,[3 2 1]) - IMAGE_MEAN;
        % Crop the center of the image
        images(:,:,:,i) = permute(im(center:center+CROPPED_DIM-1,...
            center:center+CROPPED_DIM-1,:),[2 1 3]);
    catch
        warning('Problems with file', im_name{i});
    end

end

