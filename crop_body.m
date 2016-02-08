function [ cropped_im ] = crop_body( im, bbox, part)
%CROP_BODY Summary of this function goes here
%   Return the body part specified by the input argument
[h, w, c] = size(im);
switch part
  case 'scene'
    cropped_im = im;
    x1 = 1;
    y1 = 1;
    x2 = w;
    y2 = h;
  case {'head', 'head_cacd', 'head_casia'}
    x1 = bbox(:, 1);
    y1 = bbox(:, 2);
    x2 = bbox(:, 3) + x1 - 1;
    y2 = bbox(:, 4) + y1 - 1;   
    x1 = max(1, x1);
    x2 = min(w, x2);
    y1 = max(1, y1);
    y2 = min(h, y2);
    cropped_im = im(y1: y2, x1: x2, :);
  case 'full_body'
    x1 = bbox(:, 1) - bbox(:, 3);
    x2 = bbox(:, 1) + 2 * bbox(:, 3) - 1;
    y1 = bbox(:, 2);
    y2 = bbox(:, 2) + 6 * bbox(:, 3) - 1;
    x1 = max(1, x1);
    x2 = min(w, x2);
    y1 = max(1, y1);
    y2 = min(h, y2);
    cropped_im = im(y1: y2, x1: x2, :);
  case 'upper_body'
    x1 = bbox(:, 1) - bbox(:, 3);
    x2 = bbox(:, 1) + 2 * bbox(:, 3) - 1;
    y1 = bbox(:, 2);
    y2 = bbox(:, 2) + 3 * bbox(:, 3) - 1;
    x1 = max(1, x1);
    x2 = min(w, x2);
    y1 = max(1, y1);
    y2 = min(h, y2);
    cropped_im = im(y1: y2, x1: x2, :);
end
if isempty(cropped_im)
  cropped_im = im;
end

% %% Show cropped bboxes
% in_bbox = [x1, y1, x2 - x1, y2 - y1];
% in_bbox = [in_bbox; bbox];
% showboxes(im, in_bbox);

end

