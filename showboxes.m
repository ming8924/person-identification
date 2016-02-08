function showboxes(im, boxes)

image(im);
axis image;
axis off;
set(gcf, 'Color', 'white');

if ~isempty(boxes)
  for i = 1: size(boxes, 1)
    x1 = boxes(i, 1);
    y1 = boxes(i, 2);
    w = boxes(i, 3);
    h = boxes(i, 4);
    if w > 0 && h > 0
      rectangle('Position', [x1, y1, w, h],...
      'EdgeColor','r', 'LineWidth', 3);
    end
  end
end
