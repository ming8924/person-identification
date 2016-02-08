model = '../caffe/models/bvlc_reference_caffenet/deploy.prototxt';
weights = '../caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';
caffe.set_mode_gpu();
caffe.set_device(0);
net = caffe.Net(model, weights, 'test'); % create net and load weights
