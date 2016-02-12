clear
close all
addpath(genpath('../data/pipa'))
addpath('../caffe/matlab');
addpath external/liblinear-2.1/matlab
run('external/vlfeat/toolbox/vl_setup.m');
global data;
load('data.mat');

global config;

config.DATA_DIR = '../data/pipa';
config.TEST_DIR = sprintf('%s/test', config.DATA_DIR);
config.TRAIN_DIR = sprintf('%s/train', config.DATA_DIR);
config.MODEL_DIR = 'models';
config.CNN_MODEL = sprintf('%s/alexnet_extraction.prototxt', config.MODEL_DIR);
config.MODEL_PART_NAME = {'head', 'full_body', 'upper_body', 'scene', 'head_cacd', 'head_casia'};
config.MODEL_PART_WEIGHT = config.MODEL_PART_NAME;
config.FEAT_CACHE = 'feat_cache/batch_feat';
if exist(config.FEAT_CACHE, 'dir') ~= 7
	mkdir(config.FEAT_CACHE);
end
config.OVERRIDE_EXTRACT = true;
for i = 1: numel(config.MODEL_PART_WEIGHT)
  config.MODEL_PART_WEIGHT{i} = sprintf('%s/%s.caffemodel', config.MODEL_DIR, config.MODEL_PART_WEIGHT{i});
end
for i = 1: numel(config.MODEL_PART_WEIGHT)
  config.MODEL_PART_TRAIN_FEAT{i} = sprintf('%s/%s_train_feat.mat', config.FEAT_CACHE, config.MODEL_PART_NAME{i});
  config.MODEL_PART_TEST_FEAT{i} = sprintf('%s/%s_test_feat.mat', config.FEAT_CACHE, config.MODEL_PART_NAME{i});
end
