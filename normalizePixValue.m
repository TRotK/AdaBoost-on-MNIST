function [ imgs_normed ] = normalizePixValue( imgs )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    imgs_normed = imgs/256;
    %imgs_normed = zscore(imgs);
    
end
