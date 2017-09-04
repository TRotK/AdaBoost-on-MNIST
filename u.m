function [ u ] = u( x,t )
%weak learner
%   Detailed explanation goes here
    u = -ones(1,length(x));
    u(x>=t) = 1;
    
end

