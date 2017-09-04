clear all; close all;

%% extracting data
% tic;
% addpath('hw3data');
% addpath('hw3data/training set');
% [trainImgs,trainLabels] = readMNIST('train-images-idx3-ubyte','train-labels-idx1-ubyte',20000,0);
% addpath('hw3data/test set');
% [testImgs,testLabels] = readMNIST('t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte',10000,0);
% toc;
% 
% save('hw3data','trainImgs','trainLabels','testImgs','testLabels');
load('hw3data.mat');
%% initialization
c = 10;
d = size(trainImgs,2); %d = 784
N = size(trainImgs,1); %N = 20000
N_test = size(testImgs,1); %N_test = 10000

X = trainImgs'; %784x20000 
X_test = testImgs'; %784x10000

Y = -ones(c,N);
for n = 1:N
    Y(trainLabels(n)+1,n) = 1;
end
Y_test = -ones(c,N_test);
for n = 1:N_test
    Y_test(testLabels(n)+1,n) = 1;
end

T = 250;
g = zeros(c,N); %g(x): 10x20000
g_test = zeros(c,N_test); %g(x): 10x10000
weights = zeros(10,N); %initail weights: 10x20000
W = zeros(d,102); %initial sum of error weights: 784x102
Alf = ones(c,2,T); %initial Alpha: 10x2x250
wt = zeros(c,T); %initial step size: 10x250
err_train_binary = zeros(c,T); err_train = zeros(1,T);
err_test_binary = zeros(c,T); err_test = zeros(1,T);
largestWeight_id = zeros(c,T); %largest weight indexes: 10x250
gammas = zeros(c,N,5); %margin: 10x20000x250
gamma_count = 0;
%% Adaboost
tic;

for it = 1:T
    it   
    for k = 1:c
        gk = g(k,:);
        gk_test = g_test(k,:);
        wl = it;
        jBest = Alf(k,1,wl);
        uBest = Alf(k,2,wl);
        xj = X(jBest,:); %1x20000
        xj_test = X_test(jBest,:); %1x10000
        
        %update the learned function
        if uBest <= 51
            n = uBest - 1;
            t = n/50;
            gk = gk + wt(k,wl)*u(xj,t);
            gk_test = gk_test + wt(k,wl)*u(xj_test,t);
        else
            n = uBest - 52;
            t = n/50;
            gk = gk + wt(k,wl)*(-u(xj,t));
            gk_test = gk_test + wt(k,wl)*(-u(xj_test,t));
        end
        g(k,:) = gk; %1x20000
        g_test(k,:) = gk_test; %1x10000
    
        %compute the weights
        yk = Y(k,:);
        wk = exp(-yk.*gk); %1x20000
        weights(k,:) = wk;
        
        %compute the negative gradient
        for j = 1:d
            xj = X(j,:);
            for n = 0:50
                t = n/50;

                ux = u(xj,t); %1x20000
                ydiffu_id = yk~=ux; %find i | yi!=u(xi)
                ydiffutwin_id = yk~=-ux;
                
                W(j,n+1) = sum(wk(ydiffu_id));
                W(j,n+52) = sum(wk(ydiffutwin_id));
            end
        end
        minW = min(W(:));
        [jBest,uBest] = find(W==minW,1);
        Alf(k,:,it+1) = [jBest,uBest];
        
        %compute the step size
        ep = W(jBest,uBest)/sum(wk);
        wt(k,it+1) = 1/2*log((1-ep)/ep);
    end
    
    %compute errors
    binaryPred_train = double(g>=0); 
    binaryPred_train(~binaryPred_train) = -1;
    binaryPred_test = double(g_test>=0); 
    binaryPred_test(~binaryPred_test) = -1;
    err_train_binary(:,it) = sum(binaryPred_train~=Y,2)/N;
    err_test_binary(:,it) = sum(binaryPred_test~=Y_test,2)/N_test;
    
    [gx,id] = max(g); %1x20000
    pred = id - 1;
    err_train(it) = sum(pred~=trainLabels')/N;
    [~,id_test] = max(g_test); %1x20000
    pred_test = id_test - 1;
    err_test(it) = sum(pred_test~=testLabels')/N_test;
    
    %store the index of the example of largest weight
    [~,largestWeight_id(:,it)] = max(weights,[],2);
    
    %store the margin of each example at iteration {5,10,50,100,250}
    if it==5 || it==10 || it==50 || it==100 || it==250
        gamma_count = gamma_count + 1;
        gamma = Y.*g;
        gammas(:,:,gamma_count) = gamma; %10x20000
    end
        
toc;
end

%the test error of the final classifier
% [~,id_test] = max(g_test); %1x20000
% pred_test = id_test - 1;
% err_test_final = sum(pred_test~=testLabels')/N;
err_test_final = err_test(end)

%% store results (the entire workspace)
save('Adaboost250');

%% play music when finished
[y,f]=audioread('Epilogue.mp3');
player = audioplayer(y,f);
play(player);
%stop(player);

%% plot 1 (all together)
figure(250)
for p = 1:c
    subplot(5,2,p);
    plot(1:it,err_train_binary(p,1:it),'r'); hold on;
    plot(1:it,err_test_binary(p,1:it),'b'); hold off;
    axis([1 it 0 1]);
    xlabel('#Iteration t'); ylabel('Error');
    legend('Train','Test');
    title(['Adaboost Train and Test Errors vs. #iteration for the Binary Classifier ' num2str(p)]);
end

%% a) plot 2 (split to 5)
for p = 1:c/2
    figure(510+p);
    subplot(1,2,1);
    plot(1:it,err_train_binary(2*p-1,:),'r'); hold on;
    plot(1:it,err_test_binary(2*p-1,:),'b'); hold off;
    axis([1 it 0 1]);
    xlabel('#Iteration t'); ylabel('Error');
    legend('Train','Test');
    title(['Adaboost Train and Test Errors vs. #iteration for the Binary Classifier ' num2str(2*p-1)]);
    
    subplot(1,2,2);
    plot(1:it,err_train_binary(2*p,:),'r'); hold on;
    plot(1:it,err_test_binary(2*p,:),'b'); hold off;
    axis([1 it 0 1]);
    xlabel('#Iteration t'); ylabel('Error');
    legend('Train','Test');
    title(['Adaboost Train and Test Errors vs. #iteration for the Binary Classifier ' num2str(2*p)]);
end

%% plot 3 (the final classifier)
figure(51);
plot(1:it,err_train,'r'); hold on;
plot(1:it,err_test,'b'); hold off;
axis([1 it 0 1]);
xlabel('#Iteration t'); ylabel('Error');
legend('Train','Test');
title('Adaboost Train and Test Errors vs. #iteration for the Final Classifier');

%% b) boosting encourages large margins
iter = [5,10,50,100,250];
for k = 1:10
    figure(5200+k);
    for p = 1:5
        subplot(5,1,p);
        cdfplot(gammas(k,:,p));
        xlim([-5,20]);
        xlabel('a'); ylabel('F(a) = P(\gamma \leq a)');
        title(['CDF of the Margins of Training Examples after ' num2str(iter(p)) ' Iterations for Binary Classifier ' num2str(k)]);
    end
end

%% c) the weighting mechanism makes boosting focus on the hard examples
for p = 1:c
    figure(5300+p);
    %subplot(1,2,1);
    plot(1:it,largestWeight_id(p,:),'b.');
    axis([1 it 1 N]);
    xlabel('#Iteration t'); ylabel('Index of the Example of Largest Weight');
    title(['Index of the Example of Largest Weight vs. #iteration for Binary Classifier ' num2str(p)]);
    
    saveas(gcf, ['530' num2str(p) '.png']);
    
%     subplot(1,2,2);    
%     top_id = largestWeight_id(p,:);
%     fq = arrayfun(@(i)sum(top_id==top_id(i)), 1:it);
%     fq_top = sort(unique(fq),'descend');
%     top_id(fq<fq_top(3)) = -10000; %make them not show in plot
%     plot(1:it,top_id,'r.');
%     axis([1 it 1 N]);
%     xlabel('#Iteration t'); ylabel('Index of the "Heaviest" Examples');
%     title(['Index of the Three "Heaviest" Examples vs. #iteration for Binary Classifier ' num2str(p)]);
end

%% c) Updated
for p = 1:c
%     figure;
%   
%     plot(1:it, largestWeight_id(p,:),'b.');
%     axis([1 it 1 N]); 
%     xlabel('iteration'); ylabel('index of the largest weight example');
%     saveas(gcf, ['c-1' num2str(p) '.png'])
%     title(['class ' num2str(p)]);

    figure(p);
    top_id = largestWeight(p,:);
    
    fq = arrayfun(@(i)sum(top_id==top_id(i)), 1:it);
    [C,ia,ic] = unique(top_id) ;
	C = top_id(ia);
    f = fq(ia);
    [~,ind] = sort(f,'descend');
    idx =1;
    C(ind(1:3))
    s = sum(f>=f(ind(3)));
    for idx=1:s
        subplot(1, s ,idx);
        imshow(reshape(trainImgs(C(ind(idx)),:) ,28,28)');
    end
    saveas(gcf, ['c-2' num2str(p) '.png']);
    %title(['class ' num2str(p)]);
    
%     plot(1:it,top_id,'r.');
%     axis([1 it 1 N]);
%     xlabel('iteration'); ylabel('index of the three heaviest examples');
%     title(['class ' num2str(p)]);
% 	saveas(gcf, ['c-' num2str(p) '.jpg'])
end

%% d)
a10 = 128*ones(c,d);
figure(54);
for k = 1:c
    for wl = 2:T+1
        jBest = Alf(k,1,wl);
        uBest = Alf(k,2,wl);
        if uBest <= 51 %regular weak learner
            a10(k,jBest) = 255;
        else %twin weak learner
            a10(k,jBest) = 0;
        end
    end
    ak = reshape(a10(k,:),28,28)';
    subplot(2,5,k);
    imshow(uint8(ak));
    title(['Array $$\textbf{a}$$ for the Binary Classifier ' num2str(k)],'interpreter','latex');
end

%% retrieve the data
load('Adaboost250_Final');