clc
clear all
close all
k=2;
skip=1;
for i=1:200
    im_1=double(imread(strcat('C:\Users\asifk\Documents\MATLAB\Images_1\singleauth-',num2str(i),'.tif')));
    if size(im_1,1)>500
        im_1=im_1';
    end
    [auththeta(:,:,i),authnew_wt(:,:,i),authprob_new(:,:,i),authdifftemp(:,:,i),authwt(:,:,i)]=maingmmgrad_descent(im_1,k,skip);
    im_2=double(imread(strcat('C:\Users\asifk\Documents\MATLAB\Images_1\doubleforged-',num2str(i),'.tif')));
    if size(im_2,1)>500
        im_2=im_2';
    end
    [forgedtheta(:,:,i),forgednew_wt(:,:,i),forgedprob_new(:,:,i),forgeddifftemp(:,:,i),forgedwt(:,:,i)]=maingmmgrad_descent(im_2,k,skip);
end
save('Data_200singleauth_200doubleforged');