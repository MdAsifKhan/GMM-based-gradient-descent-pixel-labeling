function [theta,new_wt,prob_new,difftemp,wt]=maingmmgrad_descent(im_1,k,skip)
%GMM based gradient descent algorithm for classification
%   Author: Mohd. Asif Khan
%   mail id: mak4086@gmail.com
%
%im_1:Input Image should be of double type
%     RGB image should be converted to gray scale
%k: Number of clusters
%skip: To skip alternate elements of image
%      for skip=1 algorithm will work on entire image,
%      skip=2 means it will take elements of rows and column at a gap of 1
%      and so on. for ex. 
%     im_1 =
%     0.9720    0.0344    0.4521    0.1649    0.3333
%     0.5427    0.3167    0.7958    0.9995    0.2813
%     0.9261    0.4774    0.7150    0.3101    0.6375
%     0.4026    0.2664    0.1956    0.4694    0.4843
%     0.1347    0.5432    0.6970    0.4305    0.4463
%     data =
% 
%     0.9720    0.4521    0.3333
%     0.9261    0.7150    0.6375
%     0.1347    0.6970    0.4463 

im_1=((im_1-min(min(im_1)))/(max(max(im_1))-min(min(im_1))));
data1=im_1;
[U,V]=size(data1);
tic
data(1:ceil(U/skip),1:ceil(V/skip))=data1(1:skip:U,1:skip:V);
[U,V]=size(data);
  %To convert  U X V data into  1 X U*V  
  xi=reshape(data',[1 U*V]);
  %To take into consideration boudary elments while computing spatial
  %relationship
  data=padarray(data,[3 3],'replicate');   
[theta,wt]=grad_descent_opt(im_1,data,xi,k,U,V);    
%To obtain probability of pixels for optimized value of parameters
      y1=gauss_dist_opt(xi,theta(:,1),theta(:,2));
      probb=bsxfun(@rdivide,wt.*y1,sum(wt.*y1));
      new_wt=reshape(max(wt)',[V U])';
     [prob_new,difftemp]=max(probb);
%To label the pixels into classes
        prob_new=reshape(prob_new',[V U])';
        difftemp=reshape(difftemp',[V U])';
 toc
end
 
%%
%%To find probability using Gaussian Model
function [ y ] = gauss_dist_opt(x,meu,sigma)
%GAUSS_DIST function for gaussian distribution
expterm=exp(-(bsxfun(@rdivide,((bsxfun(@minus,x,meu)).^2),(2.*sigma))));
y=bsxfun(@times,(1./(sqrt(2*pi.*sigma))),expterm);
end
%%
%%To Compute Mean of every pixel by taking 3 X 3 window around that pixel
 function [xim]=meanpix_opt(data,u,v)   %To calculate mean of every pixel value
   u=u+3;
   v=v+3;
     xim=(1/9)*sum(sum(data(u-1:u+1,v-1:v+1)));
 end
%%
%%Calculate weights for every pixel with respect to every class
function [wt,del,v1,tau]=weights_wt_opt(data,k,c,b,U,V)   
m_set=zeros(k,U*V);del=zeros(k,U*V);v1=zeros(k,U*V);tau=zeros(k,U*V);m_set=zeros(k,U*V);     
del=zeros(k,U*V);v1=zeros(k,U*V);tau=zeros(k,U*V);  
  i=1;
 for u=1:U
     for v=1:V
         for n=-1:1
             sum_1=0;sum_2=0;sum_4=0;
                for m=-1:1
                     xim1=meanpix_opt(data,u+n,v+m);
                     expterm=exp(-(xim1-c).^2./(2*(b.^2)));
                     sum_1=sum_1+expterm;
                     sum_2=sum_2+((xim1-c)./(2*(b.^2))).*expterm;
                     sum_4=sum_4+((xim1-c).^2./(2*(b.^3))).*expterm;
                end
                m_set(:,i)=m_set(:,i)+sum_1;     %Calculates mean value of the set of neighborhood weights
                del(:,i)=del(:,i)+sum_2;             % del (delta),v and tau are constant defined in algorithm used while optimizing c and b
                v1(:,i)=m_set(:,i);
                tau(:,i)=tau(:,i)+sum_4;   
          end
          m_set(:,i)=(1/9)*m_set(:,i);i=i+1; 
     end
  end    
      wt=bsxfun(@rdivide,m_set,sum(m_set));
end

%%
%%To Perform Gradient Descent Algorithm 
function [theta_new,wt,posterior_old]=grad_descent_opt(im_1,data,xi,k,U,V)
eta=10^-8;    %Learning Rate as error is of high order so it should be small number
%initialization done by kmeans as proposed in the paper
 [dim_1,dim_2]=size(im_1);
 zz=kmeans(reshape(im_1,[dim_1*dim_2 1]),k,'emptyaction','singleton','MaxIter',200);
 zzz=reshape(zz,[dim_1 dim_2]);
 xx=im_1(zzz==1);
 yy=im_1(zzz==2);
mu(1:2,1)=[mean(xx);mean(yy)]; 
sigma1(1:2,1)=[std(double(xx));std(double(yy))];
c=mu;b=sigma1;
theta_old=[mu sigma1 c b];
neg_likelihood_old=0;
err_old=0;
neg_likelihood_1=zeros(100,1);
for iter=1:200            %Gradient Descent algorithm to calculate optimum value of mue,sigma ,c and b(as defined in algorithm)
    c=theta_old(:,3);
    b=theta_old(:,4);
    [wt,del,v,tau]=weights_wt_opt(data,k,c,b,U,V);
     prob=gauss_dist_opt(xi,theta_old(:,1),theta_old(:,2));   %Probability for every pixel with respect to every class
     posterior_old=bsxfun(@rdivide,(wt.*prob),(sum(wt.*prob))); %Posterior Probability for every pixel with respect to every class
     
     err=-sum(sum(posterior_old.*(log(wt)-0.5*log(2*pi)...
            +bsxfun(@minus,-0.5*log(theta_old(:,2).^2),bsxfun(@rdivide,(bsxfun(@minus,xi,theta_old(:,1))).^2,(2*(theta_old(:,2).^2))))))); %Error function containing sum of error for every iteration
     
     del_sigma=-sum((posterior_old.*(bsxfun(@plus,-1./theta_old(:,2),bsxfun(@rdivide,bsxfun(@minus,xi,theta_old(:,1)).^2,theta_old(:,2).^3)))),2);%Updating derivative of sigma
     del_mu=-sum((bsxfun(@rdivide,(posterior_old.*(bsxfun(@minus,xi,theta_old(:,1)))),(theta_old(:,2).^2))),2);  %Updating derivative of mue
     del_c=-sum(posterior_old.*(del./v),2)+sum(sum(posterior_old.*(bsxfun(@rdivide,del,sum(v)))));   %Updating derivative of c
     del_b=-sum(posterior_old.*(tau./v),2)+sum(sum(posterior_old.*(bsxfun(@rdivide,tau,sum(v))))); %Updating derivative of b
     if isnan(sum(del_c))==1 | isnan(sum(del_b))==1 | isnan(sum(wt))==1
        del_c=del_c_old;
        del_b=del_b_old;
        wt=wt_old;
     end
     del_c_old=del_c;
     del_b_old=del_b;
     wt_old=wt;
     del_err=[del_mu del_sigma del_c del_b];     
     theta_new=theta_old-eta*del_err;     %Updating final value of theta 
     neg_likelihood=-sum(log(sum(wt.*prob)));
%       if abs(neg_likelihood-neg_likelihood_old)<10^-3;
      if (abs(err)-abs(err_old))<10^-3  | isnan(err)==1 | isnan(sum(sum(theta_new)))==1
          theta_new=theta_old;
          wt=wt_old;
       break;
      else    
           theta_old=theta_new;
           err_old=err;
           wt_old=wt;
%            neg_likelihood_old=neg_likelihood;
      end    

%       disp(neg_likelihood);
       neg_likelihood_1(iter)=neg_likelihood;
       err_1(iter)=err;
        disp(err_old);
end
%       figure,plot(neg_likelihood_1);
%       xlabel('Iteration');
%       ylabel('Observed Data Log-likelihood');
%       figure,plot(err_1);
%       xlabel('Iteration');
%       ylabel('Observed Error');

end