% Demonstration script for low-dimensional subspace-constrained 4D-MRI 
% reconstructions as described in our Radiotherapy and Oncology article.
%
% This code uses:
%   1) NUFFT functions from the Michigan Image Reconstruction toolbox
%   2) Wavelet class written by Miki Lustig for regularization
%
% Developer: Nikolai Mickevicius
% Created:   16 December, 2020

%% Parameters 

% do not change
global osfactor;
osfactor = 2;
mtrx = 256;
TR = 0.0046; % repetition time [sec]

% feel free to change
nbins = 10;        % number of respiratory phases 
K = 2;             % number of subspace coefficients
submtrx = 96;      % low resolution matrix size used for subspace basis function estimation
num_admm = 5;      % number of ADMM iterations to solve Equation 3
num_lsqr = 5;      % number of conjugate gradient iterations for Equation 5a
admm_rho = 1;      % rho for ADMM algorithm
lambda_wav = 5e-3; % wavelet regularization weight
lsqr_tol = 1e-5;   % tolerance for conj. grad. iterations

% put some params in structure used by vane_recon_ADMM()
par.rho = admm_rho;
par.admm_iters = num_admm;
par.wav_lambda = lambda_wav;
par.lsqr_tol = lsqr_tol;
par.lsqr_iters = num_lsqr;
parlr = par;

%% Load Data

% load 8-channel radial k-space data for a single slice of a fat-suppressed
% balanced TFE acquisition acquired on a 1.5T Elekta Unity. This file 
% contains the k-space data, k-space trajectory (scaled as needed for 
% Fessler's NUFFT), density compensation function, and the respiratory 
% motion amplitude at each spoke.
load('TFE_VANE_8ch_1slc.mat','data','ktraj','dcf','nav','lrmask');

%% Motion Averaged Reconstruction and Estimate Coil Sensitivity maps

[vol3d,smaps] = vane_mavg_recon(data, ktraj, dcf, 'espirit', []);


%% Sort Data into Bins

[rsdata,rstraj] = vane_reshuffleData(data, ktraj, nav, nbins, 'hybrid');


%% Reference Reconstruction (SENSE + Wavelet Regularization)

tRefStart = tic;

% Fourier transform operator
F = MCWNUFFT(rstraj, size(smaps,1), [0,0], sqrt(dcf));

% do the reconstruction
par.reg = 'wav';
reconRef = recon_admm(rsdata, smaps, F, par, []);

totalReferenceReconTime = toc(tRefStart);

%% Start Timer for Subspace Constrained Recon Here
tSubConstStart = tic;

%% Do Low Resolution SENSE Recon

% get low resolution data
i1 = size(data,1)/2 - submtrx;
i2 = i1 + 2*submtrx - 1;
lrdata = data(i1:i2,:,:,:,:);
lrtraj = ktraj(i1:i2,:,:,:);
lrtraj = (pi/max(abs(lrtraj(:)))).*lrtraj;
lrdcf = dcf(i1:i2);
lrxresFull = [];

% get low-resolution coil maps
[lrvol,lrsmaps] = vane_mavg_recon(lrdata, lrtraj, lrdcf, 'espirit', lrxresFull);

% make a mask (now loaded from .mat file)
% lrmask = zeros(size(lrvol));
% lrmask(abs(lrvol) > 0.07*max(abs(lrvol(:)))) = 1;
% lrmask = imfill(lrmask, ones(3,3), 'holes');

% sort low-res data into bins
[rslrdata,rslrtraj] = vane_reshuffleData(lrdata, lrtraj, nav, nbins, 'hybrid');

% low-res NUFFT operator 
lrF = MCWNUFFT(rslrtraj, size(lrsmaps,1), [0,0], sqrt(lrdcf));

% do the SENSE recon
parlr.reg = 'none';
parlr.admm_iters = 1;
parlr.lsqr_iters = 3;
lrvol5d = recon_admm(rslrdata, lrsmaps, lrF, parlr);

%% Estimate Subspace Basis Functions

evol = reshape(lrvol5d,[],nbins);
msk = reshape(lrmask,[],1);
evol = evol(find(msk),:);
evol = permute(evol,[2,1]);

[U,S,V] = svd(evol,'econ');
U = U(:,1:K);

%% Subspace Constrained Reconstruction 

% get a mask of where we have data
dataMask = permute(double(abs(rsdata(:,floor(size(rsdata,2)/2)+1,:,:,:)) > 0),[1,3,5,4,2]);

ntile = size(rsdata,1)/length(dcf);
dcf = repmat(dcf(:),[ntile,1]);

% make low-rank NUFFT operator
F = MCWLRNUFFT(rstraj, dataMask, size(smaps,1), U, sqrt(dcf));

% do the reconstruction
par.reg = 'wav';
reconSub = recon_admm(rsdata, smaps, F, par, U);

% total reconstruction time for subspace-constrained includes the
% low-resolution SENSE recon
totalSubspaceConstrainedReconTime = toc(tSubConstStart);

%% Display Results

reconRef = squeeze(reconRef);
winref = [0,0.7*max(abs(reconRef(:)))];
reconSub = squeeze(reconSub);
winsub = [0,0.7*max(abs(reconSub(:)))];

% display reconstruction times
fprintf('Total Reference Reconstruction Time: %5.2f seconds.\n', totalReferenceReconTime);
fprintf('Total Subspace Constrained Reconstruction Time: %5.2f seconds.\n', totalSubspaceConstrainedReconTime);

% display some images at the end of expiration
figure; 
subplot(1,2,1);
imagesc(abs(reconRef(:,:,1)),winref); 
axis image;
colormap gray;
axis off;
title('Reference (Expiration)');
subplot(1,2,2);
imagesc(abs(reconSub(:,:,1)),winsub); 
axis image;
colormap gray;
axis off;
title('Subspace Constrained (Expiration)');

figure; 
subplot(1,2,1);
imagesc(abs(reconRef(:,:,5)),winref); 
axis image;
colormap gray;
axis off;
title('Reference (Inspiration)');
subplot(1,2,2);
imagesc(abs(reconSub(:,:,5)),winsub); 
axis image;
colormap gray;
axis off;
title('Subspace Constrained (Inspiration)');










