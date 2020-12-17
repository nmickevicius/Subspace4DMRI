function csm = estimateCoilSensitivitiesESPIRiT(cimgs,smapsRes)

if nargin < 2
    smapsRes = [];
end
ks = fftshift(fft(fftshift(cimgs,1),[],1),1);
ks = fftshift(fft(fftshift(ks,2),[],2),2);

% zero pad the images
if ~isempty(smapsRes)
    npad = (smapsRes - size(ks,1))/2;
    ks = padarray(ks,[npad, npad, 0],'both');
end

DATA = ks;

[sx,sy,Nc] = size(DATA);
ncalib = 16; % use 24 calibration lines to compute compression
ksize = [4,4]; % kernel size

if Nc == 1
    csm = ones(sx,sy);
    return;
end

% Threshold for picking singular vercors of the calibration matrix
% (relative to largest singlular value.

eigThresh_1 = 0.02;

% threshold of eigen vector decomposition in image space.
eigThresh_2 = 0.95;

% crop a calibration area
%calib = crop(DATA,[ncalib,ncalib,Nc]);
ix1 = floor(sx/2 - ncalib/2);
ix2 = ix1 + ncalib - 1;
iy1 = floor(sy/2 - ncalib/2);
iy2 = iy1 + ncalib - 1;
calib = DATA(ix1:ix2,iy1:iy2,:);

[k,S] = dat2Kernel(calib,ksize);
idx = max(find(S >= S(1)*eigThresh_1));

[M,W] = kernelEig(k(:,:,:,1:idx),[sx,sy]);

% crop sensitivity maps
maps = M(:,:,:,end);%.*repmat(W(:,:,end)>eigThresh_2,[1,1,Nc]);

csm = maps;
