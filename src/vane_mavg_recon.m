function [vol3d,smaps] = vane_mavg_recon(data, traj, dcf, csmMethod, xresFull, logFile, smapsRes, useIFFTC, shift)

if nargin < 5
    xresFull = [];
end

if nargin < 6
    logFile = [];
end

if nargin < 7
    smapsRes = [];
end

if nargin < 8 || isempty(useIFFTC)
    useIFFTC = false;
end

if nargin < 9 || isempty(shift)
    shift = [0,0];
end

global osfactor;

% get dimensions
[xtot, nspokes, npar, nechoes, ncoils] = size(data);

if ~isempty(xresFull)
    xres = xresFull;
else
    xres = xtot / osfactor;
end

% IFT along partition direction
if useIFFTC
    data = ifftc(data,3);
else
    data = ifft(ifftshift(data,3),[],3);
end

% trajectory has shape [length(kr), 2, nspokes, nechoes]
% reshape trajectory to [length(kr)*nspokes, nechoes, 2]
traj = permute(traj,[1,3,4,2]);
traj = reshape(traj,[],nechoes,2);

ktraj = cell(nechoes,1);
for e = 1:nechoes
    ktraj{e} = squeeze(traj(:,e,:));
end

% make NUFFT operator
FT = MCWNUFFT(ktraj, xres, shift, dcf);

% NUFFT operator with reduced number of spokes for delay correction
% FTred = MCWNUFFT

% allocate output
vol3d = zeros(xres, xres, npar, nechoes);
smaps = zeros(xres, xres, npar, ncoils);
% cvol = zeros(xres, xres, ncoils, npar);

% parfor slc = 1:npar
for slc = 1:npar

    % make a copy of fourier transform operator (good practice if using
    % parfor)
    F = FT;

    % get data for current slice [samples*interleaves, echoes, 1, coils]
    sdata = reshape(data(:,:,slc,:,:),[xtot*nspokes,nechoes,1,ncoils]);

    % adjoint NUFFT to get individual coil images for each echo [n,n,e,r,c]
    cimgs = F' * sdata;

    % cvol(:,:,:,slc) = squeeze(cimgs(:,:,1,1,:));

    % estimate coil sensitivity maps from first echo
    cimgse1 = squeeze(cimgs(:,:,1,1,:));

    % do the sensitivity estimation
    if strcmpi(csmMethod, 'walsh')
        csm = estimateCoilSensitivitiesWalsh(cimgse1);
    elseif strcmpi(csmMethod,'espirit')
        csm = estimateCoilSensitivitiesESPIRiT(cimgse1,smapsRes);
    elseif strcmpi(csmMethod,'walsh-crop')
        csm = estimateCoilSensitivitiesWalsh(cimgse1,16);
    else
        csm = estimateCoilSensitivitiesWalsh(cimgse1);
    end

    % coil combination
    csmtmp = permute(conj(csm),[1,2,4,5,3]);
    cc = sum(bsxfun(@times, cimgs, csmtmp),5);
    vol3d(:,:,slc,:) = permute(cc,[1,2,4,3]);

    % store sensitivity maps
    smaps(:,:,slc,:) = permute(csm,[1,2,4,3]);

end
