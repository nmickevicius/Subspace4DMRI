function smaps = estimateCoilSensitivitiesWalsh(imgs,cropval)

if nargin < 2
    cropval = [];
end

[nrows,ncols,ncoils] = size(imgs);

if ~isempty(cropval)
    ks = fftshift(fft(fftshift(imgs,1),[],1),1);
    ks = fftshift(fft(fftshift(ks,2),[],2),2);
    zks = zeros(size(ks));
    r1 = floor(nrows/2 - cropval/2); r2 = r1 + cropval - 1;
    c1 = floor(ncols/2 - cropval/2); c2 = c1 + cropval - 1;
    zks(r1:r2,c1:c2,:) = ks(r1:r2,c1:c2,:);
    imgs = ifftshift(ifft(ifftshift(zks,2),[],2),2);
    imgs = ifftshift(ifft(ifftshift(imgs,1),[],1),1);
end

filterSize = 7;

Rs = zeros(nrows,ncols,ncoils,ncoils);
for c1 = 1:ncoils
    for c2 = 1:ncoils
        Rs(:,:,c1,c2) = filter2(ones(filterSize), imgs(:,:,c1).*conj(imgs(:,:,c2)),'same');
    end
end

smaps = zeros(nrows,ncols,ncoils);
for row = 1:nrows
    for col = 1:ncols
        [U,~] = svd(squeeze(Rs(row,col,:,:)));
        m = U(:,1);
        smaps(row,col,:) = m;
    end
end
