function vol = recon_admm(rsdata, smaps, FT, par, U)

if nargin < 5
    U = [];
end

% get some dimensions
[nsamp,npar,nechoes,ncoils,nbins] = size(rsdata);
xres = size(smaps,1); % [xres, xres, npar, ncoils]

% density precompensation
ntile = nsamp / length(FT.dcf);
rsdata = bsxfun(@times, rsdata, repmat(FT.dcf,[ntile,1]));

% IFT along partition direction
rsdata = ifft(ifftshift(rsdata,2),[],2);

% allocate output
vol = zeros(xres,xres,npar,nechoes,nbins);

% parfor slc = 1:npar
for slc = 1:npar
    
    % make a copy of NUFFT operator (good if using parfor)
    F = FT;

    % make a copy of parameters structure (good if using parfor)
    parslc = par;

    % get data with size [nsamp,nechoes,nbins,ncoils]
    data = permute(rsdata(:,slc,:,:,:),[1,3,5,4,2]);

    % get coil sensitivity maps with size [xres,xres,1,1,ncoils]
    csm = permute(smaps(:,:,slc,:),[1,2,3,5,4]);

    % sensitivity map operator
    S_for = @(a) bsxfun(@times, csm, a);
    S_adj = @(as) sum(bsxfun(@times, conj(csm), as),5);
    
    % full operator
    A_for = @(x) F*(S_for(x));
    A_adj = @(y) S_adj(F'*y);
    AHA = @(x) S_adj(F'*(F*(S_for(x))));
    
    % subspace projection operator
    if ~isempty(U)
        T_for = @(a) temporal_forward(a, U);
    end
    
    Fhy = F'*data;
    
    % scale the data so similar regularization parameters may be used
    tmp = Fhy;
    for d = 3:5
        if size(tmp,d) > 1
            tmp = dimnorm(tmp,d);
        end
    end
    tmpnorm = sort(tmp(:),'ascend');
    p100 = tmpnorm(end);
    p90 = tmpnorm(round(0.9*length(tmpnorm)));
    p50 = tmpnorm(round(0.5*length(tmpnorm)));
    if (p100-p90) < 2*(p90-p50)
        scaling = p90;
    else
        scaling = p100;
    end
    if scaling == 0
        scaling = 5000.0;
    end
    data = data./scaling;
    
    % calculate adjoint of data
    b = A_adj(data);

    % initial guess for reconstruction [N,N,echoes,K]
    x = zeros(size(b));
    
    % do the reconstruction 
    x = admm_solve(x, AHA, b, parslc);
    
    % convert from suspace coefficient images to respiratory binned images
    if ~isempty(U)
        x = T_for(x);
    end
    
    % store in output
    vol(:,:,slc,:,:) = scaling .* permute(x,[1,2,5,3,4]);
    
end





