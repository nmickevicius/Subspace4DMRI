function [rsraw,rstraj] = vane_reshuffleData(raw, traj, nav, nbins, sortMethod, expiration_proportion)

if nargin < 6
    expiration_proportion = 0.25;
end

% trajectory has shape [nsamp, 2, nilv, nechoes]
% reshape it to [nsamp, nilv, nechoes, 2]
traj = permute(traj, [1,3,4,2]);

[nsamp,~,npar,nechoes,ncoils] = size(raw);

if nbins == 1
    rsraw = reshape(raw,[],npar,nechoes,ncoils);
    rstraj = cell(nechoes,1);
    for e = 1:nechoes
        rstraj{e} = reshape(traj(:,:,e,:),[],2);
    end
    return;
end

if strcmpi(sortMethod,'hybrid')

    % determine navigator clipping to try and get similar number of spokes per bin
    [clipLow, clipHigh] = get_clips(nav, nbins);

    % do sorting with optimum clippings
    binvect = sort_hybrid(nav, nbins, clipLow, clipHigh);

elseif strcmpi(sortMethod, 'amplitudeExpirationWeighted')

    [~,ind] = sort(nav);
    np_exp = floor(length(nav)*expiration_proportion);
    np = floor((length(nav)-np_exp)/(nbins-1));
    binvect = zeros(length(nav),1);
    for bin = 1:nbins
        if bin == 1
            binInds = ind(1:np_exp);
        else
            bind1 = (bin-2)*np + np_exp + 1;
            bind2 = bind1 + np - 1;
            binInds = ind(bind1:bind2);
        end
        binvect(binInds) = bin;
    end


elseif strcmpi(sortMethod, 'amplitudeEqual')

    [~,ind] = sort(nav);           % sort based on amplitude
    np = floor(length(nav)/nbins); % number of spokes per bin
    binvect = zeros(length(nav),1);
    for bin = 1:nbins
        bind1 = (bin-1)*np + 1;
        bind2 = bind1 + np - 1;
        binInds = ind(bind1:bind2);% get np indices from sorted navigator
        binvect(binInds) = bin;
    end

end

% find the number of interleaves in each bin
nnilv = zeros(nbins,1);
for b = 1:nbins
    nnilv(b) = sum(binvect == b);
end
nilvMax = max(nnilv(:));

% allocate reshuffled data
rsraw = zeros(nsamp*nilvMax, npar, nechoes, ncoils, nbins);

% allocate reshuffled trajectory
rstraj = cell(nechoes,nbins);
for e = 1:nechoes
    for b = 1:nbins
        rstraj{e,b} = zeros(nsamp*nnilv(b),2);
    end
end

% within each bin, each echo will have same number of interleaves
% keep track of that number with totnilv
totnilv = zeros(nbins,1);

% loop through all points in navigator
for idx = 1:length(binvect)

    if binvect(idx) > 0

        % get current bin
        b = binvect(idx);

        % get indices to reshuffled array dimension 1
        rsidx1 = totnilv(b)*nsamp + 1;
        rsidx2 = rsidx1 + nsamp - 1;

        % shuffle k-space data
        tmp = permute(raw(:,idx,:,:,:), [1,3,4,5,2]); % [nsamp,npar,nechoes,ncoils,1]
        rsraw(rsidx1:rsidx2,:,:,:,b) = tmp;

        % shuffle trajectory (trajectory has size [nsamp, nilv, nechoes, 2]
        for e = 1:nechoes
            tmp = squeeze(traj(:,idx,e,:));
            rstraj{e,b}(rsidx1:rsidx2,:) = tmp;
        end

        % increment number of interleaves
        totnilv(b) = totnilv(b) + 1;

    end

end

function binvect = sort_hybrid(nav, nbins, clipLow, clipHigh)

    % making updates so all data in amplitude bins 1 and hnbins will not be
    % split into +/- slope bins

    hnbins = nbins/2 + 1; % NJM: added +1 here
    i1 = floor(0.05*length(nav)); % calculate range excluding first 5% to eliminate transients/filtering effects
    i2 = floor(0.95*length(nav)); %
    minval = min(nav(i1:i2));
    maxval = max(nav(i1:i2));
    range = maxval - minval;
    clippedMin = minval + clipLow*range;
    clippedMax = maxval - clipHigh*range;
    clippedRange = clippedMax - clippedMin;
    step = clippedRange / hnbins;
    rangeUpperVals = clippedMin + step.*(1:hnbins);

    % sort into nbins/2 bins based on amplitude
    binvect = zeros(length(nav),1);
    for bin = 1:hnbins
        if bin == 1
            binvect(nav <= rangeUpperVals(bin)) = bin;
        elseif bin == hnbins
            binvect(nav > rangeUpperVals(bin-1)) = bin;
        else
            binvect(nav <= rangeUpperVals(bin) & nav > rangeUpperVals(bin-1)) = bin;
        end
    end

    % differentiate navigator and sort into inspiration/expiration phases
    dndt = diff(nav);
    dmask = ones(length(dndt),1);
    dmask(dndt <= 0) = 2;
    dmask = [dmask(1); dmask];

    % add hnbins to the portions of binvect with positive slope while not modifying
    % the negative slope portions. Result is a hybrid sorting
    b = binvect;
    %b(dmask == 2 & binvect > 0) = nbins - b(dmask == 2 & binvect > 0) + 1;
    b(dmask == 2 & (binvect>1 & binvect<hnbins)) = nbins - b(dmask == 2 & (binvect>1 & binvect<hnbins)) + 2;
    binvect = b;

function [clipLow, clipHigh] = get_clips(nav, nbins)

clo = 0:0.01:0.2;
chi = 0:0.01:0.2;

cost = zeros(length(clo),length(chi));

for l = 1:length(clo)
    for h = 1:length(chi)
        bv = sort_hybrid(nav, nbins, clo(l), chi(h));
        nnilv = zeros(nbins,1);
        for b = 1:nbins
            nnilv(b) = sum(bv == b);
        end
        cost = std(nnilv);
    end
end
[~,ind] = min(cost(:));
[l,h] = ind2sub(size(cost),ind(1));
clipLow = clo(l);
clipHigh = chi(h);
