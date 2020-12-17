function x = admm_solve(x, Aop, b, par)

% x - guess for reconstruction (usually zeros)

z = zeros(size(x));
u = zeros(size(x));

% get some parameters from structure
rho = par.rho;
admm_iters = par.admm_iters;
wav_lambda = par.wav_lambda;
lsqr_tol = par.lsqr_tol;
lsqr_iters = par.lsqr_iters;

if strcmpi(par.reg,'wav')
    sx = size(x,1);
    sy = size(x,2);
	ssx = 2^ceil(log2(sx)); 
	ssy = 2^ceil(log2(sy));
	ss = max(ssx, ssy);
	W = Wavelet('Daubechies',4,4);
end

% start ADMM iterations
for i = 1:admm_iters

    % update x using conjugate gradient methods
    AHA = @(a) vec(rho * reshape(a,size(b)) + Aop(reshape(a,size(b))));
    [a,~,~,~] = symmlq(AHA, b(:)+rho*(z(:)-u(:)), lsqr_tol, lsqr_iters, [], [], x(:));
    x = reshape(a,size(b));

    xpu = x + u;
    
    if strcmpi(par.reg,'wav')
        if ss ~= sx || ss ~= sy
            xpu = zpad(xpu, ss, ss, size(xpu,3), size(xpu,4)); % zero pad to closest diadic
        end
        Wx = W*xpu;
        if length(size(Wx)) < length(size(xpu))
            Wx = permute(Wx,[1,2,4,3]);
        end
        Wx = SoftThresh(Wx, wav_lambda/rho);
        z = W'*Wx;
        if length(size(z)) < length(size(xpu))
            z = permute(z,[1,2,4,3]);
        end
        if ss ~= sx || ss ~= sy
            z = crop(z, sx, sy, size(z,3), size(z,4));
        end
    elseif strcmpi(par.reg,'none')
        z = x + u;
    end

    % update u
    u = xpu - z;

end
