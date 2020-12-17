classdef MCWLRNUFFT

    properties
        p
        sn
        N
        K
        T
        Nos
        nd
        dcf
        adjoint = false;
        dataMask
    end

    methods

        function obj = MCWLRNUFFT(ktraj, dataMask, N, U, dcf, shift)
            % ktraj - cell array [nechoes,nbins]
            % N     - matrix size
            % U     - basis functions

            if nargin < 5 || isempty(dcf)
                dcf = 1;
            end

            if nargin < 6 || isempty(shift)
                shift = [0,0];
            end

            dataSize = size(dataMask);
            obj.dataMask = dataMask;

            Nos = 2*N;
            nd = dataSize(1); % maximum number of samples per bin
            np = Nos*Nos;
            K = size(U,2);
            T = size(U,1);

            obj.nd = nd;
            obj.N = N;
            obj.Nos = Nos;
            obj.K = K;
            obj.T = T;
            obj.dcf = dcf;

            obj.p = cell(size(ktraj,1),1);

            for e = 1:size(ktraj,1)

                mmall = [];
                kkall = [];
                uuall = [];

                for b = 1:size(ktraj,2)

                    % get gridding coefficients for current echo/bin
                    [init,mm,kk,uu] = nufft_init_nyu(ktraj{e,b}, [N,N], [5,5], [Nos,Nos], [N,N]./2+shift, 'minmax:kb');
                    if b == 1
                        obj.sn = init.sn;
                    end

                    for sv = 1:K
                        mmall = [mmall; mm+((b-1)*nd)];
                        kkall = [kkall; kk+((sv-1)*np)];
                        uuall = [uuall; U(b,sv)*uu];
                    end

                end

                % create sparse interpolation matrix
                obj.p{e} = sparse(mmall, kkall, uuall, nd*size(ktraj,2), obj.Nos*obj.Nos*obj.K);

            end

        end

        function obj = ctranspose(obj)
            obj.adjoint = ~obj.adjoint;
        end

        function A = mtimes(obj,B)

            if obj.adjoint

                % input - [nsampmax,echoes,bins,coils]

                % density compensation
                B = bsxfun(@times, B, obj.dcf);

                % zero-out where we don't have data
                B = B .* obj.dataMask;

                A = zeros(obj.N, obj.N, size(B,2), obj.K, size(B,4));

                % loop over echoes
                for e = 1:size(B,2)

                    % re-arrange data for current echo
                    data = permute(B(:,e,:,:),[1,3,4,2]); % [nsampmax, nbins, ncoils]
                    data = reshape(data,[],size(B,4));

                    % grid to cartesian k-space
                    k = reshape(full(obj.p{e}'*data), [obj.Nos, obj.Nos, obj.K, size(B,4)]);

                    % bring to image space
                    x = ifft(ifft(k,[],1),[],2);

                    % crop to remove oversampling
                    x = x(1:obj.N,1:obj.N,:,:);

                    % correction for IFT of interpolation kernel
                    x = bsxfun(@times, x, conj(obj.sn));

                    % scaling
                    x = x * sqrt(obj.Nos*obj.Nos);

                    % store in output
                    A(:,:,e,:,:) = permute(x,[1,2,5,3,4]);

                end

            else

                A = zeros(obj.nd, size(B,3), obj.T, size(B,5));

                % IFT of interpolation kernel correction
                B = bsxfun(@times, B, obj.sn);

                for e = 1:size(B,3)

                    % rearrange data for current echo
                    data = permute(B(:,:,e,:,:),[1,2,4,5,3]); % [N,N,K,coils]

                    % bring to k-space
                    k = fft(fft(data,obj.Nos,1),obj.Nos,2);

                    % reshape k-space to [[],ncoils]
                    k = reshape(k,[],size(data,4));

                    % de-grid the k-space
                    dgk = obj.p{e} * k;

                    % scaling
                    dgk = dgk ./ sqrt(obj.Nos*obj.Nos);

                    % reshape
                    dgk = reshape(dgk,[],obj.T,size(B,5)); % [nsampmax, T, ncoils]

                    % store in output
                    A(:,e,:,:) = permute(dgk,[1,4,2,3]);

                end

                % multiply by the data mask
                A = A .* obj.dataMask;

                % density compensation
                A = bsxfun(@times, A, obj.dcf);

            end

        end

    end

end
