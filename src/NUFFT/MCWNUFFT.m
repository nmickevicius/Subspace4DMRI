classdef MCWNUFFT
    % MCWNUFFT: class for multi- coil/echo/repetition 2D NUFFT operations

    properties
        N
        st
        dcf
        nmax
        nechoes
        nreps
        ktraj
        adjoint = false;
    end

    methods

        function obj = MCWNUFFT(ktraj,mtrx,shift,dcf)
            NN = [mtrx,mtrx];
            JJ = [5,5];
            NNos = 2.*NN;
            [obj.nechoes,obj.nreps] = size(ktraj);
            obj.N = mtrx;
            obj.ktraj = ktraj;
            obj.dcf = dcf;
            obj.st = cell(obj.nechoes,obj.nreps);
            obj.nmax = 0;
            for e = 1:obj.nechoes
                for r = 1:obj.nreps
                    if size(ktraj{e,r},1) > obj.nmax
                        obj.nmax = size(ktraj{e,r},1);
                    end
                    obj.st{e,r} = nufft_init(ktraj{e,r}, NN, JJ, NNos, NN/2+shift, 'minmax:kb');
                end
            end
        end

        function obj = ctranspose(obj)
            obj.adjoint = ~obj.adjoint;
        end

        function out = mtimes(obj,inp)


            if obj.adjoint
                % inp - input data cell array with size
                % [nmax,nechoes,nreps,ncoils]
                ncoils = size(inp,4);
                out = zeros(obj.N, obj.N, obj.nechoes, obj.nreps, ncoils);
                for e = 1:obj.nechoes
                    for r = 1:obj.nreps
                        ntile = floor(size(obj.ktraj{e,r},1)/length(obj.dcf));
                        ndcf = repmat(obj.dcf(:),[ntile,1]);
                        for c = 1:ncoils
                            out(:,:,e,r,c) = nufft_adj(inp(1:size(obj.ktraj{e,r},1),e,r,c).*ndcf, obj.st{e,r});
                        end
                    end
                end
            else
                ncoils = size(inp,5);
                out = zeros(obj.nmax,obj.nechoes,obj.nreps,ncoils);
                for e = 1:obj.nechoes
                    for r = 1:obj.nreps
                        ntile = floor(size(obj.ktraj{e,r},1)/length(obj.dcf));
                        ndcf = repmat(obj.dcf(:),[ntile,1]);
                        for c = 1:ncoils
                            out(1:size(obj.ktraj{e,r},1),e,r,c) = nufft(inp(:,:,e,r,c),obj.st{e,r}).*ndcf;
                        end
                    end
                end
            end

        end

    end

end
