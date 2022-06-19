function x_sharp = exactrecov(x,k,m,d)

%=========================================================================
% EXACT RECOVERY FUNCTION
%
% Liam Thomas, Math 123 final project
% Fall 2021
% Tufts University
%
% Description -- takes in signal x in R^n and returns low-dimensional
% k-sparse representation using sparse random matrices A.
%
% Arguments: x := signal in R^n
%            k := desired sparsity of our approximation vector x_sharp
%            m := number of measurements (tracks n sublinearly)
%            d := determines sparsity of each column of A, d << n 
%
%   A denotes the random matrix generated for this experiment.
%
%   Ax is referred to as the SKETCH or MEASUREMENT VECTOR of the signal x.
%
%   A is a binary sparse matrix with columns constructed via the uniform 
%   distribution.
%
% Based on paper by Berinde and Indyk, 2008

%=========================================================================


%=========================================================================
% GENERATION OF TEST SIGNAL x0
%=========================================================================

%Initialize k-sparse test signal
sz = size(x);
n = sz(1);
x0 = zeros(n,1);

%Generate test signal peaks at random
nrand = randperm(n);
krand = nrand(1:k);
x0(krand(1:(ceil(k/2)))) = -1;
x0(krand(ceil((k/2)):k)) = 1;

%Generate column indices at random
colvec = randi([1 n], k, 1);
%TODO: Make cols distinct, change to while loop?
peaks = zeros(n,1);
peakvec = randi([-1 1],k,1);
for i = 1:k
    if (peakvec(i) == 0)
        peakvec(i) = -1;
    end
    peaks(colvec) = peakvec(i);
end

%Add peaks to test signal
x0 = x0 + peaks;

%=========================================================================
% INITIAL MEASUREMENT VECTOR ("SKETCH") y = A * x0
%
% NOTE: A is the adjacency matrix of an expander graph with high
% probability.
%=========================================================================

%Initialize and generate A
A = zeros(m,n);
dmat = randi([1 m],d,n);
col_ind = repmat([1:n],d,1);
in = sub2ind([m n],dmat(:),col_ind(:));
A(in) = 1;


%TODO: make col indices distinct and cols of A distinct with while loop

%Compute our initial measurement, y
y = A * x0;


%=========================================================================
% Recovery of x#: first iteration
%=========================================================================

%Use l1-Magic package to recover x# from y = Ax0 with 1e-3 tolerance.

%This optimization algorithm solves the following problem:
%           min norm(x0) subject to Ax0 = y

%That is, the optimization attempts to find the x0 vector with the smallest
%1-norm that explains the observations y.

x_sharp = l1eq_pd(x0, A, [], y, 1e-3);

%Initialize error vector for use at end of experiment with l2 norm
%Record error for first iteration
error = zeros(11,1);
error(1) = norm(x_sharp - x0);

%=========================================================================
% Recovery of x#: 10 more iterations

% NOTE: we retain the same initial measurement x0 for each iteration.
%=========================================================================

%Intitialize count for while loop
count = 1;
while (count < 11)
    
    dmat = randi([1 m],d,n);
    col_ind = repmat([1:n],d,1);
    in = sub2ind([m n],dmat(:),col_ind(:));
    A(in) = 1;
    
    %Compute
    y = A * x0;
    x_sharp = l1eq_pd(x0, A, [], y, 1e-3);
    
    %Record error for each iteration and return upon convergence
    error(count) = norm(x_sharp - x0);
    if (error(count) < 1e-3)
        break
    end
    
    count = count + 1;
    
end

%=========================================================================
% Optional code to determine success or failure and print result
% - suppressed for now
%=========================================================================

%fprintf('Error on each iteration:\n',error);
%for i = 1:11
 %   if (error(i) > 1e-3)
  %      fprintf('FAILURE\n');
   % else
    %    fprintf('SUCCESS\n');
    %end
%end
