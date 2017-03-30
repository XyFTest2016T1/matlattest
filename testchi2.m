function x = testchi2(p,v);
%CHI2INV Inverse of the chi-square cumulative distribution function (cdf).
%	X = CHI2INV(P,V)  returns the inverse of the chi-square cdf with V  
%	degrees of freedom at the values in P. The chi-square cdf with V 
%	degrees of freedom, is the gamma cdf with parameters V/2 and 2.	
%
%	The size of X is the common size of P and V. A scalar input
%	functions as a constant matrix of the same size as the other input.	

%	References:
%	   [1]  M. Abramowitz and I. A. Stegun, "Handbook of Mathematical
%	   Functions", Government Printing Office, 1964, 26.4.
%	   [2] E. Kreyszig, "Introductory Mathematical Statistics",
%	   John Wiley, 1970, section 10.2 (page 144)

%	Copyright (c) 1993 by The MathWorks, Inc.
%	$Revision: 1.1 $  $Date: 1993/05/24 18:53:51 $

if nargin < 2, 
    error('Requires two input arguments.');
end

[errorcode p v] = distchck(2,p,v);

if errorcode > 0
    error('The arguments must be the same size or be scalars.');
end

% Call the gamma inverse function. 
x = testGam(p,v/2,2);

% Return NaN if the degrees of freedom is not a positive integer.
k = find(v < 0  |  round(v) ~= v);
if any(k)
    x(k) = NaN * ones(size(k));
end
