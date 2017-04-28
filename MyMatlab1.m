%% 
% This is an example submission for IEEE 14-bus system to test the end-end work flow 
% on Grid Optimization Platform
% Author: Dr. Xiaoyuan Fan
% Pacific Northwest National Laboratory
% March 21, 2017

function MyMatlab1( )
%MYMATLAB1 Summary of this function goes here
%   Detailed explanation goes here
%% 
% This is an example submission for IEEE 14-bus system to test the end-end work flow 
% on Grid Optimization Platform
% Author: Dr. Xiaoyuan Fan
% Pacific Northwest National Laboratory
% March 21, 2017
global pfotemp

pfocase = pfotemp;

close all;

Nbus = 14; % Can choose 5, 14, 30, 118 for example computing
%% Step 1: Reading the raw file
starttime1 = cputime;
Inputraw = [pfocase '/powersystem.raw'];
fp=fopen(Inputraw,'r');
totalrows = numel(cell2mat(textscan(fp,'%1c%*[^\n]')));
fclose(fp);
fp=fopen(Inputraw,'r');
firstline = fgetl(fp);
for n = 2:totalrows-1
    fgetl(fp);
end
lastline = fgetl(fp);
fclose(fp);
Outputlog = fopen('log_temp.txt','a+');
fprintf(Outputlog,'%d %s\n', 1, firstline);
fprintf(Outputlog,'%d %s\n', totalrows, lastline);
fclose(Outputlog);

%% Step 2: Exmaple computation
for n = 1: 1
   %ExampleComputing(14);
end
%% Step 3: Write out the solution1.txt
%Inputraw = 'solutionraw1.txt';
pseudo_sol = [pfocase '/solution1.txt'];
fp=fopen(pseudo_sol,'r');
totalsolrows = numel(cell2mat(textscan(fp,'%1c%*[^\n]')));
fclose(fp);
fp=fopen(pseudo_sol,'r');
fp_out = fopen('solution1.txt','w');
for n = 1:totalsolrows
    currentline = fgetl(fp);
    fprintf(fp_out,'%s\n',currentline);
end
fclose(fp);
fclose(fp_out);
elapsedtime1 = cputime - starttime1;
fp_log = fopen('log_temp.txt','a+');
fprintf(fp_log,'%f\n',elapsedtime1);
fclose(fp_log);

end

