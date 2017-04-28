%% 
% This is an example submission for IEEE 14-bus system to test the end-end work flow 
% on Grid Optimization Platform
% Author: Dr. Xiaoyuan Fan
% Pacific Northwest National Laboratory
% March 21, 2017

function MyMatlab2( )
%MYMATLAB2 Summary of this function goes here
%   Detailed explanation goes here
%% Step 4: Example computation

global pfotemp

pfocase = pfotemp;

starttime2 = cputime;
for n = 1: 1
   %ExampleComputing(14);
end
%% Step 5: Write out the solution2.txt
pseudo_sol = [pfocase '/solution2.txt']
fp=fopen(pseudo_sol,'r');
totalsolrows = numel(cell2mat(textscan(fp,'%1c%*[^\n]')));
fclose(fp);

fp=fopen(pseudo_sol,'r');
fp_out = fopen('solution2.txt','w');
totalsolrows = numel(cell2mat(textscan(fp,'%1c%*[^\n]')));
fclose(fp);
fp=fopen(Inputraw,'r');
fp_out = fopen('solution2.txt','w');
for n = 1:totalsolrows
    currentline = fgetl(fp);
    fprintf(fp_out,'%s\n',currentline);
end
fclose(fp);
fclose(fp_out);
%% Step 6: Write out the log file 
elapsedtime2 = cputime - starttime2;
fp_log = fopen('log_temp2.txt','a+');
fprintf(fp_log,'%f\n',elapsedtime2);
fclose(fp_log);

end

