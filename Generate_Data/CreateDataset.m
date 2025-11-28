tblname = 'C:\AM 2\Graduation Project\Middle stage\OptiNet\DataElem\DStot.xlsx';
%delete(tblname);

input_size = 9000;%3500*4;

tbl1 = readtable('C:\AM 2\Graduation Project\Middle stage\OptiNet\DataElem\DS1.xlsx',...
    'TextType','String', 'Sheet', 5);
idx1 = randperm(size(tbl1,1));
idx1 = idx1(1:input_size);
tbl1 = table2array(tbl1(idx1,:));
% 
% tbl2 = readtable('C:\AM 2\Graduation Project\Middle stage\OptiNet\DataElem\DataSet22.xlsx',...
%     'TextType','String', 'Sheet', 5);
% idx2 = randperm(size(tbl2,1));
% idx2 = idx2(1:input_size);
% tbl2 = table2array(tbl2(idx2,:));
% 
% tbl3 = readtable('C:\AM 2\Graduation Project\Middle stage\OptiNet\DataElem\DataSet23.xlsx',...
%     'TextType','String', 'Sheet', 5);
% idx3 = randperm(size(tbl3,1));
% idx3 = idx3(1:2000);
% tbl3 = table2array(tbl3(idx3,:));
% 
% tbl4 = readtable('C:\AM 2\Graduation Project\Middle stage\OptiNet\DataElem\DataSet24.xlsx',...
%     'TextType','String', 'Sheet', 3);
% idx4 = randperm(size(tbl4,1));
% idx4 = idx4(1:1000);
% tbl4 = table2array(tbl4(idx4,:));
% 
% tbl5 = readtable('C:\AM 2\Graduation Project\Middle stage\OptiNet\DataElem\DataSet25.xlsx',...
%     'TextType','String', 'Sheet', 5);
% idx5 = randperm(size(tbl5,1));
% idx5 = idx5(1:input_size);
% tbl5 = table2array(tbl5(idx5,:));

tbl5 = readtable('C:\AM 2\Graduation Project\Middle stage\OptiNet\DataElem\DS2.xlsx',...
    'TextType','String', 'Sheet', 5);
idx5 = randperm(size(tbl5,1));
idx5 = idx5(1:input_size);
tbl5 = table2array(tbl5(idx5,:));

tbl6 = readtable('C:\AM 2\Graduation Project\Middle stage\OptiNet\DataElem\DataSet31.xlsx',...
    'TextType','String', 'Sheet', 7);
idx6 = randperm(size(tbl6,1));
idx6 = idx6(1:input_size*2);
tbl6 = table2array(tbl6(idx6,:));



%new_table= [tbl1;tbl2; tbl3; tbl4; tbl5];
%new_table= [tbl1;tbl2];%tbl5;tbl6];
 new_table = [tbl1;tbl5;tbl6];

new_table = array2table(new_table);

 writetable(new_table,tblname,'Sheet',5);