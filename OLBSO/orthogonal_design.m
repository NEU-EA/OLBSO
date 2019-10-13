function table_OA=orthogonal_design(factor_num,level_num,strength_num)
% input: factor_num : the number of factors
%              level_num  : the number of levels (all factors have the same level number)
%              strength_num: the strength number
% output: table_OA  : the corrsponding orthogonal design table with size of
%                    N * FactorNum (N=LevelNum^StrengthNum).
% ------------------------------------------------------------------------
% Note: This is not a complete orthogonal design tables, only the cases that
%        all factors have the same levels are considers.
% Note:  Only level number {2,3,4,5,7,8,9} are considered.
% Note:  The number of designs is N=LevelNum^StrengthNum, and can be high
%        as hundreds and thousands, which is suitable for engineering
%        approxmation.
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% check the strength_num is big enough
if (level_num^strength_num-1)/(level_num-1) < factor_num
    error('the strength_num is too small to contain all the factors')
end

% ------------------------------------------------------------------------
% initial the table value
t = level_num;
u = strength_num;
q=(t^u-1)/(t-1);
Table=zeros(t^u,q);

% ------------------------------------------------------------------------
% first step, the basic columns
basic_column=zeros(1,u);
for ii=1:u
    basic_column(ii)=(t^(ii-1)-1)/(t-1)+1;
    Table(:,basic_column(ii))=select_column(t,u,ii);       % t^i column
end

% ------------------------------------------------------------------------
%  second step, the mutual columns 
% adding table
add_matrix = add_table(t);              
% multiplying table
multiply_matrix=multiply_table(t);           

for ii=1:u-1
    % the basic column is i+1
    for jj=1:(t^ii-1)/(t-1)
        for k=1:t-1
            % the mutual column between two basic column is formed by
            % mutual the  basic column to all the former columns
            Table(:,basic_column(ii+1)+(jj-1)*(t-1)+k)=mutual(Table(:,jj),Table(:,basic_column(ii+1)),k,add_matrix,multiply_matrix);
        end
    end
end

% ------------------------------------------------------------------------
% randomly choose factor_num column from Table
p=randperm(size(Table,2));
table_OA=Table(:,p(1:factor_num));

end



function y = select_column(t,u,ui)
% ------------------------------------------------------------------------
% input : t the number of levels
%              u  the strength number
% output : the ui orthogonal column
% ------------------------------------------------------------------------
y=zeros(t^u,1);
% the number of levels
for i=1:t
    for j=1:t^(ui-1)
        %   the j level has t^(ui-1)  continuously blocks
        %   the index of each block has distance t^(u-1)-t^(u-ui)
        %   so each block has t^(u-ui) columns
        y((i-1)*t^(u-ui)+1+(j-1)*t^(u-ui)*t:i*t^(u-ui)+(j-1)*t^(u-ui)*t)=i-1;
    end
    
end

end


function y =mutual(a,b,k,add_matrix,multiply_matrix)
% ------------------------------------------------------------------------
% the default is :a*k+b
%  b is basic column
% ------------------------------------------------------------------------
l=length(a);
y=zeros(l,1);
for i=1:l
    y(i)=add_matrix(multiply_matrix(k+1,a(i)+1)+1,b(i)+1);
end
end




function add_matrix=add_table(t)
switch t
    case 2
        add_matrix=...
            [0	1;
            1	0];
    case 3
        add_matrix=...
            [0	1	2
            1	2	0
            2	0	1];
    case 4
        add_matrix=...
            [0	1	2	3;
            1	0	3	2;
            2	3	0	1;
            3	2	1	0];
    case 5
        add_matrix=...
            [0	1	2	3	4;
            1	2	3	4	0;
            2	3	4	0	1;
            3	4	0	1	2;
            4	0	1	2	3];
    case 7
        add_matrix= ...
            [0	1	2	3	4	5	6;
            1	2	3	4	5	6	0;
            2	3	4	5	6	0	1;
            3	4	5	6	0	1	2;
            4	5	6	0	1	2	3;
            5	6	0	1	2	3	4;
            6	0	1	2	3	4	5];
    case 8
        add_matrix=...
            [0	1	2	3	4	5	6	7;
            1	0	6	4	3	7	2	5;
            2	6	0	7	5	4	1	3;
            3	4	7	0	1	6	5	2;
            4	3	5	1	0	2	7	6;
            5	7	4	6	2	0	3	1;
            6	2	1	5	7	3	0	4;
            7	5	3	2	6	1	4	0];
    case 9
        add_matrix=...
            [0	1	2	3	4	5	6	7	8;
            1	5	8	4	6	0	3	2	7;
            2	8	6	1	5	7	0	4	3;
            3	4	1	7	2	6	8	0	5;
            4	6	5	2	8	3	7	1	0;
            5	0	7	6	3	1	4	8	2;
            6	3	0	8	7	4	2	5	1;
            7	2	4	0	1	8	5	3	6;
            8	7	3	5	0	2	1	6	4];
end

end



function multiply_matrix=multiply_table(t)
switch t
    case 2
        multiply_matrix=...
            [0	0;
            0	1];
    case 3
        multiply_matrix=...
            [0	0	0;
            0    1	  2;
            0   2	1];
    case 4
        multiply_matrix=...
            [0	0	0	0;
            0	 1	 2	 3;
            0	2	3	1;
            0	3	1	2];
    case 5
        multiply_matrix=...
            [0	0	0	0	0;
            0	1	2	3	4;
            0	2	4	1	3;
            0	3	1	4	2;
            0	4	3	2	1];
    case 7
        multiply_matrix= ...
            [0	0	0	0	0	0	0;
            0  1	2	3	4	5	6;
            0	 2	4	6	1	3	5;
            0	 3	6	2	5	1	4;
            0	 4	1	5	2	6	3;
            0	 5	3	1	6	4	2;
            0	 6	5	4	3	2	1];
    case 8
        multiply_matrix=...
            [0	0	0	0	0	0	0	0;
            0	1	2	3	4	5	6	7;
            0	2	3	4	5	6	7	1;
            0	3	4	5	6	7	1	2;
            0	4	5	6	7	1	2	3;
            0	5	6	7	1	2	3	4;
            0	6	7	1	2	3	4	5;
            0	7	1	2	3	4	5	6];
    case 9
        multiply_matrix=...
            [0	0	0	0	0	0	0	0	0;
            0	1	2	3	4	5	6	7	8;
            0	2	3	4	5	6	7	8	1;
            0	3	4	5	6	7	8	1	2;
            0	4	5	6	7	8	1	2	3;
            0	5	6	7	8	1	2	3	4;
            0	6	7	8	1	2	3	4	5;
            0	7	8	1	2	3	4	5	6;
            0	8	1	2	3	4	5	6	7];
end
end
