function [individual,EFs,count,graph]= OL_select_operation(n_d,value_space,objfun,func_num,EFs,max_EFs,count,graph,n_iteration,best_fitness)

% 对两个矩阵进行正交学习，选出适应度最好的个体
indi_best_combine=zeros(1,n_d);
%value_space=[indi_1(1,:)',indi_2(1,:)'];
%  get the corrsponing OA table
[factor_num,level_num]=size(value_space);
strength_num = floor(log2(factor_num))+1;
sample_OA=zeros(level_num^strength_num,factor_num);
table_OA=orthogonal_design(factor_num,level_num,strength_num);
% get the design points in real scaling
for ii=1:level_num^strength_num
    for jj=1:factor_num
        sample_OA(ii,jj)=value_space(jj,table_OA(ii,jj)+1);
    end
end
[row,col]=size(sample_OA);
for id=1:row
    individual_temp = sample_OA(id,1:n_d);
%     fv = fun(individual_temp);
    fv=feval(objfun,individual_temp',func_num);
    EFs = EFs+1;
     if EFs > max_EFs
            return;
     end
    count=count+1;
    if count==50
        graph = [graph;EFs,best_fitness(n_iteration)];
        count=0;
    end
    sample_OA(id,col+1) = fv;
end
%找到最佳组合
for column=1:col
    level=zeros(1,2);
    level_one=0;
    level_two=0;
    best_combine=sample_OA(:,[column col+1]);
    for subrow=1:row
        if best_combine(subrow,1)==value_space(column,1)
            level(1,1)=level(1,1)+best_combine(subrow,2);
            level_one=level_one+1;
        else
            level(1,2)=level(1,2)+best_combine(subrow,2);
            level_two=level_two+1;
        end
    end
    if level(1,1)/level_one<=level(1,2)/level_two
        indi_best_combine(1,column)=value_space(column,1);
    else
        indi_best_combine(1,column)=value_space(column,2);
    end
end
sort_sample = sortrows(sample_OA,col+1);
indi_temp(1,:) = sort_sample(1,1:col);
if feval(objfun,indi_temp',func_num)>=feval(objfun,indi_best_combine',func_num)
    individual=indi_best_combine;
else
    individual=indi_temp;
end


