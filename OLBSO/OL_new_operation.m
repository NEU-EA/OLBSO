function [individual,EFs,count,graph]= OL_new_operation(n_d,value_space_new,indi_temp,stepSize,objfun,func_num,EFs,max_EFs,step_value,count,graph,n_iteration,best_fitness)

% 对两个矩阵进行正交学习，选出适应度最好的个体
indi_best_combine=zeros(1,n_d);
%value_space=[indi_1(1,:)',indi_2(1,:)'];
%  get the corrsponing OA table
[factor_num,level_num]=size(value_space_new);
strength_num = floor(log2(2*factor_num));
sample_OA_new=zeros(level_num^strength_num,factor_num);
table_OA_new=orthogonal_design(factor_num,level_num,strength_num);
%                 get the design points in real scaling
for ii=1:level_num^strength_num
    for jj=1:factor_num
        sample_OA_new(ii,jj)=value_space_new(jj,table_OA_new(ii,jj)+1);
    end
end
[row,col]=size(sample_OA_new);
for index=1:row
    indi_temp_step(1,1:n_d) = indi_temp(1,:) + stepSize .*sample_OA_new(index,1:col).*abs(step_value);
%     fv = fun(indi_temp_step);
    fv=feval(objfun,indi_temp_step',func_num);
    EFs = EFs+1;
     if EFs > max_EFs
            return;
     end
     count=count+1;
     if count==50
         graph = [graph;EFs,best_fitness(n_iteration)];
         count=0;
     end
    sample_OA_new(index,col+1) = fv;
end
%找到最优组合
for column=1:col
    level=zeros(1,2);
    level_one=0;
    level_two=0;
    best_combine=sample_OA_new(:,[column col+1]);
    for subrow=1:row
        if best_combine(subrow,1)==value_space_new(column,1)
            level(1,1)=level(1,1)+best_combine(subrow,2);
            level_one=level_one+1;
        else
            level(1,2)=level(1,2)+best_combine(subrow,2);
            level_two=level_two+1;
        end
    end
    if level(1,1)/level_one<=level(1,2)/level_two
        indi_best_combine(1,column)=value_space_new(column,1);
    else
        indi_best_combine(1,column)=value_space_new(column,2);
    end
end
sort_sample_new = sortrows(sample_OA_new,col+1);
new_step(1,:) = sort_sample_new(1,1:col);
indi_temp_bc(1,:) = indi_temp(1,:) + stepSize .*indi_best_combine(1,1:col).*abs(step_value);
if feval(objfun,indi_temp_bc',func_num)<=sort_sample_new(1,col+1)
    individual=indi_temp_bc;
else
    individual=indi_temp(1,:) + stepSize .*new_step(1,:).*abs(step_value)
end