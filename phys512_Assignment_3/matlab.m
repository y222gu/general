
%%
mat_files_result_folder = uigetdir('');
param_data = readmatrix(chain);
figure
count = 0;
for i=1:6
    for j=1:6
        count = count + 1;
        subplot(6,6,count)
        x = param_data(:,i);
        y = param_data(:,j);
        scatter(x,y,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75])
        set(gca,'YTickLabel',[])
        set(gca,'XTickLabel',[])
        if j==1
            ylabel(param_names(i),'FontSize', fontsize);
        end
        if i==6
            xlabel(param_names(j),'FontSize', fontsize);
        end
    end
end

%%

your_vec = chain;
vec_len = length(your_vec(:,1));
histogram(your_vec(1:round(vec_len/2)), nbins, 'Normalization','probability');
hold on
histogram(your_vec(round(vec_len*(1/2)):vec_len), nbins, 'Normalization','probability');
hold on

