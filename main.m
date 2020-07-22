close all;
clear all;
fname = input('Enter a filename to load data for training/testing: ','s');
load(fname);

% Put your NB training function below
[ parameterList, labelProbs ] = NBTrain(AttributeSet, LabelSet); % NB training
% Put your NB test function below
[predictLabel, accuracy, confusion] = NBTest( parameterList , labelProbs, testAttributeSet, validLabel); % NB test

fprintf('********************************************** \n');
fprintf('Overall Accuracy on Dataset %s: %f \n', fname, accuracy);
fprintf('********************************************** \n');


fprintf('\n\n');
fprintf('Confusion Matrix: \n');

 
    fprintf('X=Actual Class, Y=Predicted class\n\n');
    fprintf('\t  ');
for c=1:size(confusion, 1)
                   fprintf('  %d   ', (c -1));
                  
end
fprintf('\n\t___');
for c=1:size(confusion, 1)
    fprintf('______')
end

    fprintf('\n');

for y=1:size(confusion, 1)
    fprintf('\t%d |', y-1);
    for x=1:size(confusion, 1)
        fprintf('%d', confusion(y, x));
        for space=1:(6-max(max(ceil(log10(confusion(y, x)+1))), 1))
            fprintf(' ');
        end
    end
    fprintf('\n');
end
    %fprintf('             |
    %fprintf('Pred Class'  |   
