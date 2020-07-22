function [ means, vars, classprob ] = NBTrainCont( attributes, labels )
%%
%NBTRAIN Trains a naive bayes classifier using a set of training data
%   Produces a 3d matrix, the first row contains data referring to the
%   overall classification and the rest contains data about each feature

    % get the number of features
    numFeatures = size(attributes, 2);
    % get the maximum value recorded for the features
    valueCount = max(attributes(:)) + 1;
    % count how many labels we're classifying as, adding one as we're
    % indexing from 0
    labelCount = max(labels) + 1;
    % count the sample size
    totalSamples = size(attributes, 1);
    
    % count the total observed counts for each label to calc probs
    labelNumbers = zeros(labelCount);
    
    % calculate the mean and variance for each feature for each label
    classMeans = zeros(labelCount, numFeatures);
    
    for ex=1:totalSamples
        for f=1:numFeatures   
            classMeans(labels(ex)+1, f) = classMeans(labels(ex) + 1, f) + attributes(ex, f);
        end
        
        % increase total count for each label
        labelNumbers(labels(ex) + 1) = labelNumbers(labels(ex) + 1) + 1;
    end
    
    for l=1:labelCount
        classMeans(l, :) = classMeans(l, :) ./ labelNumbers(l);
    end
    
    % variance now
    
    classVariance = zeros(labelCount, numFeatures);
    for ex=1:totalSamples
        for f=1:numFeatures
            classVariance(labels(ex) + 1, f) = classVariance(labels(ex) + 1, f) + ((attributes(ex, f) - classMeans(labels(ex) + 1, f)) ^ 2);
        end
    end
    
    for l=1:labelCount
        classVariance(l, :) = classVariance(l, :) ./ labelNumbers(l);
    end
    
    
    classprob = zeros(labelCount)
   
    for label=1:labelCount
        % calculate probability of each label occuring
        classprob(label) = labelNumbers(label) / totalSamples;
    end
    
    means = classMeans;
    vars = classVariance;
end