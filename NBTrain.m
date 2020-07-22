function [ training, labelProbs ] = NBTrain( attributes, labels )
%NBTRAIN Trains a naive bayes classifier using a set of training data
%   Produces a 3d matrix, the first row contains data referring to the
%   overall classification and the rest contains data about each feature
    paramList = {};
    % get the number of features
    numFeatures = size(attributes, 2);
    % get the maximum value recorded for the features
    valueCount = max(attributes(:)) + 1;
    % vector to store the max numbers of each feature
    featureMax = zeros(numFeatures, 1);

    for f=1:numFeatures
        featureMax(f) = max(attributes(:, f));
    end

    % count how many labels we're classifying as, adding one as we're
    % indexing from 0
    labelCount = max(labels) + 1;
    % count the sample size
    totalSamples = size(attributes, 1);
    
    % create a 3d array to store the counts of the occurances of each
    % feature for each possible value & label
    featureCount = zeros(numFeatures, valueCount, labelCount);
    % count the total observed counts for each label to calc probs
    labelNumbers = zeros(labelCount);

    % loop through every sample
    for ex=1:totalSamples
        % loop through each feature in that sample
        for f=1:numFeatures
            % get the value of that feature, adding 1 as we're using as an
            % index
            attributeValue = attributes(ex, f) + 1;
            % increase the count for that value for that label of a feature
            featureCount(f, attributeValue, labels(ex)+1) = featureCount(f, attributeValue, labels(ex)+1) + 1;
        end
        % increase total count for each label
        labelNumbers(labels(ex) + 1) = labelNumbers(labels(ex) + 1) + 1;
    end
    
    for f=1:numFeatures
        for v=1:valueCount
           for l=1:labelCount
              % calculate the P(x|l) given us knowing l as the label
              % 0-case
              if featureCount(f, v, l) == 0
                m = max(1, ceil(totalSamples * 0.005));
                featureCount(f, v, l) = (m * 1/featureMax(f)) / (labelNumbers(l) + m)
              else
                featureCount(f, v, l) = featureCount(f, v, l) / labelNumbers(l);
              end
           end
        end
    end

    labelProbs = labelNumbers ./ totalSamples;
        
    
    training = featureCount;
end


