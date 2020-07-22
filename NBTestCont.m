function [ predicLabel, accuracy, confusion, predictions ] = NBTestCont( means, vars, labelprobs, testSet, labelSet )
%NBTEST Summary of this function goes here
%   Detailed explanation goes here

    % get the number of labels testing against by counting our z-dimension
    numOfLabels = size(means, 1);
    numOfTests = size(testSet, 1);
    numOfFeatures = size(means, 2);
    
    predictions = zeros(numOfTests, numOfLabels);
   
    for t=1:numOfTests
        for l=1:numOfLabels
            predictions(t, l) = labelprobs(l);
            for f=1:numOfFeatures
                p = 1/(sqrt(vars(l, f)) * sqrt(2*pi));
                p = p * exp(-(testSet(t, f) - means(l, f))^2 / (2 * vars(l, f)));
                
                if (p > 0)
                    predictions(t, l) = predictions(t, l) * p;
                end
            end
        end
    end
                
        
    % create a vector to store our label predictions in
    predicLabel = zeros(numOfTests, 1);
    % count how many are correct by comparing with 
    numOfCorrect = 0;
    % create our 2d confusion matrix for recording incorrect values
    confusion = zeros(numOfLabels, numOfLabels);
    
    for test=1:numOfTests
        % go through every test, get the index of the label that has the
        % highest probability
        [pMax, index] = max(predictions(test, :));
        % set that as our prediction, minus 1 due to 0-indexing
        predicLabel(test) = index - 1;
        
        % check if our prediction matches the true label set
        if (predicLabel(test) == labelSet(test))
            numOfCorrect = numOfCorrect + 1;
        end
        
        % update our confusion matrix
        confusion(index, labelSet(test) + 1) = confusion(index, labelSet(test) + 1) + 1;
    end
    
    % calculate accuracy
    accuracy = numOfCorrect/numOfTests;
    
end


