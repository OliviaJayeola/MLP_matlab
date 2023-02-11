clear;

W1 = [-0.2 0.1 -0.1 -0.2]; % [1  2  3  4
%                             w1 w3 w4 w6]

W2 = [-0.3 0.1 0.1 0.1 0.2]; % [1  2  3  4  5
%                               w9 w8 w7 w2 w5]

X1 = [0;1]; % rows of input matrices = number of examples
X2 = [1;1]; % each matrix = input for each example

NHIDDENS = [1 2 3];
index_X1_straight_to_out = 4; % index/indicies after NHIDDENS
index_X2_straight_to_out = 5; % index/indicies after index_X1_straight_to_out

index_input_to_hidden = [1 2 2 3]; % indicies of inputs corresponding to hidden layers 
index_X1_to_hidden = [1 2]; % indicies of inputs from X1 to hidden
index_X2_to_hidden = [3 4]; % indicies of inputs from X2 to hidden

Desired = [1;0];
Learnrate = 0.2;


for i = 1:size(X1,1) % incremental

    % forward prop

    OUT = 0; % output resets on each epoch

    % initialise output for hidden layers
    for j = 1:length(NHIDDENS) 
    
        in_to_hidden(i,j) = X1(i)*W1(i,j); % X1 * W1

        if j == NHIDDENS(2); in_to_hidden(i,j) = ...
                X1(i)*W1(i,j)+X2(i)*W1(i,j+1); end % X1 * W3 + X2 * W4

        if j == NHIDDENS(3); in_to_hidden(i,j) = X2(i)*W1(i,j+1); end % X2 * W6
        
        X = dlarray(in_to_hidden(i,j));
        Y(i,j) = sigmoid(X);
        OUT = OUT + Y(i,j)*W2(i,j);
    end
    
    % init output for straight to out
    for j = length(NHIDDENS)+1:index_X2_straight_to_out % to make the code general
        if ismember(j,index_X1_straight_to_out) == 1 
            OUT = OUT + X1(i)*W2(i,j); end 
        if ismember(j,index_X2_straight_to_out) == 1 
            OUT = OUT + X2(i)*W2(i,j); end 
    end

    ERROR = Desired(i) - OUT;
    BETA = ERROR; % deriv 
   

% Backprop of output layer

    for j = 1:length(W2) % amount of weights in output layer
            if j <= length(NHIDDENS) % inputs that had activation function
            dw_out(i,j) = Y(i,j)*BETA; 
            deltaw_out(i,j) = Learnrate*dw_out(i,j); 
            W2(i+1,j) = W2(i,j) + deltaw_out(i,j); end
        
        if ismember(j,index_X1_straight_to_out) == 1
            W2(i+1,j) = W2(i,j) + (Learnrate*BETA*X1(i)); end % perceptron rule
        
        if ismember(j,index_X2_straight_to_out) == 1 
            W2(i+1,j) = W2(i,j) + (Learnrate*BETA*X2(i)); end % perceptron rule
    end

% Backprop of hidden layer

    for j = 1:length(W1)
        % calculate error/beta of hidden layer
        if j <= length(NHIDDENS) 
            hiddenerr(i,j) = W2(i,j)*BETA;
            hiddenbeta(i,j) = Y(i,j)*(1-Y(i,j))*hiddenerr(i,j); end %deriv
        
        % calculate error of weights in hidden layer 
        % there are two weights from X1 to hidden
        if ismember(j, index_X1_to_hidden) == 1 
            % to make code general
            dw_hidden(i,j) = X1(i)*hiddenbeta(i,index_input_to_hidden(j)); 
            deltaw_hidden(i,j) = Learnrate*dw_hidden(i,j); end

        % there are two weights from X2 to hidden
        if ismember(j, index_X2_to_hidden) == 1 
            dw_hidden(i,j) = X2(i)*hiddenbeta(i,index_input_to_hidden(j));
            deltaw_hidden(i,j) = Learnrate*dw_hidden(i,j); end

        W1(i+1,j) = W1(i,j) + deltaw_hidden(i,j);
    end

    W1
    W2   
end






