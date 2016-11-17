epochs = 100;
parserange = randperm(101);   % Which metadata to run over
gam = 1e-1;         % Numerical optimization
posth = 0.01;        % Positive sampling
alph = 0.1;        % Learning rate
negsamp = 25;       % Number of negative samples

maxval = 10^6; % CDF granularity

% Create a CDF map. Mikolov uses 10^8.
wordfreqs = wordcounts / sum(wordcounts) ;
wordcdfmap = createCDFmap(wordcounts, maxval);
disp('To negatively sample:>> wordcdfmap( round( rand*maxval ) );');

% Hash vector initialize randomly
Vh = rand(200, length(metadata));

% Loop through epochs and the parserange
for j = 1:epochs
    for i = parserange
        
        % imvector is the AlexNET vector
        vh = Vh(:,i); % / norm(alexvecs(i,:)');
        
        % positive samples
        if length(metadata{i}) % sometimes, the metadata is empty
            samples = positive_sample(wordfreqs, posth, metadata{i});            
            if ~length(samples) % on the off-chance that the samples are empty
                continue;
            end
        else
            continue;
        end        
        vp = wordvecs( :, samples );
        
        % negative samples
        samples = negative_sample(wordcdfmap, maxval, negsamp);
        vn = wordvecs( :, samples );
        
        % positive and negative weightings
        poswt = 1 - sigmoid( gam* vp'*vh );
        negwt = -sigmoid( gam* vn'*vh );
        vpw = sum(diag(poswt)*vp')';
        vnw = sum(diag(negwt)*vn')';
        
        % Updated Vh
        Vh(:,i) = alph*(vpw + vnw) + vh;
        
        % disp(['Finished processing metadata ' int2str(i) ' in epoch ' int2str(j)]);
        
    end
    disp(['Finished epoch ' int2str(j)]);
end

