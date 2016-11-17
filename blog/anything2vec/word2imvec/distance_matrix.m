checkval = 1;

% Display associated metadata
words( metadata{checkval} )

% Create normalized word vectors
Vw = wordvecs ./ sqrt( repmat( sum(wordvecs.^2), [200, 1] ) );

% Normalize AlexNET vector
vq = W * alexvecs(checkval,:)'; vq = vq ./ sqrt( sum(vq.^2) );

% Compute dot product and then sort all results
[dval idx] = sort(vq'*Vw, 'descend');

% Display most relevant words
words( idx(1:30) )
