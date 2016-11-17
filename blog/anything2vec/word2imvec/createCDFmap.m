function wordcdfmap = createCDFmap(wordfreqs, maxval)

% Create a table for the CDF so that we can negatively sample into
% distribution

% Calculate the unnormalized CDF from the PDF
wordcdf(1) = wordfreqs(1);
for i = 2:length(wordfreqs)
    wordcdf(i) = wordcdf(i-1)+wordfreqs(i);
end
% Re-normalize to 1.
wordcdf = (wordcdf / sum(wordfreqs));

% Make the table with "maxval" entries
wordcdfs = wordcdf * maxval;
wordcdfmap = zeros( maxval , 1 );
i = 1; j = 1;
while i <= maxval
    if i <= wordcdfs(j)
        wordcdfmap(i) = j;
        i = i+1;
    else
        j = j+1;
    end
end