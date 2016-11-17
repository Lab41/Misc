function index = positive_sample(wordfreqs, thresh, possamp)

whichones = ~ (wordfreqs(possamp) > (rand(length(possamp),1) * thresh));
index = possamp(whichones);