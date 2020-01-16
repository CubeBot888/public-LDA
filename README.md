# Playing around with LDA
I wanted to test drive Latent Dirichlet Allocation (LDA), specifically its ability to associate documents with topics, where topics are likely defined by probability distributions over words. 
I know a potential limitation of LDA is that it uses a bag-of-words representation, considering marginal probability assignments (Ie. Bayesian predictive posteriors) to infer topics from word lists, thereby ignoring word sequences.  I believe that using Long-Short Term Memory (LSTM) cells could account for the word orderings, and guess that they may improve on LDA (and in addition provide for multiple topic labellings, though I believe Hierarchical LDA could also achieve this). 
An LSTM is, however, data greedy. Training deep learning algorithms on sparse data sets is likely to lead to difficulties in assessing bias-variance problems. Because I don’t have access to large amounts labelled data (and really I wanted to have a bash at NLP using unsupervised modelling), I wrote this Jupyter notebook implementation of LDA. 

## Method
To effectively implement LDA I will use the latent semantic modelling tool Gensim (link: https://radimrehurek.com/gensim/about.html).  To scrape a reduced amount of data from the web, I will use a Python client to query Wikipedia's API (link: https://github.com/martin-majlis/Wikipedia-API).
To begin, I collected a list of Wiki page summaries from a randomly generated list of concepts. Ideally, LDA should be trained on a diverse data set, else embeddings are unlikely to be well separated/defined by the algorithm. However, in the interest of time, I’ve just settled for a relatively diverse but EXTREMELY limited list of 11 Wiki concept write-ups.  
Each page’s summery (the initial block of text seen on a Wiki page) is tokenized using pre-processing functionality provided by NLTK (link: https://www.nltk.org/). Next, I transform the summaries into document-topic and topic-word distribution matrices, using LDA.
For interest sake, in my notebook I’ve printed out a subset of the topics as word lists with weightings.  Finally, I tested a new page’s summary. 

## Conclusion
Whilst evidently trained on too little data, the printed subset of topic-word clustering’s do appear to have common themes running through them.
The results for the singular test run on the Wiki page for 'Cluster analysis' appear related to the suggested topics’ words, with a strong relation to "mathematics".
In conclusion, the topic-word assignments seem reasonable, demonstrating LDA’s amazing ability to work on small training sets. 

## Further work
As already mentioned, I could gather more training data.
I could also improve LDA’s performance by using term frequency–inverse document frequency (TF-IDF) before training, to better emphasise how important a word is to a document in the corpus.

References
For some guidance I used the following:
-	https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
-	https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

