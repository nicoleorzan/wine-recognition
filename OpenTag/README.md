# OpenTag

**Reference**: G. Zheng, S. Mukherjee†, X. L. Dong†, F. Li ''OpenTag: Open Attribute Value Extraction from Product Profile'' (2018), University of Utah https://arxiv.org/abs/1806.01264

#### What's new?

1) **Word embeddings** which takes into account the tag of the word<br>
2) Novel **attention mechanism** (which cames after the BiLSTM layer) to explain model's decision<br>
3) **Active learning** to reduce at minimum the human labeling

#### NB: I implemented only the Active learning part, with Tag Flips method


### 1) Word embeddings

"Pretrained embeddings have a single representation for each token.
This does not serve our purpose as the same word can have a different representation in different contexts. For instance, ‘duck’ (bird) as a flavor-attribute value should have a different representation than ‘duck’ as a brand-attribute value. Therefore, we learn the word representations conditioned on the attribute tag (e.g., ‘flavor’)"


### 2) Attention mechanism
"With an attention mechanism, instead of encoding the full source sequence into a fixed-length vector, we allow the decoder to attend to different parts of the source sentence at each step of the output generation. Importantly, we let
the model learn what to attend to based on the input sentence and what it has produced so far."


### 3) Active learning

Starting with a small set of labeled instances L. The learner iteratively requests labels for one or more instances
from a large unlabeled pool of instances U using some query strategy Q, which we need to define.

Baseline: method of **Least Confidence** (LC). Therefore, the query strategy selects the sample x with maximum uncertainty given by:
$$
Q(x) = 1 - P(y^*|x, \phi)
$$
where $y^*$ is the best possible tag sequence for x.

Problems:<br>
1) the uncertainty on a single tag can pull down all sequence probability<br>
2) when the oracle reveals a tag, this may impact only on a few other tags, having a low impact on the fulll sequence.


#### Method of Tag Flips

We simulate a committee of tag learners: C = {$\phi_1$, ...., $\phi_E$}. The most informative sample is the one for which there is a mahor disagreement among committee members.<br>
To do this we train a single model for E epochs (using **dropout** and regularization): for each epoch we learn a different set of models and parameters, simulating in this way a committee of learners.<br>
We define  **Flip** to be a change in the tag of a token of a given sequence across successive epochs. If the tokens of a given sample sequence frequently change tags across successive epochs, this indicates we are uncertain about the sample. So we query for labels of the sample eith the highest number of tag flips. So to compute tag flips we do not need to know the labels, but we compare tag before and after the new epoch.
