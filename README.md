### Named Entities Recognition for Wines

In this repositry you can find my work on wines, with the goal of recognizing wine names, kinds and characteristics applying **Natural Language Processing** techniques on wine labels.

In the jupyter notebook above you can find a theory description and some code implementing the different NLP techniques I studied. In particular:

* In "Image Processing" you can find the preprocessing and application of OCR (Optical Character Recognition) over some flat wine labels, to obtain the sentences which were then analyzed, mave with opencv and pytesseract.
* In "Bi_LSTM" you can find a theory review from Recurrent Neural Network to Binary-LSTM, and an implementation of a BiLSTM network with keras and tensorflow to recognize wine aromas, taste and fruit gist.
* In "CondRandField" you can find a review on Maximum Entropy Models, Hidden Markov Models and Conditional Random Fields, and again the recognition of aromas, taste and fruit gist.
I have to thank [depends-on-the-definition](https://www.depends-on-the-definition.com/introduction-named-entity-recognition-python/) for making me understand the package usage.
* In "Clustering" I made an easy cluster of wine labels to better understand the division of aromas
* In "NER_Spacy" you can find a description of the [Spacy](https://spacy.io/) package for Named Entity Recognition and an application for the recognition of wine names, vineyard, year of production, taste, aromas and alcohol level
* In the folder **Opentag** I made an implementation of the amazon work on food labels applied on wines.
