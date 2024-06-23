Summary:
========

The lexical substitution corpus CoInCo ("Concepts in Context") is
based on contiguous texts provided in MASC (the manually annotated
subcorpus of the Open American National Corpus, OANC). It contains for
every content word in selected (complete) text files substitute words
collected via crowdsourcing (using Amazon Mechanical Turk, AMT) by 6
different annotators per token. The dataset comprises more than
150,000 responses for around 15,000 targets in about 2,500 sentences
containing approximately 35,000 words. Targets are roughly balanced
across the genres "news" and "fiction".

Annotators saw a target sentence with the highlighted two (sometimes
one, when the only leftover in the sentence) target words (that was
tagged in MASC as noun, verb, adjective, or adverb), and the preceding
and following sentence in the original text file as context (but shown
less prominent). They were asked to provide as many as possible
substitutes (up to 5) that would not change the meaning of the
sentence or otherwise mark why they did not provide one, choosing from
"proper name", "part of a fixed expression", "no replacement
possible", or "other problem (with description)". English as the
mother tongue of annotators could not be guaranteed, but we set the
residence address constraint for AMT participants to U.S.A.

We created target ID lists for a development/test set (65%/35% target
number split) that are nearly balanced across the two genres (news,
fiction) within each set. For each set we selected MASC text files and
included all the targets in these files - so both development and test
set contain full substitution data from sentences in contiguous
texts. Note that we used the complete target set for the models
described in the paper.


License
=======
The CoInCo download includes a sample of the MASC corpus,
which is available under the CC-BY-3.0-US license.
You can find MASC at http://www.anc.org/data/masc/.

The lexical substitution annotations that we added for CoInCo
are hereby published under the same CC-BY-3.0-US license.
Find the full text of the license here:
https://creativecommons.org/licenses/by/3.0/us/


Citation
========

More details can be found in:

Gerhard Kremer, Katrin Erk, Sebastian Padó, Stefan Thater: 
What Substitutes Tell Us – Analysis of an 
"All-Words" Lexical Substitution Corpus. 
Proceedings of EACL April, 2014. Gothenburg, Sweden.

@inproceedings{kremer14:_what_subst_tell_us,
  address = {Gothenburg, Sweden},
  author = {Kremer, Gerhard and Erk, Katrin and Pad\'o, Sebastian and Thater, Stefan},
  booktitle = {Proceedings of EACL},
  pages = {540--549},
  title = {What Substitutes Tell Us -- Analysis of an
                  'All-Words' Lexical Substitution Corpus},
  url = {http://www.aclweb.org/anthology/E14-1057.pdf},
  year = 2014
}




Files:
======
coinco.xml - "concepts in context" lexical substitution data in XML format
testset-tokenIDs.txt - target IDs for the test set, one per line
devset-tokenIDs.txt - target IDs for the development set, one per line



XML structure:
==============

We represent the annotated data in a simple XML format. The corpus is
a list of sentences with tokens as elements. For each token, we
provide the original MASC part-of-speech tags, the lemma and
part-of-speech tag as determined by TreeTagger, and the list of the
manually annotated substitutes (if applicable). Another binary
attribute ("problematic") marks targets (when set to "yes") where less
than 2 annotators provided substitutes. For each substitute, we
provide its annotation frequency and its TreeTagger lemma and PoS-tag.
To map sentences back onto the MASC corpus, we finally included the
original MASC text filename and sentence IDs as well as the context
sentences shown with the respective target sentence.

The original data was processed in UTF-8 character encoding. In the
XML file, non-ASCII characters have been transformed into XML
entities.


XML element structure tree: 
===========================
("+": at least one element of this type)

document
|__sent+
   |__precontext
   |__targetsentence
   |__postcontext
   |__tokens
      |__token+
         |__substitutions
            |__subst+


XML elements w/ attributes:
===========================

sent
(the top category containing all data for a target sentence)
. MASCfilename: the text file name as given in MASC
. MASCsentenceID: the sentence ID within the text file, as given in MASC

targetsentence
(the original target sentence as appearing in MASC,
containing substitution targets and non-content words)

precontext
(the sentence in MASC appearing before the target sentence)

postcontext
(the sentence in MASC appearing after the target sentence)

tokens
(containing each word token of the target sentence)

token
- for non-targets:
  . id: dummy-wordID ("XXX")
  . wordform: word token, taken from MASC
  . lemma: word lemma, taken from TreeTagger
  . posMASC: dummy-entry ("XXX")
  . posTT: part-of-speech, taken from TreeTagger
- for targets (w/ substitutions):
  . id: target token ID, unique across substitution corpus
  . wordform: wordform, taken from MASC (but w/o hyphens)
  . lemma: word lemma, taken from TreeTagger
  . posMASC: part-of-speech, taken from MASC
  . posTT: part-of-speech, taken from TreeTagger
  . problematic: "yes" if less than two annotators entered a substitute (otherwise: "no")

substitutions
(containing all substitution lemmata)

subst
(lexical substitution for the target token)
. lemma: lemma for substitute, taken from TreeTagger
  in case of multi-word substitutes, lemmata for all tokens are given
. pos: part-of-speech for substitute, taken from TreeTagger
  in case of multi-word substitutes, PoS-tags for all tokens are given
. freq: number of annotators that produced the substitute w/ this lemma



Additional information on data peculiarities:
=============================================

From the target set we heuristically removed auxiliary verbs that were
part-of-speech-tagged incorrectly in the gold standard.

In general, tokenisation is the same as in MASC (note that we verified
this for the target words, only). But, to provide lemmata from the
TreeTagger output for target sentences, we had sentences tokenised by
the TreeTagger. One observation there was that words connected with a
hyphen are tokenised in MASC as separate tokens, whereas they were
tokenised by the TreeTagger as single tokens. To enable a practically
feasible and correct automatic assignment of lemmata (via word
position indexes) to tokens, for TreeTagger input we substituted
hyphens in the target sentences with spaces - therefore, hyphens are
not included in tokens. Nevertheless, the XML-element "targetsentence"
contains the original sentence as appearing in MASC (i.e., with
hyphens - to check for tokenisation differences, concatenate all
tokens of a sentence by spaces and compare with "targetsentence").

Similarly, 2.6 % of the target sentences in MASC (65) were tokenised
differently from the TreeTagger output because of apostrophes (e.g.,
in names). We completely removed these sentences from the data
collection.

We notice that, because of duplicate sentenceID/range-lines in MASC,
in some cases (concerning 534 out of 2,474 sentences) we had shown the
annotators a copy of the respective target sentence as "postcontext"
(resulting in less informative context then would have been possible
during annotation). We corrected this bug, but kept the "postcontext"
in the resulting XML as it was presented in the experiment. For the
correct "postcontext" sentence, please consult the MASC data.

Because of duplicate sentence IDs (see above), some target instances
were annotated by more participants than intended, and sometimes the
same annotator saw the same target instance twice.

In the open tasks (free to everyone, to find capable annotators), there
were 10 annotators per target instance. In the closed tasks (only open
to trustfully, invited annotators from the open tasks), 6 annotators
processed each target instance. 490 targets were processed by more
than 6 annotators.



MASC text files with lexical substitution annotations:
======================================================

Genre fiction:

lw1.txt
captured_moments.txt
Nathans_Bylichka.txt


Genre newspaper/-wire:

20000410_nyt-NEW.txt
20000415_apw_eng-NEW.txt
20000419_apw_eng-NEW.txt
20000424_nyt-NEW.txt
A1.E1-NEW.txt
A1.E2-NEW.txt
enron-thread-159550.txt
NYTnewswire1.txt
NYTnewswire2.txt
NYTnewswire3.txt
NYTnewswire4.txt
NYTnewswire5.txt
NYTnewswire6.txt
NYTnewswire7.txt
NYTnewswire8.txt
NYTnewswire9.txt
wsj_0006.txt
wsj_0026.txt
wsj_0027.txt
wsj_0032.txt
wsj_0068.txt
wsj_0073.txt
wsj_0106.txt
wsj_0120.txt
wsj_0124.txt
wsj_0127.txt
wsj_0132.txt
wsj_0135.txt
wsj_0136.txt
wsj_0144.txt
wsj_0150.txt
wsj_0151.txt
wsj_0152.txt
wsj_0157.txt
wsj_0158.txt
wsj_0159.txt
wsj_0160.txt
wsj_0161.txt
wsj_0165.txt
wsj_0167.txt
wsj_0168.txt
wsj_0169.txt
wsj_0171.txt
wsj_0172.txt
wsj_0173.txt
wsj_0175.txt
wsj_0176.txt
wsj_0184.txt
wsj_0187.txt
wsj_0189.txt
wsj_1640.mrg-NEW.txt
wsj_2465.txt
