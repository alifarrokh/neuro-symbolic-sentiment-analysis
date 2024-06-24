# English Lexical Substitution Task Dataset
The **English Lexical Substitution Task [(SemEval 2007 Task 10)](https://aclanthology.org/S07-1009.pdf)** dataset is available on [GitHub](https://github.com/orenmel/lexsub).

## Dataset Files
The following three files are used in this project:

1. **`lst_all.xml`**: Contains 2010 example sentences for 205 polysemous words.
2. **`lst.gold.candidates`**: Used to extract candidate words (K=15) for each polysemous target word.
3. **`lst_all.gold`**: Used to extract ground truth candidates.

## Note
The ground truth labels of the following sentence IDs are not available in `lst_all.gold`:
- **567, 773, 794, 804, 1298, 1886, 1937**