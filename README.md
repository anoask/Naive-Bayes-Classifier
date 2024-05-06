# Naive-Bayes-Classifier
Sequential Probability Estimation Model with Trigram Approach
Assignment Objective:
The objective for this project was simple: using a defined 10k sentence subset of the Brown Corpus, build a Naive Bayes Classifier that can identify the source genre of one of the sentences in the subset.
Implementation Steps:
The first step was to obtain our subset. The range we were provided was [32000,42000). To obtain the sentences given that range is very easy, but to categorize them into what source genre was a little more difficult. To do this, we created two methods: get_sentences_with_fids(corpus,start,end) and convert_fid_to_genre(fid_sentences,output). get_sentences_with_fids returned a dictionary of all of the sentences with a key that corresponded to the fileid of the source of the sentence. Using the fileid, we were able to make a dictionary that identified the source genre of the sentence which was done in convert_fid_to_genre.
The second step was to do the preprocessing of the sentences. We had two methods for this: preprocess_data(sentences) and process_sentence(sentence). Using the sentences we had gotten from step 1, preprocess_data would use process_sentence to tokenize all of the words in each sentence, and eliminate words that consisted only of symbols using a regex query. preprocess_data would return two dictionaries with the key being the source genre, and the entries being the corresponding processed sentences and vocab.
The third step was to build a probability model. We created a class called TrigramLaplaceSmoothing which initialized with a vocab, sentences, count model, and alpha. Either sentences or a model is required. This class had multiple methods, the ones used outside testing were:
1. def add_sentence(self,sentence)
a. This allows us to add a sentence to the model and vocab.
2. def set_alpha(self,alpha)
a. This allows us to update the alpha of the model.
3. def probability(self,w1,w2,w3)
a. Given w1,w2, it returns the probability of w3 using Laplace Smoothing and the
class defined self.alpha. The formula for this is
(𝐶𝑜𝑢𝑛𝑡(𝑤1, 𝑤2) + 𝑎)/(𝐶𝑜𝑢𝑛𝑡(𝑤1, 𝑤2, 𝑤3) + 𝑎 * |𝑙𝑒𝑛(𝑣𝑜𝑐𝑎𝑏)|)
where a is the classifier's alpha.
  
 4. def probability_of_sentence(self,sentence)
a. Using self.probability(w1,w2,w3), it returns the probability of a sentence existing
in the model. It should be known that the probability returned is the sum of log(self.probability(w1,w2,w3)), so that we don’t have to deal with small floats close to zero.
The fourth step was to build the Naive Bayes Classifier. We created a class TrigramClassifier that initialized with a corpus, sentence range, a tuple containing the split between training,validation, and testing percentages(e.g. (0.8,0.1,0.1)), and an alpha value. The initializer obtains the sentences using methods mentioned in step 1, and then shuffles them using np.random.shuffle. Then the tuple is used to split the sentences into training, validation, and testing subsets that are automatically processed through the method preprocess_data that is mentioned in step 2. Finally, a dictionary named classifiers is made with the key being a genre of the sentences and the entry being a TrigramLaplaceSmoothing object initialized with training_sentences and training_vocab of the class of the key. This class had multiple methods, the ones used outside testing were:
1. def predict_genres_of_sentence_probabilities(self,sentence)
a. This returns a dictionary of key:genre value:corresponding probability of a
sentence using the method probability_of_sentence() method in TrigramLaplaceSmoothing plus
𝑙𝑜𝑔((𝐶𝑜𝑢𝑛𝑡(𝑠𝑒𝑛𝑡𝑒𝑛𝑐𝑒𝑠)𝑐𝑙𝑎𝑠𝑠 + 𝑎)/(𝑡𝑜𝑡𝑎𝑙_𝑠𝑒𝑛𝑡𝑒𝑛𝑐𝑒𝑠 + 𝑎 * |𝐶𝑜𝑢𝑛𝑡(𝑡𝑜𝑡𝑎𝑙_𝑣𝑜𝑐𝑎𝑏)|).
This requires a smoothing parameter in the case a class that exists in the corpus
that doesn’t exist in the training data is found. 2. def predict_genres_of_sentence(self,sentence)
a. This returns the classification based off of the highest probability in the dictionary received from predict_genres_of_sentence_probabilities provided the input sentence.
3. def evaluate(self,level=’test’)
a. Given a level(train,val,test), this returns a dictionary containing a scoresheet of
true positives, true negatives, false positives, and false negatives per genre. The
level defines what set the models are being evaluated on. 4. def reset_validation(self,level)
a. If validation has occurred, this will reset the TrigramLaplaceSmoothing classes to the values they were prior validation.
5. def validation(self,alphas=[0.001,0.005,0.01,0.25,0.5,0.8,1])
a. Given a list of alphas, this will find which alpha combination for the classifiers
returns the highest correct combinations. Once complete, it will set the alpha values of each classifier to the corresponding alpha value that returned the most successes and then supplement the model with the validation sentences and vocab using the add_sentence method in TrigramLaplaceSmoothing class.
The fifth step was to use TrigramClassifier and evaluate its output. We created one new method for this: confusion_matrix(scores) which takes the output from evaluate and prints out the precision,recall, F1 score per genre and overall model accuracy.

 Evaluation:
Our model performs better than a dummy classifier. The range of overall accuracy we’ve seen is from 0.71-0.73 depending on the randomized training, validating, and testing set. Our corpus only has two genres, Learned(62% of our subset) and Fiction(38% of our subset). Our training percent was 80%, validation percent was 10%, and testing was also 10%. Like recommended in the provided code, we evaluated our code three times. Once just with the training data before the validation data was introduced, once with training data and validation after the validation data was introduced, and finally once on the testing data. The following tables display our results:
Training Evaluation(Before Validation Data Introduced)
   Genre: Learned
Genre: Fiction
TP:4210
FN:727
TP:3036
FN:0
FP:0
TN:3063
FP:727
TN:4210
Precision: 1 Recall: 0.853 F1-Score: 0.921
Precision: 0.808 Recall: 1.0 F1-Score: 0.894
Overall Accuracy:0.909
Validation Evaluation(After Validation Data Introduced and Alpha’s selected)
 Genre: Learned
Genre: Fiction
TP:5450
FN:81
TP:3352
FN:117
FP:117
TN:3352
FP:81
TN:5450
 Precision: 0.979 Recall: 0.985 F1-Score: 0.982
Precision: 0.976 Recall: 0.966 F1-Score: 0.971
Overall Accuracy:0.978
Testing Evaluation
Precision: 0.842 Recall: 0.643 F1-Score: 0.729
Precision: 0.599 Recall: 0.815 F1-Score: 0.690
 Genre: Learned
Genre: Fiction
TP:389
FN:216
TP:322
FN:73
FP:73
TN:322
FP:216
TN:389
 Overall Accuracy:0.711

Challenges:
Once we started working on the correct objective, we only ran into two distinct issues. The first was getting the genres of our subset of the Brown Corpus. We mentioned our solutions to this in step 1, and am glad that it ended up working correctly. The second issue we ran into was selecting the hyperparameters for each classifier during validation testing. After researching different methods and testing, we decided that the optimal way to find the best way to do it was to provide a list of possible alpha-values, and then find which combination returned the greatest success. This is really only possible because of the low number of genres our subset had, as the number of possible combinations of alphas in a provided list is an exponential problem, but our method worked out very well for us.
