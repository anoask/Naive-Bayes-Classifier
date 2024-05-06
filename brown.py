# %% [markdown]
# Imports NLTK and Regex

# %%
import nltk
from nltk import bigrams, trigrams
from collections import defaultdict, Counter
from nltk.corpus import brown #brown dataset required
from nltk.stem import PorterStemmer
import regex
import copy
import numpy as np
nltk.download("brown")

# %% [markdown]
# Preprocess steps like tokenization and normalization function

# %%
def process_sentence(sentence):
  stemmer = PorterStemmer()
  new_sentence = ["<S>","<S>"]
  stem_sentence = [stemmer.stem(word) for word in sentence] #Stem words
  #for word in sentence:
  for word in stem_sentence:
    word_span = regex.match("([[:punct:]])*",word).span() #remove punctuation
    if(word_span[0] == 0 and word_span[1] == len(word)): #If the 'word' is punctuation, dont add it to processed_sentences
      continue
    new_sentence.append(word.lower())
  new_sentence.append("<E>")
  return new_sentence

def preprocess_data(sentences):#Given a list of [(topic,[sentence])], it returns two dicts, one of vocab set and one of processed&tokenized sentences with key being genre
  # TODO: Add preprocessing steps (tokenization, normalization, etc.)
  processed_sentences = defaultdict(lambda:[])
  vocab = defaultdict(lambda:{"<S>","<E>"})
  for topic_sentence in sentences:
    sentence = topic_sentence[1]
    new_sentence = process_sentence(sentence)
    for word in new_sentence:
      vocab[topic_sentence[0]].add(word.lower())
    processed_sentences[topic_sentence[0]].append(new_sentence)
  return processed_sentences, vocab

def convert_fid_to_genre(fid_sentences,output='dict'):#Given a dict of FID:Sentences, it converts it to either a dict or list of genre to sentences
  fid_to_genre= {"a":"news",
                 "b":"editorial",
                 "c":"reviews",
                 "d":"religion",
                 "e":"hobbies",
                 "f":"lore",
                 "g":"belles_lettres",
                 "h":"government",
                 "j":"learned",
                 "k":"fiction",
                 "l":"mystery",
                 "m":"science_fiction",
                 "n":"adventure",
                 "p":"romance",
                 "r":"humor"
                 }
  if(output=='dict'):
    genre_sentences=defaultdict(lambda:[])
    for key in fid_sentences:
      genre_sentences[fid_to_genre[key[1]]]+=fid_sentences[key]
    return genre_sentences
  else:#make a list of tuples, where the first value is the key and the second is the sentence
    genre_sentences=[]
    for key in fid_sentences.keys():
      for sentence in fid_sentences[key]:
        genre_sentences.append((fid_to_genre[key[1]],sentence))
    return genre_sentences

def get_sentences_with_fids(corpus=brown,start=32000,end=42000): #Get sentences in given subset and returns them in a dictionary with correlating FID
  fid_sentences = {}
  count = 0
  sentences = set(tuple(row) for row in brown.sents()[start:end])
  sentences_list = [row for row in brown.sents()[start:end]]
  fidCounter = -1
  found = -1
  for fid in corpus.fileids():
    fidCounter+=1
    sent = [row for row in corpus.sents(fileids=fid)]
    #genre_sentences[cat] = defaultdict(lambda:0)
    sent_tuple = set(tuple(row) for row in sent)


    if(sent_tuple.issubset(sentences)):
      if(count==0 and sent!=sentences_list[:len(sent)]):#The first FID will more than likely not start at our start length
        for i in range(0,end-start-len(sent)):
          for j in range(0,len(sent)):
            if(sent[j]!=sentences_list[j+i]):
              break
            if(j==len(sent)-1):
              found=i
          if(found>-1):
            found = i
            break
        if(found<0):
          print("ERROR in get_sentences_with_fid1, set not found")
          return None
        count+=found
      fid_sentences[fid] = sent
      count+=len(sent)

  if(found>-1 or count<(end-start-1)):
    first_fid = set(tuple(row) for row in sentences_list[:found])
    last_fid = set(tuple(row) for row in sentences_list[count:])
    for fid in corpus.fileids():
      if(found==-1 and count>(end-start-1)):
        break
      if fid in fid_sentences.keys():
        continue
      sent = corpus.sents(fileids=fid)
      sent_tuple = set(tuple(row) for row in sent)
      if(first_fid.issubset(sent_tuple)):
        fid_sentences[fid] = sentences_list[:found]
        found = -1
      if(last_fid.issubset(sent_tuple)):
        fid_sentences[fid] = sentences_list[count:]
        count+=len(sentences_list[count:])
    # for i in range(0,len(sent)):
    #   while(sent[i] in sentences and i<len(sent)):
    #     fid_sentences[fid].append(sent[i])
    #     i+=1
  #print(fid_sentences)

  return fid_sentences

# %% [markdown]
# A. PRESS: REPORTAGE
# B. PRESS: EDITORIAL
# C. PRESS: REVIEWS
# D. RELIGION
# E. SKILL AND HOBBIES
# F. POPULAR LORE
# G. BELLES-LETTRES
# H. MISCELLANEOUS: GOVERNMENT & HOUSE ORGANS
# J. LEARNED
# K: FICTION: GENERAL
# L: FICTION: MYSTERY
# M: FICTION: SCIENCE
# N: FICTION: ADVENTURE
# P. FICTION: ROMANCE
# R. HUMOR
# 

# %% [markdown]
# Prepare and download data

# %%
#print(brown.sents()[1].categories())
fid = get_sentences_with_fids()
#cat = convert_fid_to_genre(fid,output='dict')
sentences = convert_fid_to_genre(fid,output='list')
# print(len(sentences))
#print(genre_sentences(sentences))
#print(sentences[6432])
#print(len(sentences))
processed_sentences,vocab = preprocess_data(sentences)

tot=0
for k in processed_sentences.keys():
    tot+=len(processed_sentences[k])
    print(k,len(processed_sentences[k]))
print(tot)
#print(processed_sentences[6432])
print(len(vocab))

# %% [markdown]
# Initialize Models

# %%
#alpha = 1
prob_model = defaultdict(lambda:defaultdict(lambda: defaultdict(lambda: 0))) #topic->w1w2->w3
laplace_model = defaultdict(lambda:defaultdict(lambda: defaultdict(lambda: 0))) #topic->w1w2->w3

# %% [markdown]
# Build a Trigram Model

# %%
for topic in processed_sentences:
  for sentence in processed_sentences[topic]:
    for w1, w2, w3 in trigrams(sentence, pad_right=False, pad_left=False):
      prob_model[topic][(w1, w2)][w3] += 1
      laplace_model[topic][(w1, w2)][w3] += 1

# %% [markdown]
# Transform the counts to probabilities

# %%
#prob_model = copy.deepcopy(model)
for topic in prob_model:
  for w1_w2 in prob_model[topic]:
    total_count = float(sum(prob_model[topic][w1_w2].values()))
    for w3 in prob_model[topic][w1_w2]:
      prob_model[topic][w1_w2][w3] /= total_count

# %% [markdown]
# Implement Laplace Smoothing

# %%
#laplace_model = copy.deepcopy(model)
class TrigramLaplaceSmoothing:
  def __init__(self, vocab,sentences=None,model=None, alpha = 1):
    self.alpha = alpha
    self.vocab = vocab
    if(sentences==None and model==None):
      print("Either Sentences or Model must have a value")
      print(1/0)
    elif(sentences==None):
      self.model = copy.deepcopy(model)
    elif(model==None):
      self.sentences=sentences
      self.model = defaultdict(lambda: defaultdict(lambda: 0))
      for sentence in sentences:
        for w1, w2, w3 in trigrams(sentence, pad_right=False, pad_left=False):
          self.model[(w1, w2)][w3] += 1
    else:
      self.model = copy.deepcopy(model)
      self.sentences=sentences

    self.laplace = defaultdict(lambda: defaultdict(lambda: 0))

  def add_sentence(self, sentence):#add new sentence during validation, and add new vocab to set
    for w1, w2, w3 in trigrams(sentence, pad_right=False, pad_left=False):
      self.model[(w1, w2)][w3] += 1
    for word in sentence:
      self.vocab.add(word)

  def set_model(self,model):#set model
    self.model = copy.deepcopy(model)

  def set_alpha(self,alpha):#set alpha
    self.alpha = alpha

  def make_it_smooth(self):#redoes all laplace if required
    total_words = len(self.vocab)
    for w1_w2 in self.model:
      total_count = float(sum(self.model[w1_w2].values()))
      for w3 in self.model[w1_w2]:
        self.laplace[w1_w2][w3] = (self.model[w1_w2][w3]+self.alpha)/(total_count+self.alpha*total_words)

  def stored_probabilty(self, w1, w2, w3):#Gets stored probability if exists.  It's faster, but if alpha changes, then you need to call make_it_smooth first
    if(self.laplace[w1,w2][w3]==0):
      total_words = len(self.vocab)
      total_count = float(sum(self.model[w1,w2].values()))
    #return (self.model[w1,w2][w3]+self.alpha)/(total_count+self.alpha*total_words)
      self.laplace[w1,w2][w3] = (self.model[w1,w2][w3]+self.alpha)/(total_count+self.alpha*total_words)
    return self.laplace[w1,w2][w3]

  def probability(self, w1, w2, w3):#Gets laplace smoothed probability, unstored so we don't have to worry about wrong alpha values
    total_words = len(self.vocab)
    total_count = float(sum(self.model[w1,w2].values()))
    return (self.model[w1,w2][w3]+self.alpha)/(total_count+self.alpha*total_words)

  def probability_of_sentence(self,sentence):#Finds the probability of the sentence existing
    if(type(sentence) is str):
      sentence = sentence.split()
    if(not type(sentence) is list):
      print("Type error in class TrigramLaplaceSmoothing function probability_of_sentence")
      print(1/0)
    processed_sentence = process_sentence(sentence)
    pre_prob = 0
    for w1, w2, w3 in trigrams(processed_sentence, pad_right=False, pad_left=False):
      pre_prob += np.log(self.probability(w1,w2,w3))#Use logs so we dont have to use small decimals
    return pre_prob

  def predict_next_word_in_sentence(self,sentence):#Guesses the next word in the sentence
    if(type(sentence) is str):
      sentence = sentence.split()
    if(not type(sentence) is list):
      print("Type error in class TrigramLaplaceSmoothing function predict_next_word_in_sentence")
      print(1/0)
    processed_sentence = process_sentence(sentence)[:-1]
    pre_prob = 0
    #print(processed_sentence)
    for w1, w2, w3 in trigrams(processed_sentence, pad_right=False, pad_left=False):
      #print(w1,w2,w3,pre_prob, self.model[w1,w2][w3])
      pre_prob += np.log(self.probability(w1,w2,w3))
    next_word_dict = {}
    #print(pre_prob,processed_sentence[-2],processed_sentence[-1])
    for next_word in self.vocab: #any word could be next
      next_word_dict[next_word] = pre_prob+np.log(self.probability(processed_sentence[-2],processed_sentence[-1],next_word))

    #print(max(next_word_dict,key=next_word_dict.get),next_word_dict[max(next_word_dict,key=next_word_dict.get)])
    return max(next_word_dict,key=next_word_dict.get)

fiction = TrigramLaplaceSmoothing(model=laplace_model['fiction'],vocab=vocab['fiction'],alpha=1)
learned = TrigramLaplaceSmoothing(model=laplace_model['learned'],vocab=vocab['learned'],alpha=1)
#test.make_it_smooth()
print(fiction.predict_next_word_in_sentence("The book"))
print(learned.predict_next_word_in_sentence("The book"))
print(fiction.probability_of_sentence("The book was complete"))
print(learned.probability_of_sentence("The book was complete"))




# %% [markdown]
# Predict Next Word

# %%
def predict_next_word(w1, w2):
  # TODO: Implement the prediction logic
  # Example: return max(model[w1, w2], key=model[w1, w2].get)
  return max(laplace_model[w1,w2],key=laplace_model[w1,w2].get)
  #pass
print(predict_next_word('The','Fulton') == 'County')

# %% [markdown]
# Implement a classifier using the trigram model -- have a simple threshold anything >=0.5 is yes or else 0

# %%
class TrigramClassifier:
  def __init__(self, corpus=brown,range=(32000,42000),trn_val_test=(0.6,0.2,0.2),alpha=1):
    self.trn_val_test = trn_val_test #Should add up to 1

    self.sentences = convert_fid_to_genre(get_sentences_with_fids(corpus,range[0],range[1]),output='list')#The sentences provided by the corpus
    np.random.shuffle(self.sentences) #shuffled

    #Splits up training, testing, and validation data
    trn = self.sentences[:int(len(self.sentences)*trn_val_test[0])]
    val = self.sentences[int(len(self.sentences)*trn_val_test[0]):int(-len(self.sentences)*trn_val_test[2])]
    test = self.sentences[int(-len(self.sentences)*trn_val_test[2]):]
    self.training_sentences,self.training_vocab = preprocess_data(trn)
    self.validation_sentences,self.validation_vocab = preprocess_data(val)
    self.testing_sentences,self.testing_vocab = preprocess_data(test)

    #Classes, global alpha, and whether or not validation is completed
    self.alpha = alpha
    self.classes = self.training_sentences.keys()
    self.validation_complete = False

    #Each classifieer in the model
    self.classifiers = {}
    for c in self.classes:
      self.classifiers[c] = TrigramLaplaceSmoothing(sentences = self.training_sentences[c],vocab = self.training_vocab[c])

  def print_sorted_dict(self, dict):#prints a guesses sorted
    word_list = []
    for key in dict:
      word_list.append((key,dict[key]))
      word_list.sort(key=lambda item:item[1],reverse=True)
    for item in word_list:
      print(item)

  def predict_genres_of_sentence_probabilities(self,sentence):#Returns a dict of each genre and the corresponding prob
    if(not(type(sentence) is str or type(sentence) is list)):
      print("Type invalid in predict_genre_of_sentence; unsuppported type: "+str(type(sentence)))
      print(1/0)

    genre_probs = {}
    total_sents = 0
    total_vocab = set()
    for category in self.training_sentences:
      total_sents+=len(self.training_sentences[category])
      total_vocab.update(self.training_vocab[category])
    if(self.validation_complete):#If validation complete, then self.validation is a part of the model vocab and sentences
      for category in self.validation_sentences:
        total_sents+=len(self.validation_sentences[category])
        total_vocab.update(self.validation_vocab[category])

    for c in self.classifiers:#I could change the global alpha, but it as 1 has done well
      if(self.validation_complete):
        genre_probs[c] = self.classifiers[c].probability_of_sentence(sentence) + np.log((len(self.training_sentences[c])+len(self.validation_sentences[c])+self.alpha)/(total_sents+len(total_vocab)*self.alpha))
      else:
        genre_probs[c] = self.classifiers[c].probability_of_sentence(sentence) + np.log((len(self.training_sentences[c])+self.alpha)/(total_sents+len(total_vocab)*self.alpha))

    return genre_probs


  def predict_genres_of_sentence(self,sentence):#Returns the classification based off of the highest prob dict
    genre_guesses = self.predict_genres_of_sentence_probabilities(sentence)
    if(genre_guesses==None):
      print("predict_genres_of_sentence_probabilities returned no guesses")
      return ""
    return max(genre_guesses,key=genre_guesses.get)

  def test_sentence(self, sentence, genre):#Given a sentence and a genre, returns if the classification is right
    if(self.predict_genres_of_sentence(sentence)==genre):
      return True
    return False


  def evaluate(self,level='test'):#Does testing on the testing slice
    total = 0
    scores = {}
    eval_sentences = {}
    if(level=='test'):
      eval_sentences = self.testing_sentences
    elif(level=='val'):
      eval_sentences = copy.deepcopy(self.validation_sentences)
      for genre in self.training_sentences:
        eval_sentences[genre]+=self.training_sentences[genre]
    elif(level=='train'):
      eval_sentences=self.training_sentences
    else:
      print("Invalid level provided in Class TrigramClassifier method Evaluate")
      print(1/0)

    for genre in eval_sentences:
      scores[genre] = [0,0,0,0]#TP FN FP TN

    for genre in eval_sentences:
      for sentence in eval_sentences[genre]:
        total+=1
        guess = self.predict_genres_of_sentence(sentence)
        for g in eval_sentences:
          if(g==guess and guess==genre):
            scores[g][0]+=1#TP increase
          elif(g!=guess and guess==genre):
            scores[g][3]+=1#TN increase
          elif(g==genre and guess!=genre):
            scores[g][1]+=1#FN increase
          elif(g==guess and guess!=genre):
            scores[g][2]+=1#FP increase
          else:
            print(g,guess,genre)
            print(1/0)
    return scores


  def reset_validation(self):
    if(self.validation_complete):
      for genre in self.classifiers:
        self.classifiers[genre] = TrigramLaplaceSmoothing(vocab=self.training_vocab[genre],sentences=self.training_sentences[genre],alpha=1)
      self.validation_complete = False

  def validation(self,alphas=[0.001,0.005,0.01,0.25,0.5,0.8,1]):#Finds the best alpha and adds validation sentences into training model
    if(self.validation_complete):
      print("Validation has already been completed, if you would like to do it again, please call the reset_validation method prior to running validation")
      return

    total_count = 0
    for genre in self.classifiers:
      total_count+=len(self.validation_sentences[genre])

    #THE FOLLOWING CODE GETS ALL ALPHA COMBINATIONS GIVEN GENRE AND ALPHA LIST, the number of combinations grow exponentially with class size, but thankfully we only have 2
    max_alpha = {}
    combinations = [0]*len(self.classifiers)
    for i in range(0,len(alphas)**len(self.classifiers)):
      key = [alphas[a] for a in combinations]
      for j in range(0,len(combinations)):
        combinations[j]+=1
        if(combinations[j]%len(alphas)==0 and j+1<len(combinations)):
          combinations[j]=0
          continue
        break
      max_alpha[tuple(key)] = 0

    #Finds best alpha combination
    for a in max_alpha:
      i=0
      for genre in self.classifiers:
        self.classifiers[genre].set_alpha(a[i])
        i+=1
      correct = 0
      for genre in self.classifiers:
        for sentence in self.validation_sentences[genre]:
          if(self.test_sentence(sentence,genre)):
            correct+=1
      max_alpha[a]=correct

    #Sets best alpha for classes
    alpha = max(max_alpha,key=max_alpha.get)
    i=0
    for genre in self.classifiers:
      print(genre,alpha[i],max_alpha[alpha])
      self.classifiers[genre].set_alpha(alpha[i])
      i+=1

    #Add validation data into classifiers
    for genre in self.classifiers:
      for sentence in self.validation_sentences[genre]:
        self.classifiers[genre].add_sentence(sentence)
    self.validation_complete = True




# %% [markdown]
# 

# %% [markdown]
# Split your dataset into training, validation and testing sets - Do random sampling without replacement to create these sets, use basic python and numpy and not sklearn or other lib

# %%
#test = TrigramClassifier(corpus=brown,range=(32000,42000),trn_val_test=(0.9,0.075,0.025),alpha=1)
test = TrigramClassifier(corpus=brown,range=(32000,42000),trn_val_test=(0.8,0.01,0.1),alpha=1)
#test = TrigramClassifier()

# %% [markdown]
# Train your classifier and run predictions on the validation set per epoch
# 

# %%
#test.validation([0.001,0.00125,0.0015,0.00175,0.002,0.005,0.006,0.0065,0.0075,0.008,0.01,0.1,0.2,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1])
trainScore = test.evaluate('train')
test.validation()
valScore = test.evaluate('val')
#test.classifiers['learned'].set_alpha(.0075)
testScores = test.evaluate()

# %% [markdown]
# Evaluate the performance of your model (accuracy, precision, recall, F1-score) -- check model performance on train and validation only

# %%
def confusion_matrix(scores):
  for key in scores:
    accuracy = float((scores[key][3]+scores[key][0])/(scores[key][0]+scores[key][1]+scores[key][2]+scores[key][3]))
    print("'"+key+"' overall accuracy:{0}".format(accuracy))
    print("\tTP:{0}\tFN:{1}".format(scores[key][0],scores[key][1]))
    print("\tFP:{0}\tTN:{1}".format(scores[key][2],scores[key][3]))
    precision = float(scores[key][0]/(scores[key][0]+scores[key][2]))
    recall = float(scores[key][0]/(scores[key][0]+scores[key][1]))
    f1 = float(scores[key][0]/(scores[key][0]+0.5*(scores[key][1]+scores[key][2])))
    print("Precision: {0}".format(precision))
    print("Recall: {0}".format(recall))
    print("F1-Score: {0}".format(f1))
    print("")

scores1 = {'fiction': [73, 20, 47, 110], 'learned': [110, 47, 20, 73]}
scores2 = {'fiction': [661, 136, 426, 777], 'learned': [777, 426, 136, 661]}

# %% [markdown]
# Stores these values in a list, to use later with matplotlib to show your train and validation curve

# %%
print("Train Eval")
confusion_matrix(trainScore)
print("Val Eval")
confusion_matrix(valScore)

# %% [markdown]
# Evaluate performance on test set only Once

# %%
print("Test Eval")
confusion_matrix(testScores)


