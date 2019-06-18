# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:02:35 2019
https://github.com/mammothb/symspellpy/tree/master/symspellpy
https://www.kaggle.com/yk1598/symspell-spell-corrector/code
https://davidkoh.me/understanding-the-basics-of-text-correction-in-the-context-of-ecommerce-search/
@author: shreyas.shandilya
"""
import re
import pandas as pd
from difflib import SequenceMatcher
fname = r"C:\Users\shreyas.shandilya\Desktop\frequency_dictionary_en_82_765.txt"
file_db=r'C:\Users\shreyas.shandilya\Desktop\KPI_db_updated.csv'
def damerau_levenshtein_distance(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences.
    This method has not been modified from the original.
    Source: http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/
    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.
    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.
    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.
    >>> dameraulevenshtein('ba', 'abc')
    2
    >>> dameraulevenshtein('fee', 'deed')
    2
    It works with arbitrary sequences too:
    >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = (oneago, thisrow, [0] * len(seq2) + [x + 1])
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]

def get_deletes_list(w, delete_distance_max):
    #returns all possible combinations of the string with a maximum of delete_distance_max number of deletes
    deletes_list = []
    queue = [w]
    for x in range(delete_distance_max):
        temporary_queue = []#for future rounds of deletion of characters
        for word in queue:
            if len(word)>delete_distance_max:
                for c in range(len(word)):
                    word_del_c = word[:c] + word[c+1:]
                    if word_del_c not in deletes_list:
                        deletes_list.append(word_del_c)
                    if word_del_c not in temporary_queue:
                        temporary_queue.append(word_del_c)
        queue = temporary_queue
    return deletes_list

    
def create_dictionary_entry(w,frequency):
    #the dictionary will be in the format of {"key":[[autocorrected word 1, autocorrected word 2, etc],frequency]}
    #frequency refers to the number  of times a word occured in the training corpus
    #the keys are the strings created by deleting delete_distance_max number of characters from a word
    #Creating the dictionary from the corpus used to check for spellings
    if w not in dictionary:
        dictionary[w] = [[w],frequency]   
    else:
        dictionary[w][0].append(w)
        dictionary[w] = [dictionary[w][0],frequency]  
    deletes_list = get_deletes_list(w,2)
    #thr following lines of code adds all strings created by deleting a maximum of delete_distance_max number of characters from the word
    for word in deletes_list:
        if word not in dictionary:
            dictionary[word] = [[w],0]#this implies that word does not occur once  in the training corpus
        else:
            dictionary[word][0].append(w)

def create_dictionary(fname):
    #Populates the dictionay with all the words occuring in the training corpus
    total_frequency =0
    with open(fname) as file:
        for line in file:
            create_dictionary_entry(line.split()[0],line.split()[1])
    for keyword in dictionary:
        total_frequency += float(dictionary[keyword][1])
    for keyword in dictionary:
        dictionary[keyword].append(float(dictionary[keyword][1])/float(total_frequency))


def get_suggestions(w):
    search_deletes_list = get_deletes_list(w,2)
    search_deletes_list.append(w)
    candidate_corrections = []
    #Does not correct words which are existing in the dictionary and that has a frequency greater than 1 in the training corpus
    if (w in dictionary and int(dictionary[w][1])>0) or (w in dictionary_db['words'].values):#add condition if the word exists in the database or not
        return w
    else:
        for query in search_deletes_list:
            if query in dictionary:
                for suggested_word in dictionary[query][0]:
                    word_len=len(w)
                    sug_len=len(suggested_word)
                    edit_distance = float(damerau_levenshtein_distance(w, suggested_word))
                    frequency = float(dictionary[suggested_word][1])
                    score = frequency*(0.003**(edit_distance))
                    length=max(len(w),len(suggested_word))
                    score2 = SequenceMatcher(None, w, suggested_word).ratio()
                    score3 = 1.0-(edit_distance/length)
                    score4= (word_len+sug_len-edit_distance)/(word_len+sug_len)
                    #A similarity value from 0 to 1.0 (1 - (distance / length)), -1 if\distance is negative. Here length is the length of the longer string           
                    candidate_corrections.append((suggested_word,frequency,score,edit_distance,score2,score3,score4))
#minimal edit distance and maximum similarity  can be used as measure of correctness of a particular spelling candidate
        return candidate_corrections

    
#def get_corrections(w):
#    try:
#        return get_suggestions(w)
#    except:
#        return "no result"
def flag_(word):
    if word in dictionary_db['words'].values:
        return 1
    else:
        return 0
    
def spell_check(sent):
    sent=sent.split(' ')
    correction_=[]
    abb=re.compile(r'[[a-zA-z]\.]+')
    for word in sent:
            if word.isdigit():#check for numbers
                correction_.append(word)
            elif word.isupper() or bool(abb.match(word)) :#check for the presence of abbreviation
                correction_.append(word)
            else:
                candidates_=get_suggestions(word)
                if type(candidates_)==list:
                    candidate_df=pd.DataFrame(candidates_[:10],columns=['candidate','corpus_frequency','score','edit_distance','seq_match_score','similarity_score','similarity_score2'])
                    candidate_df.loc[:,'flag']=candidate_df['candidate'].apply(flag_)
                    flag=candidate_df['flag'].sum() 
                    if flag==0:
                        candidate_df=candidate_df.sort_values(by='score',ascending=False)
                        correction=candidate_df.iloc[0]['candidate']
#                        print(candidate_df)
                    else:
                        candidate_df['score_final']=candidate_df['similarity_score2']+candidate_df['flag']
                        candidate_df=candidate_df.sort_values(by='score_final',ascending=False)
                        correction=candidate_df.iloc[0]['candidate']
#                        print(candidate_df)
                    correction_.append((correction,candidate_df))
                else:
                    correction_.append(candidates_)
#    correction_=' '.join(correction_)
    return correction_
dictionary={}
data=pd.read_csv(file_db)
dictionary_db=pd.Series(list(data.columns)[1:8])
dictionary_db=dictionary_db.append(data['Brand'][:],ignore_index=True)
dictionary_db=dictionary_db.append(data['Country'][:],ignore_index=True)
dictionary_db=dictionary_db.apply(str.lower)
dictionary_db=dictionary_db.apply(str.split)
lst=[]
for i in dictionary_db.values:
    lst.extend(i)
dictionary_db=pd.Series(lst)
dictionary_db= pd.value_counts(dictionary_db).to_frame().reset_index()
dictionary_db=dictionary_db.rename(columns={'index':'words',0:'frequency'})
dictionary_db.apply(lambda x: create_dictionary_entry(x['words'], x['frequency']), axis=1)       
        
create_dictionary(fname)
    