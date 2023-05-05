############ Welcome to HW7 ############
# TODO: Andrew-id: 


# Imports
# Don't import any other library

import argparse
import numpy as np
import sys
from utils import make_dict, parse_file
import logging

# Setting up the argument parser
# don't change anything here

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .txt file')
parser.add_argument('index_to_word', type=str,
                    help='path to index_to_word.txt file')
parser.add_argument('index_to_tag', type=str,
                    help='path to index_to_tag.txt file')
parser.add_argument('init', type=str,
                    help='path to store the hmm_init.txt (pi) file')
parser.add_argument('emission', type=str,
                    help='path to store the hmm_emission.txt (A) file')
parser.add_argument('transition', type=str,
                    help='path to store the hmm_transition.txt (B) file')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


# Hint: You might find it useful to define functions that do the following:
# 1. Calculate the init matrix
# 2. Calculate the emission matrix
# 3. Calculate the transition matrix
# 4. Normalize the matrices appropriately

def matching(dict, list):
    for l in list:
        for i in range(len(l)):
            if l[i] in dict:
                l[i]= dict[l[i]]
    return list
                
def normalisation(x):
    return x/ np.sum(x, axis=0)

def normalised(x):
    return x/ np.sum(x)

def prob_matrices(state, labels, tag_dict, word_dict):
    emission = np.ones((len(tag_dict),len(word_dict)))
    transition = np.ones((len(tag_dict),len(tag_dict)))
    init = np.ones((len(tag_dict)))
    for i,j in zip(state, labels):
        for k in range(len(i)):
            emission[i[k],j[k]]+=1
        for k in range(len(i)-1):
            transition[i[k],i[k+1]]+=1
    for s in state:
        init[s[0]]+=1        
    return normalisation(emission.T), normalisation(transition.T), normalised(init)




# TODO: Complete the main function
def main(args):

    # Get the dictionaries
    word_dict = make_dict(args.index_to_word)
    tag_dict = make_dict(args.index_to_tag)


    # Parse the train file
    # Suggestion: Take a minute to look at the training file,
    # it always hels to know your data :)
    sentences, tags = parse_file(args.train_input)

    labels, states = matching(word_dict,sentences), matching(tag_dict,tags)
     
    
    Emission, Transition, init = prob_matrices(states, labels, tag_dict, word_dict)
    #print("Emission :" , Emission.T)
    #print("Transition :", Transition.T)
    #print("init :", init.reshape((len(init),1)))

    # logging.debug(f"Num Sentences: {len(sentences)}")
    # logging.debug(f"Num Tags: {len(tags)}")
    
    
    # Train your HMM
   # init = # TODO: Construct your init matrix
    #emission = # TODO: Construct your emission matrix
    #transition = # TODO: Construct your transition matrix

    # Making sure we have the right shapes
    ##logging.debug(f"init matrix shape: {init.shape}")
    ##logging.debug(f"emission matrix shape: {emission.shape}")
    ##logging.debug(f"transition matrix shape: {transition.shape}")


    # Saving the files for inference
    # We're doing this for you :)
    np.savetxt(args.init, init.reshape((len(init),1)))
    np.savetxt(args.emission, Emission.T)
    np.savetxt(args.transition, Transition.T)

    return 

# No need to change anything beyond this point
if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    main(args)