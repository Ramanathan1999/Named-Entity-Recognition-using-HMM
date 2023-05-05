############ Welcome to HW7 ############
# TODO: Andrew-id: 


# Imports
# Don't import any other library

import numpy as np
import sys
from utils import make_dict, parse_file, get_matrices, write_predictions, write_metrics
import argparse
import logging
import time

#start = time.time()

# Setting up the argument parser
# don't change anything here
parser = argparse.ArgumentParser()
parser.add_argument('validation_input', type=str,
                    help='path to validation input .txt file')
parser.add_argument('index_to_word', type=str,
                    help='path to index_to_word.txt file')
parser.add_argument('index_to_tag', type=str,
                    help='path to index_to_tag.txt file')
parser.add_argument('init', type=str,
                    help='path to the learned hmm_init.txt (pi) file')
parser.add_argument('emission', type=str,
                    help='path to the learned hmm_emission.txt (A) file')
parser.add_argument('transition', type=str,
                    help='path to the learned hmm_transition.txt (B) file')
parser.add_argument('prediction_file', type=str,
                    help='path to store predictions')
parser.add_argument('metric_file', type=str,
                    help='path to store metrics')                    
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')



# Hint: You might find it helpful to define functions 
# that do the following:
# 1. Calculate Alphas
# 2. Calculate Betas
# 3. Implement the LogSumExpTrick
# 4. Calculate probabilities and predictions

def matching(dict, list):
    
    for l in list:
        for i in range(len(l)):
            if l[i] in dict:
                l[i]= dict[l[i]]
    return list


def predictedtags(predict,tag_dict):
    taglist = []
    keys = list(tag_dict.keys())
    
    
    for i in predict:
        tag = keys[i]
        taglist.append(tag)
        
    
    return taglist
    

def forward(word_dict,tag_dict, labelseq, init ,emit,transition):
 
    alpha = np.zeros((len(tag_dict),len(labelseq)))
   # beta = np.zeros((len(tag_dict),len(fliplabelseq)))
    
    for i in range(len(labelseq)):
        if i == 0:
            alpha[:,i] = init * emit[:,labelseq[i]].T
            
            print(emit[:,labelseq[i]])
            #beta[:,i] = 1
            
        else:
           # w_alpha = transition.T * emit[:,labelseq[i]]
            w_alpha = transition.T * alpha[:,i-1].T
           # print("w_alpha:", w_alpha)
           
            #alpha[:,i] = np.matmul(w_alpha, alpha[:,i-1])
            alpha[:,i] = np.matmul(w_alpha, emit[:,labelseq[i]])
            
           # w_beta = transition * emit[:, fliplabelseq[i]]
           #         zaa print("w_beta:", w_beta)
           # beta[:,i]= np.matmul(w_beta, beta[:,i-1])
            
    return alpha.T

def backward(word_dict,tag_dict,labelseq,init,emit,transition):
    
    beta = np.zeros((len(tag_dict),len(labelseq)))
    beta[:,-1]=1
    
    for i in range(len(labelseq)):
            y =len(labelseq)-2-i 
           # w_beta = transition * emit[:,labelseq[i]]
            w_beta = transition * beta[:,y+1].T
           # print("w_beta:",w_beta)
          #  beta[:,y]= np.matmul(w_beta, beta[:,y+1])
            beta[:,y]= np.matmul(w_beta, emit[:,labelseq[i]])
            
    return beta.T
                                 
                                 
    


# LOOK AT THIS SECOND
def expsum(vector):
    m= np.max(vector)
    expdiff = 0
    for i in range(len(vector)):
        expdiff += np.exp(vector[i]-m)
    return m + np.log(expdiff)
        
    
    
def forwardbackwardlog(word_dict,tag_dict, labelseq, fliplabelseq,init,emit,transition):
    alpha = np.zeros((len(labelseq),len(tag_dict)))
    beta = np.zeros((len(fliplabelseq),len(tag_dict)))
    poweralpha = np.zeros((len(tag_dict)))
    powerbeta = np.zeros((len(tag_dict)))
    emitt = emit.T
    
    for i in range(len(labelseq)):
        if i == 0:
            alpha[i,:] = np.log(init.T) + np.log(emit[:,labelseq[i]].T)
            beta[i,:] = 0
        
        else:
            for j in range(len(tag_dict)):            
                for k in range(len(tag_dict)):
                    poweralpha[j] = alpha[i-1][k] + np.log(transition[k][j])
                    powerbeta[j] = beta[i-1][k] + np.log(transition[j][k]) +np.log(emit[k][fliplabelseq[i]])
                alpha[i,j] = np.log(emitt[labelseq[i]][j]) + expsum(poweralpha)
            beta[i,:] = expsum(powerbeta)
    
    return alpha, beta[::-1,:]


    
def forwardlog(word_dict,tag_dict, labelseq,init,emit,transition):
    alpha = np.zeros((len(labelseq),len(tag_dict)))                
    #poweralpha = np.zeros((len(tag_dict)))           
    emitt = emit.T        
    for i in range(len(labelseq)):
        if i == 0:
           alpha[i,:] = np.log(init.T) + np.log(emit[:,labelseq[i]].T) 
        else:
            for j in range(len(tag_dict)): 
                pn = []
                for k in range(len(tag_dict)):
                    poweralpha = alpha[i-1][k] + np.log(transition[k][j]) 
                    pn.append(poweralpha)
                alpha[i,j] = np.log(emitt[labelseq[i]][j]) + expsum(pn)
               # print("alpha log at ", i ," th word :", alpha)
                
    return alpha

def backwardlog(word_dict,tag_dict, labelseq, init,emit,transition):
    beta = np.zeros((len(labelseq),len(tag_dict)))                
   # powerbeta = np.zeros((len(tag_dict))) 
    beta[len(labelseq)-1,:] = 0 
    
    for i in range(len(labelseq)):
        y = len(labelseq)-2-i
       # pn=[]
        for j in range(len(tag_dict)): 
            pn = []
            for k in range(len(tag_dict)):
                powerbeta = beta[y+1][k] + np.log(transition[j][k]) +np.log(emit[k][labelseq[y+1]])
                pn.append(powerbeta)
            beta[y,j] = expsum(pn) 
            
    return beta








def forward1 (word_dict,tag_dict, labelseq, init,emit,transition):
    alpha = np.zeros((len(labelseq),len(tag_dict)))
   # poweralpha = np.zeros((len(tag_dict)**2))
    emitt = emit.T
    for i in range(len(labelseq)):
        if i == 0:
            alpha[i,:] = (init.T) * (emit[:,labelseq[i]].T) 
        else:
            for j in range(len(tag_dict)):            
                for k in range(len(tag_dict)):
                      #poweralpha[j] += alpha[i-1][k] * transition[k][j]
                      #print("poweralpha at ", i , " th step :", poweralpha)
                      alpha[i,j] += emitt[labelseq[i]][j] * alpha[i-1][k] * transition[k][j]
               # print("alpha at ", i , " th step :", alpha)
    return alpha

def backward1 (word_dict,tag_dict, labelseq, init,emit,transition):
    beta = np.zeros((len(labelseq),len(tag_dict)))                
    beta[-1,:] = 1
    emitt = emit.T
    
    for i in range(len(labelseq)-1):
        y = len(labelseq)-2-i
        for j in range(len(tag_dict)):            
            for k in range(len(tag_dict)):
                beta[y,k] += beta[y+1][k] * (transition[j][k]) * (emitt[labelseq[y+1]][k])
                #print("beta at ", i , " th step :", beta)
                    
    return beta
    
    
    
    
    

# TODO: Complete the main function
def main(args):

    # Get the dictionaries
    word_dict = make_dict(args.index_to_word)
    
    # print("word dict:", word_dict)
   
    tag_dict = make_dict(args.index_to_tag)
    #print("tag dict:", tag_dict)

    # Parse the validation file
    sentences, tags = parse_file(args.validation_input)
    
    print("sentences :", sentences)
    #print("tags :", tags)
    
    labels, states = matching(word_dict,sentences), matching(tag_dict,tags)
    print("sentences after:", sentences)
    #rev = list(reversed(labels))
    print("labels :" , labels)
    #print("reversed labels :", rev)
    #print("states :" ,states)

    # Load your learned matrices
    # Make sure you have them in the right orientation
    init, emission, transition = get_matrices(args)
    #print("init :", init)
    #print("emission :", emission)
    #print("transmition :", transition)
    predicted_tags = []
    avg_log_likelihood_list = [] 
    
    for labelseq in labels:
        
        alpha= forward1(word_dict,tag_dict, labelseq, init, emission, transition)
        beta = backward1(word_dict, tag_dict, labelseq, init, emission, transition)
        #print("alpha 1:",alpha)
        #print("beta 1", beta)
        probabilities = alpha * beta
        predict = np.argmax(probabilities.T, axis=0)
        #print(predict)
        
    #    alphalog, betalog = forwardbackwardlog(word_dict,tag_dict, labelseq, list(reversed(labelseq)) ,init,emission,transition)
    
        alphalog= forwardlog(word_dict,tag_dict, labelseq, init, emission, transition)
        betalog = backwardlog(word_dict, tag_dict, labelseq, init, emission, transition)
        probabilitieslog = (alphalog) + (betalog)
       # print(alphalog)
               
        predictlog = np.argmax(probabilitieslog.T, axis=0)
        #print("log predict", predictlog)
        
        palpha = np.exp(alphalog)
        #print("palpha :", palpha)
        
        
        log_likelihood = expsum(alphalog[-1,:])
        
        log_likelihood1 = np.log(np.sum(np.log(alpha[-1,:])))
        #print("log likelihood :", log_likelihood1)
        #print("log log likelihood :", log_likelihood)
        
        avg_log_likelihood_list.append(log_likelihood)
        
        predicted_tags_ = predictedtags(predictlog, tag_dict)
        predicted_tags.append(predicted_tags_)
    
    print(predicted_tags)
        
    avg_log_likelihood = sum(avg_log_likelihood_list)/len(avg_log_likelihood_list)
    print(avg_log_likelihood)
    
    
    # sentex = predicted_tags(labels,word_dict)
    # print(sentex)
    
    sentences, tags = parse_file(args.validation_input)

    
    # TODO: Conduct your inferences
    
    
    # TODO: Generate your probabilities and predictions

  
    ##predicted_tags = #TODO: store your predicted tags here (in the right order)
    ##avg_log_likelihood = # TODO: store your calculated average log-likelihood here
    
    accuracy = 0 # We'll calculate this for you

    # Writing results to the corresponding files.  
    # We're doing this for you :)
    print("Tags :",tags)
    print(" Predicted Tags :",predicted_tags)
    print("Sentences :",sentences)
    accuracy = write_predictions(args.prediction_file, sentences, predicted_tags, tags)
    write_metrics(args.metric_file, avg_log_likelihood, accuracy)
    print(accuracy)

    return

if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    main(args)

#end = time.time()
#print("time :", end-start)
