/*---------------------------------------------------
 * file:    topics.c
 * purpose: print topics
 * usage:   topics
 * inputs:  Nwt.txt, vocab.txt
 * author:  newman@uci.edu
 *-------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "tlib.h"

/*==========================================
 * main
 *========================================== */
int main(int argc, char* argv[])
{
  int i, t, w, **Ntw, *Nt, dum;
  char **word;
  FILE *fp;
  int *prob_w_given_t, *indx;

  int T; // number of topics
  int W; // number of unique words
  
  if (argc != 1) {
    fprintf(stderr, "usage: %s\n", argv[0]);
    exit(-1);
  }
  
  Ntw = read_sparse_trans("Nwt.txt",&T,&W); assert(T>0); assert(W>0);
  fp = fopen("vocab.txt","r");              assert(fp);
  word = (char**)malloc(W*sizeof(char*));   assert(word);
  for (w = 0; w < W; w++) {
    word[w] = (char*)malloc(100*sizeof(char));
    dum = fscanf(fp,"%s",word[w]);
  }
  fclose(fp);

  prob_w_given_t = ivec(W);
  indx           = ivec(W);
  Nt             = ivec(T);

  for (t = 0; t < T; t++) {
    Nt[t] = 0;
    for (w = 0; w < W; w++) Nt[t] += Ntw[t][w];
  }
  
  for (t = 0; t < T; t++) {
    
    for (w = 0; w < W; w++) prob_w_given_t[w] = Ntw[t][w];
    isort(W, prob_w_given_t, -1, indx);
    
    printf("[t%d] ", t+1);
    for (i = 0; i < 8; i++) printf("%s ", word[indx[i]]);
    printf("...\n");
    
  }
  
  return 0;
}
