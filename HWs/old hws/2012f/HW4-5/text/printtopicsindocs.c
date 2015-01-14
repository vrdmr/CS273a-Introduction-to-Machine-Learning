/*---------------------------------------------------
 * file:    topicsindocs.c
 * purpose: compute topicsindocs
 * usage:   topicsindocs
 * inputs:  Ndt.txt, docs.txt
 * outputs: topicsindocs.txt
 * author:  David Newman, newman@uci.edu
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
  int i, t, d, **dt, dtot;
  //double alpha;
  char **doc;
  FILE *fp;
  int *prob_t_given_d, *indx;

  int T; // number of topics
  int D; // number of docs
  
  if (argc != 1) {
    fprintf(stderr, "usage: %s\n", argv[0]);
    exit(-1);
  }
  
  dt = read_sparse("Ndt.txt",&D,&T);       assert(D>0); assert(T>0);
  fp = fopen("docs.txt","r");             assert(fp);
  doc = (char**)malloc(D*sizeof(char*));  assert(doc);
  for (d = 0; d < D; d++) {
    doc[d] = (char*)malloc(100*sizeof(char));
    fscanf(fp,"%s",doc[d]);
  }
  fclose(fp);

  prob_t_given_d = ivec(T);
  indx           = ivec(T);
  
  for (d = 0; d < D; d++) {

    printf("<%s> ", doc[d]);

    dtot = 0;
    for (t = 0; t < T; t++) { prob_t_given_d[t] = dt[d][t]; dtot += dt[d][t]; }
    isort(T, prob_t_given_d, -1, indx);
   
    for (i = 0; i < 4; i++) {
      if ((1.0*dt[d][indx[i]])/dtot > 0.1)
	printf("t%d ", indx[i]+1);
    }
    printf("\n");
    
  }
  
  return 0;
}
