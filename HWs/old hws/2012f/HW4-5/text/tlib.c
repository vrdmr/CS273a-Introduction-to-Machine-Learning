/*---------------------------------------------------
 * file:    tlib.c
 * purpose: various utility routines
 * author:  newman@uci.edu
 *-------------------------------------------------*/

// grep //$ tlib.c

#include "tlib.h"

/*------------------------------------------
 * private to util.c
 *------------------------------------------ */

static int icomp(const void *, const void *); /* comparison for isort */
static int dcomp(const void *, const void *); /* comparison for dsort */
static int    *icomp_vec;                     /*  data used for isort */
static double *dcomp_vec;                     /*  data used for dsort */

/*------------------------------------------
 * allocation routines
 * imat
 * dmat
 * ivec
 * dvec
 *------------------------------------------ */

int **imat(int nr, int nc) //
{
  int N = nr*nc;
  int *tmp = (int*) calloc(N,sizeof(int));
  int **x  = (int**)calloc(nr,sizeof(int*));
  int r;
  assert(tmp);
  assert(x);
  for (r = 0; r < nr; r++) x[r] = tmp + nc*r;
  return x;
}
  
void free_imat(int **x) //
{
  free(x[0]);
  free(x);
}

double **dmat(int nr, int nc) //
{
  int N = nr*nc;
  double *tmp = (double*) calloc(N,sizeof(double));
  double **x  = (double**)calloc(nr,sizeof(double*));
  int r;
  assert(tmp);
  assert(x);
  for (r = 0; r < nr; r++) x[r] = tmp + nc*r;
  return x;
}

void free_dmat(double **x) //
{
  free(x[0]);
  free(x);
}

float **fmat(int nr, int nc) //
{
  int N = nr*nc;
  float *tmp = (float*) calloc(N,sizeof(float));
  float **x  = (float**)calloc(nr,sizeof(float*));
  int r;
  assert(tmp);
  assert(x);
  for (r = 0; r < nr; r++) x[r] = tmp + nc*r;
  return x;
}

void free_fmat(float **x) //
{
  free(x[0]);
  free(x);
}

int *ivec(int n) //
{
  int *x = (int*)calloc(n,sizeof(int));
  assert(x);
  return x;
}
  
double *dvec(int n) //
{
  double *x = (double*)calloc(n,sizeof(double));
  assert(x);
  return x;
}

float *fvec(int n) //
{
  float *x = (float*)calloc(n,sizeof(float));
  assert(x);
  return x;
}

/*------------------------------------------
 * vector routines
 * imax
 * dmax
 *------------------------------------------ */

int imax(int n, int *x) //
{
  int i, xmax=x[0];
  for (i = 0; i < n; i++) xmax = MAX(xmax,x[i]);
  return xmax;
}

double dmax(int n, double *x) //
{
  int i;
  double  xmax=x[0];
  for (i = 0; i < n; i++) xmax = MAX(xmax,x[i]);
  return xmax;
}

int imin(int n, int *x) //
{
  int i, xmin=x[0];
  for (i = 0; i < n; i++) xmin = MIN(xmin,x[i]);
  return xmin;
}

double dmin(int n, double *x) //
{
  int i;
  double  xmin=x[0];
  for (i = 0; i < n; i++) xmin = MIN(xmin,x[i]);
  return xmin;
}

int isum(int n, int *x) //
{
  int i, xsum=0;
  for (i = 0; i < n; i++) xsum += x[i];
  return xsum;
}

double dsum(int n, double *x) //
{
  int i;
  double xsum=0;
  for (i = 0; i < n; i++) xsum += x[i];
  return xsum;
}

double ddot(int n, double *x, double *y) //
{
  int i;
  double dot=0;
  for (i = 0; i < n; i++) dot += x[i]*y[i];
  return dot;
}

/*------------------------------------------
 * countlines
 * 
 *------------------------------------------ */
int countlines(char *fname) //
{
  int lines = 0;
  char buf[BUFSIZ];
  FILE *fp = fopen(fname ,"r"); assert(fp);
  while (fgets(buf, BUFSIZ, fp)) lines++;
  fclose(fp);
  lines -= 3; // less 3 header lines
  assert(lines>0);
  return lines;
}

int countN(char *fname) //
{
  int i, count, N = 0;
  char buf[BUFSIZ];
  FILE *fp = fopen(fname ,"r"); assert(fp);
  for (i = 0; i < 3; i++) fgets(buf, BUFSIZ, fp); // skip 3 header lines
  while (fscanf(fp, "%*d%*d%d", &count) != EOF) N += count;
  fclose(fp);
  assert(N>0);
  return N;
}

/*------------------------------------------
 * sort: call qsort library function
 * isort
 * dsort
 *------------------------------------------ */

void isort(int n, int *x, int direction, int *indx) //
{
  int i;
  assert(direction*direction==1);
  icomp_vec = ivec(n);
  for (i = 0; i < n; i++) {
    icomp_vec[i] = direction*x[i];
    indx[i] = i;
  }
  qsort(indx,n,sizeof(int),icomp);
  free(icomp_vec);
}
static int icomp(const void *pl, const void *p2)
{
  //size_t i = * (size_t *) pl;
  //size_t j = * (size_t *) p2;
  //return (icomp_vec[i] - icomp_vec[j]);
  int i = * (int *) pl;
  int j = * (int *) p2;
  return (icomp_vec[i] - icomp_vec[j]);
}

int *dsort(int n, double *x) //
{
  int *indx = ivec(n);
  int i;
  dcomp_vec = dvec(n);
  for (i = 0; i < n; i++) {
    dcomp_vec[i] = -x[i];
    indx[i] = i;
  }
  qsort(indx,n,sizeof(int),dcomp);
  free(dcomp_vec);
  return indx;
}

static int dcomp(const void *pl, const void *p2)
{
  size_t i = * (size_t *) pl;
  size_t j = * (size_t *) p2;
  if (dcomp_vec[i] >  dcomp_vec[j]) return  1;
  if (dcomp_vec[i] <  dcomp_vec[j]) return -1;
  if (dcomp_vec[i] == dcomp_vec[j]) return  0;
  return 0;
}

void read_docwordcountbin(int NNZ, int *w, int *d, int *c, char *fname) //
{
  FILE *fp;
  int chk;
  fp = fopen(fname,"r"); assert(fp);
  fscanf(fp,"%d", &chk);
  fscanf(fp,"%d", &chk);
  fscanf(fp,"%d", &chk);
  chk = fread(w,sizeof(int),NNZ,fp); assert(chk==NNZ);
  chk = fread(d,sizeof(int),NNZ,fp); assert(chk==NNZ);
  chk = fread(c,sizeof(int),NNZ,fp); assert(chk==NNZ);
  fclose(fp);
}

int countNNZ(int nr, int nc, int **x) //
{
  int i, j, NNZ=0;
  for (i = 0; i < nr; i++) 
    for (j = 0; j < nc; j++) 
      if (x[i][j] > 0) NNZ++;
  return NNZ;
}

int countNNZ_d(int nr, int nc, double **x) //
{
  int i, j, NNZ=0;
  for (i = 0; i < nr; i++) 
    for (j = 0; j < nc; j++) 
      if (fabs(x[i][j]) > 1e-6) NNZ++;
  return NNZ;
}

void write_sparse(int nr, int nc, int **x, char *fname) //
{
  FILE *fp = fopen(fname,"w");
  int i, j;
  assert(fp);
  fprintf(fp, "%d\n", nr);
  fprintf(fp, "%d\n", nc);
  fprintf(fp, "%d\n", countNNZ(nr,nc,x));
  for (i = 0; i < nr; i++) 
    for (j = 0; j < nc; j++) 
      if (x[i][j] > 0) fprintf(fp, "%d %d %d\n", i+1 , j+1 , x[i][j]);
  fclose(fp);
}

void write_sparse_d(int nr, int nc, double **x, char *fname) //
{
  FILE *fp = fopen(fname,"w");
  int i, j;
  assert(fp);
  fprintf(fp, "%d\n", nr);
  fprintf(fp, "%d\n", nc);
  fprintf(fp, "%d\n", countNNZ_d(nr,nc,x));
  for (i = 0; i < nr; i++) 
    for (j = 0; j < nc; j++) 
      if (fabs(x[i][j]) > 1e-6) fprintf(fp, "%d %d %.0f\n", i+1 , j+1 , x[i][j]);
  fclose(fp);
}

void write_sparsebin(int nr, int nc, int **x, char *fname) //
{
  int i, j, k, chk;
  int NNZ   = countNNZ(nr,nc,x);
  int *col1 = ivec(NNZ);
  int *col2 = ivec(NNZ);
  int *col3 = ivec(NNZ);
  FILE *fp  = fopen(fname,"w"); assert(fp);
  for (i = 0, k = 0; i < nr; i++) 
    for (j = 0; j < nc; j++) 
      if (x[i][j] > 0) {
	col1[k] = i;
	col2[k] = j;
	col3[k] = x[i][j];
	k++;
      }
  assert(k==NNZ);
  fwrite(&nr, sizeof(int),1,fp);
  fwrite(&nc, sizeof(int),1,fp);
  fwrite(&NNZ,sizeof(int),1,fp);
  chk = fwrite(col1,sizeof(int),NNZ,fp); assert(chk==NNZ);
  chk = fwrite(col2,sizeof(int),NNZ,fp); assert(chk==NNZ);
  chk = fwrite(col3,sizeof(int),NNZ,fp); assert(chk==NNZ);
  fclose(fp);
  free(col1);
  free(col2);
  free(col3);
}

int **read_sparse(char *fname, int *nr_, int *nc_) //
{
  FILE *fp = fopen(fname,"r");
  int i, j, c, nr, nc, NNZ;
  int **x;
  assert(fp);
  fscanf(fp,"%d", &nr);  assert(nr>0);
  fscanf(fp,"%d", &nc);  assert(nc>0);
  fscanf(fp,"%d", &NNZ); assert(NNZ>0);
  x = imat(nr,nc);
  while (fscanf(fp, "%d%d%d", &i, &j, &c) != EOF) {
    i--;
    j--;
    assert(i<nr);
    assert(j<nc);
    assert(c>0);
    x[i][j] = c;
  }
  fclose(fp);
  *nr_ = nr;
  *nc_ = nc;
  return x;
}

void read_docID_wordID(char *fname, int *d, int *w) //
{
  FILE *fp = fopen(fname,"r");
  int i=0, dd, tmp, ww;
  assert(fp);
  while (fscanf(fp, "%d%d%d", &dd, &tmp, &ww) != EOF) {
    d[i] = --dd;
    w[i] = --ww;
    i++;
  }
  fclose(fp);
}

int **read_sparse_trans(char *fname, int *nr_, int *nc_) //
{
  FILE *fp = fopen(fname,"r");
  int i, j, c, nr, nc, NNZ;
  int **x;
  assert(fp);
  fscanf(fp,"%d", &nr);  assert(nr>0);
  fscanf(fp,"%d", &nc);  assert(nc>0);
  fscanf(fp,"%d", &NNZ); assert(NNZ>0);
  x = imat(nc,nr);
  while (fscanf(fp, "%d%d%d", &i, &j, &c) != EOF) {
    i--;
    j--;
    assert(i<nr);
    assert(j<nc);
    assert(c>0);
    x[j][i] = c;
  }
  fclose(fp);
  *nr_ = nc;
  *nc_ = nr;
  return x;
}

void read_dw(char *fname, int *d, int *w, int *D, int *W) //
{
  int i,wt,dt,ct,count,NNZ;
  FILE *fp = fopen(fname ,"r"); assert(fp);
  count = 0;
  fscanf(fp,"%d", D);    assert(*D>0);
  fscanf(fp,"%d", W);    assert(*W>0);
  fscanf(fp,"%d", &NNZ); assert(NNZ>0);
  while (fscanf(fp, "%d%d%d", &dt, &wt, &ct) != EOF) {
    for (i = count; i < count+ct; i++) {
      w[i] = wt-1;
      d[i] = dt-1;
    }
    count += ct;
  }
  fclose(fp);
}

void fill_Nd(int N, int *d, int *Nd) //
{
  int i;
  for (i = 0; i < N; i++) Nd[d[i]]++;
}

void read_dwc(char *fname, int *d, int *w, int *c, int *D, int *W) //
{
  FILE *fp = fopen(fname,"r");
  int i=0, dd, ww, cc, NNZ;
  assert(fp);
  fscanf(fp,"%d", D);    assert(*D>0);
  fscanf(fp,"%d", W);    assert(*W>0);
  fscanf(fp,"%d", &NNZ); assert(NNZ>0);
  while (fscanf(fp, "%d%d%d", &dd, &ww, &cc) != EOF) {
    d[i] = --dd;
    w[i] = --ww;
    c[i] = cc;
    i++;
  }
  assert(i==NNZ);
  fclose(fp);
}

int read_NNZbin(char *fname) //
{
  int NNZ;
  FILE *fp = fopen(fname,"r"); assert(fp);
  assert(fread(&NNZ,sizeof(int),1,fp)); // nr
  assert(fread(&NNZ,sizeof(int),1,fp)); // nc
  assert(fread(&NNZ,sizeof(int),1,fp)); // NNZ
  fclose(fp);
  return NNZ;
}

int read_NNZ(char *fname) //
{
  int NNZ;
  FILE *fp = fopen(fname,"r"); assert(fp);
  fscanf(fp,"%d", &NNZ); // nr
  fscanf(fp,"%d", &NNZ); // nc
  fscanf(fp,"%d", &NNZ); // NNZ
  fclose(fp);
  return NNZ;
}

void read_sparsebin(char *fname, int *col1, int *col2, int *col3) //
{
  int nr, nc, NNZ, chk;
  FILE *fp = fopen(fname,"r"); assert(fp);
  assert(fread(&nr, sizeof(int),1,fp)); assert(nr>0);
  assert(fread(&nc, sizeof(int),1,fp)); assert(nc>0);
  assert(fread(&NNZ,sizeof(int),1,fp)); assert(NNZ>0);
  chk = fread(col1,sizeof(int),NNZ,fp); assert(chk==NNZ);
  chk = fread(col2,sizeof(int),NNZ,fp); assert(chk==NNZ);
  chk = fread(col3,sizeof(int),NNZ,fp); assert(chk==NNZ);
  fclose(fp);
}

void write_ivec (int n, int *x, char *fname) //
{
  FILE *fp = fopen(fname,"w");
  int i;
  assert(fp);
  for (i = 0; i < n; i++)  fprintf(fp, "%d\n", x[i]+1 );
  fclose(fp);
}

void write_dvec (int n, double *x, char *fname) //
{
  FILE *fp = fopen(fname,"w");
  int i;
  assert(fp);
  for (i = 0; i < n; i++)  fprintf(fp, "%.6f\n", x[i] );
  fclose(fp);
}

void read_ivec (int n, int *x, char *fname) //
{
  FILE *fp = fopen(fname,"r");
  int i;
  assert(fp);
  for (i = 0; i < n; i++)  { fscanf(fp, "%d", x+i ); x[i]--; }
  fclose(fp);
}

void read_dvec (int n, double *x, char *fname) //
{
  FILE *fp = fopen(fname,"r");
  int i;
  assert(fp);
  for (i = 0; i < n; i++)  { fscanf(fp, "%lf", x+i ); }
  fclose(fp);
}

/*------------------------------------------
 * randperm
 *------------------------------------------ */

int *randperm(int n) //
{
  int *order = ivec(n);
  int k, nn, takeanumber, temp;
  for (k=0; k<n; k++) order[ k ] = k;
  nn = n;
  for (k=0; k<n; k++) {
    // take a number between 0 and nn-1
    takeanumber = (int) (nn*drand48());
    temp = order[ nn-1 ];
    order[ nn-1 ] = order[ takeanumber ];
    order[ takeanumber ] = temp;
    nn--;
  }
  return order;
}

/*------------------------------------------
 * randomassignment
 *------------------------------------------ */
void randomassignment(int N, int T, int *w, int *d, int *z, int **Nwt, int **Ndt, int *Nt) //
{
  int i, t;
  for (i = 0; i < N; i++) {
    t = (int)(T*drand48());
    z[i] = t;
    Nwt[w[i]][t]++;
    Ndt[d[i]][t]++;
    Nt[t]++;
  }
}

void randomassignment_d(int N, int T, int *w, int *d, int *z, double **Nwt, double **Ndt, double *Nt) //
{
  int i, t;
  for (i = 0; i < N; i++) {
    t = (int)(T*drand48());
    z[i] = t;
    Nwt[w[i]][t]++;
    Ndt[d[i]][t]++;
    Nt[t]++;
  }
}

void randomassignmentC(int NNZ, int T, int *w, int *d, int *c, int *z, int **Nwt, int **Ndt, int *Nt) //
{
  int i,j,jj, t;
  j = 0;
  for (i = 0; i < NNZ; i++) {
    for (jj = 0; jj < c[i]; jj++) {
      t = (int)(T*drand48());
      z[j] = t;
      Nwt[w[i]][t]++;
      Ndt[d[i]][t]++;
      Nt[t]++;
      j++;
    }
  }
}

void randomassignmentC_d(int NNZ, int T, int *w, int *d, int *c, int *z, double **Nwt, double **Ndt, double *Nt) //
{
  int i,j,jj, t;
  j = 0;
  for (i = 0; i < NNZ; i++) {
    for (jj = 0; jj < c[i]; jj++) {
      t = (int)(T*drand48());
      z[j] = t;
      Nwt[w[i]][t]++;
      Ndt[d[i]][t]++;
      Nt[t]++;
      j++;
    }
  }
}

void randomassignment_rank(int N, int T, int *w, int *d, int *drank, int *z, int **Nwt, int **Ndt, int *Nt) //
{
  int i, t;
  for (i = 0; i < N; i++) {
    t = (int)(T*drand48());
    z[i] = t;
    Nwt[w[i]][t] += drank[d[i]];
    Ndt[d[i]][t] += drank[d[i]];
    Nt[t] += drank[d[i]];
  }
}

void assignment(int N, int *w, int *d, int *z, int **Nwt, int **Ndt, int *Nt) //
{
  int i, t;
  for (i = 0; i < N; i++) {
    t = z[i];
    Nwt[w[i]][t]++;
    Ndt[d[i]][t]++;
    Nt[t]++;
  }
}

void randomassignment_2layer(int N, int T, int S, int *w, int *d, int *z, int *y, int **Nwt, int **zy, int **Ndt, int *Nt, int *ytot) //
{
  int i, t, s;
  for (i = 0; i < N; i++) {
    t = (int)(T*drand48());
    s = (int)(S*drand48());
    z[i] = t;
    y[i] = s;
    Nwt[w[i]][t]++;
    zy[t][s]++;
    Ndt[d[i]][s]++;
    Nt[t]++;
    ytot[s]++;
  }
}

void randomassignment2(int N, int T, int *d, int *z, int **Ndt) //
{
  int i, t;
  for (i = 0; i < N; i++) {
    t = (int)(T*drand48());
    z[i] = t;
    Ndt[d[i]][t]++;
  }
}

/*------------------------------------------
 * sample_chain
 *------------------------------------------ */
void sample_chain (int N, int W, int T, double alpha, double beta, int *w, int *d, int *z, int **Nwt, int **Ndt, int *Nt, int *order) //
{
  int ii, i, t;
  double totprob, U, cumprob;
  double *prob = dvec(T);
  double wbeta = W*beta;
  int wid, did;
  int *word_vec;
  int *doc_vec;
  
  for (ii = 0; ii < N; ii++) {

    i = order[ ii ];

    wid = w[i];
    did = d[i];

    word_vec = Nwt[wid];
    doc_vec  = Ndt[did];
      
    t = z[i];
    Nt[t]--;
    word_vec[t]--;
    doc_vec[t]--;
    totprob = 0;
    
    for (t = 0; t < T; t++) {
      prob[t] = (doc_vec[t] + alpha) * (word_vec[t] + beta) / (Nt[t] + wbeta);
      totprob += prob[t];
    }
    
    U = drand48()*totprob;
    cumprob = prob[0];
    t = 0;
    while (U>cumprob) {
      t++;
      cumprob += prob[t];
    }
    
    z[i] = t;
    word_vec[t]++;
    doc_vec[t]++;
    Nt[t]++;
  }
  
  free(prob);
}

void sample_chain_d (int N, int W, int T, int *w, int *d, int *z, double **Nwt, double **Ndt, double *Nt, int *order) //
{
  int ii, i, t;
  double totprob, U, cumprob;
  double *prob = dvec(T);
  int wid, did;
  double *word_vec;
  double *doc_vec;
  
  for (ii = 0; ii < N; ii++) {

    i = order[ ii ];

    wid = w[i];
    did = d[i];

    word_vec = Nwt[wid];
    doc_vec  = Ndt[did];
      
    t = z[i];
    Nt[t]--;
    word_vec[t]--;
    doc_vec[t]--;
    totprob = 0;
    
    for (t = 0; t < T; t++) {
      prob[t] = doc_vec[t] * word_vec[t] / Nt[t];
      totprob += prob[t];
    }
    
    U = drand48()*totprob;
    cumprob = prob[0];
    t = 0;
    while (U>cumprob) {
      t++;
      cumprob += prob[t];
    }
    
    z[i] = t;
    word_vec[t]++;
    doc_vec[t]++;
    Nt[t]++;
  }
  
  free(prob);
}

void sample_chain_d_icm (int N, int W, int T, int *w, int *d, int *z, double **Nwt, double **Ndt, double *Nt, int *order) //
{
  int ii, i, t;
  double prob;
  int wid, did;
  double *word_vec;
  double *doc_vec;
  double maxprob=0;
  int   imaxprob=-1;
  
  for (ii = 0; ii < N; ii++) {

    i = order[ ii ];

    wid = w[i];
    did = d[i];

    word_vec = Nwt[wid];
    doc_vec  = Ndt[did];
      
    t = z[i];
    Nt[t]--;
    word_vec[t]--;
    doc_vec[t]--;
    
    maxprob=0;
    imaxprob=-1;
    for (t = 0; t < T; t++) {
      prob = doc_vec[t] * word_vec[t] / Nt[t];
      if (prob > maxprob) {
	 maxprob = prob;
	imaxprob = t;
      }
    }
    
    t = imaxprob;
    
    z[i] = t;
    word_vec[t]++;
    doc_vec[t]++;
    Nt[t]++;
  }
  
}

/*------------------------------------------
 * sample_chain_alpha
 *------------------------------------------ */
void sample_chain_alpha (int N, int W, int T, double *alpha, double beta, int *w, int *d, int *z, int **Nwt, int **Ndt, int *Nt, int *order) //
{
  int ii, i, t;
  double totprob, U, cumprob;
  double *prob = dvec(T);
  double wbeta = W*beta;
  int wid, did;
  int *word_vec;
  int *doc_vec;

  for (ii = 0; ii < N; ii++) {

    i = order[ ii ];

    wid = w[i];
    did = d[i];

    word_vec = Nwt[wid];
    doc_vec  = Ndt[did];
      
    t = z[i];      // take the current topic assignment to word token i
    Nt[t]--;     // and substract that from the counts
    word_vec[t]--;
    doc_vec[t]--;
    totprob = 0;
    
    for (t = 0; t < T; t++) {
      prob[t] = (doc_vec[t] + alpha[t]) * (word_vec[t] + beta) / (Nt[t] + wbeta);
      totprob += prob[t];
    }
    
    U = drand48()*totprob;
    cumprob = prob[0];
    t = 0;
    
    // sample a topic t from the distribution
    while (U>cumprob) {
      t++;
      cumprob += prob[t];
    }
    
    z[i] = t;      // assign current word token i to topic t
    word_vec[t]++; // and update counts
    doc_vec[t]++;
    Nt[t]++;
  }

  free(prob);
}

/*------------------------------------------
 * sample_chain_rank
 *------------------------------------------ */
void sample_chain_rank (int N, int W, int T, double alpha, double beta, int *w, int *d, int *drank, int *z, int **Nwt, int **Ndt, int *Nt, int *order) //
{
  int ii, i, t;
  double totprob, U, cumprob;
  double *prob = dvec(T);
  double wbeta = W*beta;
  
  for (ii = 0; ii < N; ii++) {

    i = order[ ii ];
      
    t = z[i];      // take the current topic assignment to word token i
    Nt[t] -= drank[d[i]];
    Nwt[w[i]][t] -= drank[d[i]];
    Ndt[d[i]][t] -= drank[d[i]];
    totprob = 0;
      
    for (t = 0; t < T; t++) {
      prob[t] = (Nwt[w[i]][t] + beta)/(Nt[t]+  wbeta)*(Ndt[d[i]][t]+  alpha);
      totprob += prob[t];
    }
    
    U = drand48()*totprob;
    cumprob = prob[0];
    t = 0;
    
    // sample a topic t from the distribution
    while (U>cumprob) {
      t++;
      cumprob += prob[t];
    }
    
    z[i] = t;      // assign current word token i to topic t
    Nwt[w[i]][t] += drank[d[i]];
    Ndt[d[i]][t] += drank[d[i]];
    Nt[t] += drank[d[i]];
  }

  free(prob);
}

void sample_chain0 (int N, int W, int T, double alpha, double beta, int *w, int *d, int *z, int **Nwt, int **Ndt, int *Nt) //
{
  int i, t;
  double totprob, U, cumprob;
  double *prob = dvec(T);
  double wbeta = W*beta;
  
  for (i = 0; i < N; i++) {
    
    t = z[i];      // take the current topic assignment to word token i
    Nt[t]--;     // and substract that from the counts
    Nwt[w[i]][t]--;
    Ndt[d[i]][t]--;
    
    for (t = 0, totprob = 0.0; t < T; t++) {
      prob[t] = (Ndt[d[i]][t] + alpha) * (Nwt[w[i]][t] + beta) / (Nt[t] + wbeta);
      totprob += prob[t];
    }
    
    
    U = drand48()*totprob;
    cumprob = prob[0];
    t = 0;
    
    // sample a topic t from the distribution
    while (U>cumprob) {
      t++;
      cumprob += prob[t];
    }
    
    z[i] = t;      // assign current word token i to topic t
    Nwt[w[i]][t]++; // and update counts
    Ndt[d[i]][t]++;
    Nt[t]++;
  }

  free(prob);  
}

void sample_chain_2layer (int N, int W, int T, int S, double alpha, double beta, double gamma, int *w, int *d, int *z, int *y, int **Nwt, int **zy, int **Ndt, int *Nt, int *ytot) //
{
  int    i, t, s;
  double totprob, U, cumprob, term1, term2, term3;
  double wbeta  = W*beta;
  double tgamma = T*gamma;
  double **prob = dmat(T,S);
  
  for (i = 0; i < N; i++) {

    t = z[i];
    s = y[i];
    Nt[t]--;
    ytot[s]--;
    Nwt[w[i]][t]--;
    zy[t][s]--;
    Ndt[d[i]][s]--;

    totprob = 0;      
    for (t = 0; t < T; t++) {
      for (s = 0; s < S; s++) {
	term1 = (Nwt[w[i]][t] + beta) / (Nt[t] + wbeta);
	term2 = (zy[t][s]  + gamma)  / (ytot[s] + tgamma);
	term3 = (Ndt[d[i]][s] + alpha);
	prob[t][s] = term1*term2*term3;
	totprob += prob[t][s];
      }
    }
    
    U = drand48()*totprob;
    cumprob = prob[0][0];
    t = 0;
    s = 0;
    while (U>cumprob) {
      t++;
      if (t >= T) { s++; t=0; }
      cumprob += prob[t][s];
    }
    
    z[i] = t;
    y[i] = s;
    Nt[t]++;
    ytot[s]++;
    Nwt[w[i]][t]++;
    zy[t][s]++;
    Ndt[d[s]][t]++;
  }
  
  free_dmat(prob);

}

void resample_chain (int N, int W, int T, double alpha, double beta, int *w, int *d, int *z, int **Nwt, int **Ndt, int *Nt) //
{
  int i, t;
  double totprob, U, cumprob;
  double *prob = dvec(T);
  double wbeta = W*beta;

  for (i = 0; i < N; i++) {

    t = z[i];
    Ndt[d[i]][t]--;
    totprob = 0;
      
    for (t = 0; t < T; t++) {
      prob[t] = (Nwt[w[i]][t] + beta)/(Nt[t] + wbeta)*(Ndt[d[i]][t] + alpha);
      totprob += prob[t];
    }
    
    U = drand48()*totprob;
    cumprob = prob[0];
    t = 0;
    
    // sample a topic t from the distribution
    while (U>cumprob) {
      t++;
      cumprob += prob[t];
    }
    
    z[i] = t;
    Ndt[d[i]][t]++;
  }
  
  free(prob);
}

void oversample_Ndt (int N, int W, int T, double alpha, double beta, int *w, int *d, int *z, int **Nwt, int **Ndt, int *Nt) //
{
  int i, t, k, ntimes=4;
  double totprob, U, cumprob;
  double *prob = dvec(T);
  double wbeta = W*beta;

  for (i = 0; i < N; i++) {

    totprob = 0;
    for (t = 0; t < T; t++) {
      prob[t] = (Nwt[w[i]][t] + beta)/(Nt[t] + wbeta)*(Ndt[d[i]][t] + alpha);
      totprob += prob[t];
    }
    
    for (k = 0; k < ntimes; k++) {
      U = drand48()*totprob;
      cumprob = prob[0];
      t = 0;
      while (U>cumprob) {
	t++;
	cumprob += prob[t];
      }
      Ndt[d[i]][t]++;
    }
  }
  
  free(prob);
}

void loglike (int N, int W, int D, int T, double alpha, double beta, int *w, int *d, int **Nwt, int **Ndt, int *Nt, int *Nd) //
{
  int    i, j, t;
  double llike;
  static int init = 0;
  static double **prob_w_given_t;
  static double **prob_t_given_d;
  static double *Nd_;
  double Nt_;

  if (init==0) {
    init = 1;
    prob_w_given_t = dmat(W,T);
    prob_t_given_d = dmat(D,T);
    Nd_ = dvec(D);
    for (j = 0; j < D; j++) Nd_[j] = Nd[j] + T*alpha;
  }
  
  for (t = 0; t < T; t++) {
    Nt_ = Nt[t] + W*beta;
    for (i = 0; i < W; i++) prob_w_given_t[i][t] = (Nwt[i][t]+beta) / Nt_;
    for (j = 0; j < D; j++) prob_t_given_d[j][t] = (Ndt[j][t]+alpha)/ Nd_[j];
  }
   
  llike = 0;
  for (i = 0; i < N; i++)
    llike += log(ddot(T, prob_w_given_t[w[i]], prob_t_given_d[d[i]]));
  
  printf(">>> llike = %.6e    ", llike);
  printf("pplex = %.4f\n", exp(-llike/N));
}

void chk_sum (int n, int T, int **x, int *sumx) //
{
  int i, t, sum;
  for (t = 0; t < T; t++) {
    sum = 0;
    for (i = 0; i < n; i++) sum += x[i][t];
    assert(sum==sumx[t]);
  }
}

void chk_sum_d (int n, int T, double **x, double *sumx) //
{
  int i, t;
  double sum;
  for (t = 0; t < T; t++) {
    sum = 0.0;
    for (i = 0; i < n; i++) sum += x[i][t];
    if (fabs(sum-sumx[t])>1e-6*n/T) {
      printf("error in chk_sum_d: N=%d  T=%d  t=%d  sum=%f  chk=%f\n", n, T, t, sum, sumx[t]);
    }
    //assert(fabs(sum-sumx[t])<1e-6*n/T);
  }
}

void getNt (int n, int T, int **x, int *Nt) //
{
  int i, t, sum;
  for (t = 0; t < T; t++) {
    sum = 0;
    for (i = 0; i < n; i++) sum += x[i][t];
    Nt[t] = sum;
  }
}

void add_smooth_d (int n, int T, double **x, double smooth) //
{
  int i, t;
  for (i = 0; i < n; i++)
    for (t = 0; t < T; t++) 
      x[i][t]+=smooth;
}

void add_smooth1d (int T, double *x, double smooth) //
{
  int t;
  for (t = 0; t < T; t++) 
    x[t]+=smooth;
}

void set_smooth_d (int n, int T, double **x, double smooth) //
{
  int i, t;
  for (i = 0; i < n; i++)
    for (t = 0; t < T; t++) 
      x[i][t]=smooth;
}

void set_smooth1d (int T, double *x, double smooth) //
{
  int t;
  for (t = 0; t < T; t++) 
    x[t]=smooth;
}

double pplex(int N, int W, int T, double alpha, double beta, int *w, int *d, int **Nwt, int **Ndt) //
{
  int i, t;
  double mypplex, llike=0, p1, p2, Z, pwd;
  double *zwt = dvec(T);
  
  for (t=0;t<T;t++) for (zwt[t]=0, i=0;i<W;i++) zwt[t]+=Nwt[i][t]+beta;
  
  for (i=0;i<N;i++) {
    Z=pwd=0;
    for (t=0;t<T;t++) {
      p1=Nwt[w[i]][t]+beta; p2=Ndt[d[i]][t]+alpha; Z+=p2;
      pwd += p1*p2/zwt[t];
    }
    llike += log( pwd/Z );
  }
  
  mypplex = exp(-llike/N);
  
  return mypplex;
}

double pplex_d(int N, int W, int T, int *w, int *d, double **Nwt, double **Ndt) //
{
  int i, t;
  double mypplex, llike=0, p1, p2, Z, pwd;
  double *zwt = dvec(T);
  
  for (t=0;t<T;t++) for (zwt[t]=0, i=0;i<W;i++) zwt[t]+=Nwt[i][t];
  
  for (i=0;i<N;i++) {
    Z=pwd=0;
    for (t=0;t<T;t++) {
      p1=Nwt[w[i]][t]; p2=Ndt[d[i]][t]; Z+=p2;
      pwd += p1*p2/zwt[t];
    }
    llike += log( pwd/Z );
  }
  
  mypplex = exp(-llike/N);
  
  return mypplex;
}
