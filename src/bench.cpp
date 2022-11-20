// Compute a forward/backward double precision complex FFT using fftMPI
// change FFT size by editing 3 "FFT size" lines
// run on any number of procs 

// Run syntax:
// % simple               # run in serial
// % mpirun -np 4 simple  # run in parallel 

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <queue>
#include <string>
//#include "comm_part.h"

#include "fft3d.h" 
#include "ffttype.h"

using namespace FFTMPI_NS; 

// FFT size 

#define NFAST 512
#define NMID 512
#define NSLOW 512

// precision-dependent settings 

#ifdef FFT_SINGLE
int precision = 1;
#else
int precision = 2;
#endif 

#define ITERATIONS 10
// main program 

int worst_64procs(MPI_Comm comm, MPI_Comm *newcomm){
    int key,rank,ret;
    MPI_Comm_rank(comm, &rank);
    if ( rank < 32 ){
        key = (rank % 2 == 0) ? rank : rank + 31;
    }
    else{
        key = ( rank % 2 == 0 ) ? rank - 31 : rank;
    }

    /* for (i=0; i<2; i++){ */
    /*     if ( rank % 2 == i ){ */
    /*         key = rank/2+i*32; */
    /*     } */
    /* } */
    ret = MPI_Comm_split(comm, 0, key, newcomm);
    return ret;
}

int worst_1024procs(MPI_Comm comm, MPI_Comm *newcomm){
    int key,rank,ret,i;
    MPI_Comm_rank(comm, &rank);
    for (i=0; i<32; i++){
        if ( rank % 32 == i ){
            key = rank/32+i*32;
        }
    }
    ret = MPI_Comm_split(comm, 0, key, newcomm);
    return ret;
}


int intrasock(MPI_Comm comm, MPI_Comm *newcomm){
    int key,rank,ret,size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if ( rank % 2 == 0 ){
        key  = rank / 2;
    }
    else{
        key = rank/2 + size/2;
    }
    ret = MPI_Comm_split(comm, 0, key , newcomm);
    return ret;
}


int intrasock_size(MPI_Comm comm, MPI_Comm *newcomm){
    int key,rank,ret,size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if ( rank % 2 == 0 ){
        key  = rank / 2;
    }
    else{
        key = rank/2 + size/2;
    }
    ret = MPI_Comm_split(comm, 0, key , newcomm);
    return ret;
}

int weird_64procs(MPI_Comm comm, MPI_Comm *newcomm){
    int key,rank,ret;
    MPI_Comm_rank(comm, &rank);
    if ( rank == 0 ){
        key = 0;
    }
    else if ( rank < 16 ){
        key = rank + 16;
    }
    else if ( rank < 32 ){
       key = rank + 17;
    }
    else if ( rank >= 32 && rank < 48 ){
        key = rank - 31;
    }
    else if ( rank == 48 ){
        key = rank - 16;
    }
    else{
        key = rank;
    }
    ret = MPI_Comm_split(comm, 0, key , newcomm);
    return ret;

}

int inter_64procs(MPI_Comm comm, MPI_Comm *newcomm){
    int key,rank;
    MPI_Comm_rank(comm, &rank);
    if ( rank >= 32 && rank < 48  ){
        key = rank / 32;
        key += rank % 32;
    }
    else {
        key = 0;
    }
    return key;

}


/* groups of nodes with 64 processes */
int intrasock_nodes_64procs(MPI_Comm comm, MPI_Comm *newcomm, int comm_size, int nnodes){
    int key,rank,ret,rr,offset;
    MPI_Comm_rank(comm, &rank);
    rr = rank / (comm_size*nnodes);
    offset = rr*comm_size*nnodes;
    rr = rank % (comm_size*nnodes);
    if ( rank % 2 == 0 ){
        key = rr / 2 + offset;
    }
    else{
        key = (rr+1)/2 + (comm_size-1) + offset;
    }
    ret = MPI_Comm_split(comm, 0, key , newcomm);
    return ret;
}

int rename_vsc5(MPI_Comm comm, MPI_Comm *newcomm,int numa_size,int nnumas){
  int key,rank,ret;
  MPI_Comm_rank(comm, &rank);
  key = rank%numa_size*nnumas+rank/numa_size;
  ret = MPI_Comm_split(comm, 0, key, newcomm);
  return ret;
}

std::queue<std::string> args_to_vec(int argc, char *argv[]){
  std::queue<std::string> res;
  for (int i =2; i!=argc; ++i) {
    res.push(argv[i]);
  }
  return res;
 }

int main(int narg, char **args)
{
  // setup MPI 
  int partscheme = 0;
  MPI_Init(&narg,&args);
  MPI_Comm world;
  if ( narg > 1 ){
    partscheme = atoi(args[1]);
  }
  switch (partscheme) {
    case 0:
      world = MPI_COMM_WORLD;
      break;
    case 1:
      rename_vsc5(MPI_COMM_WORLD, &world, 16, 8);
      break;
    case 2:
      weird_64procs(MPI_COMM_WORLD, &world);
      break;
    case 3:
      worst_1024procs(MPI_COMM_WORLD, &world);
      break;
    case 4:
      intrasock(MPI_COMM_WORLD, &world);
      break;
    case 5:
      intrasock_nodes_64procs(MPI_COMM_WORLD, &world, 64, 2);
      break;
    default:
      world = MPI_COMM_WORLD;

  }


  int me,nprocs;
  MPI_Comm_size(world,&nprocs);
  MPI_Comm_rank(world,&me); 
  if ( me == 0 ){
    printf("Precison = %d\nPartitioning scheme = %d\n",FFT_PRECISION,partscheme);
    fflush(stdout);
  }
  // instantiate FFT 

  FFT3d *fft = new FFT3d(world,precision); 

  // simple algorithm to factor Nprocs into roughly cube roots 

  int npfast,npmid,npslow; 

  npfast = (int) pow(nprocs,1.0/3.0);
  while (npfast < nprocs) {
    if (nprocs % npfast == 0) break;
    npfast++;
  }
  int npmidslow = nprocs / npfast;
  npmid = (int) sqrt(npmidslow);
  while (npmid < npmidslow) {
    if (npmidslow % npmid == 0) break;
    npmid++;
  }
  npslow = nprocs / npfast / npmid; 

  if ( me == 0 ){
    printf("npfast: %d, npmid: %d, npslow: %d\n",npfast,npmid,npslow);
  }
  // partition grid into Npfast x Npmid x Npslow bricks

  int nfast,nmid,nslow;
  int ilo,ihi,jlo,jhi,klo,khi;
  int iterations = 10;
  nfast = nmid = nslow = 32;
  std::queue<std::string> arguments = args_to_vec(narg,args);
  if (!arguments.empty()){
    nfast = std::stoi(arguments.front());
    arguments.pop();
  }
  if (!arguments.empty()){
    nmid = std::stoi(arguments.front());
    arguments.pop();
  }
  if (!arguments.empty()){
    nslow = std::stoi(arguments.front());
    arguments.pop();
  }
  if (!arguments.empty()){
    iterations = std::stoi(arguments.front());
    arguments.pop();
  }
  // if ( me == 0 )
  //   std::cout << nfast << "x" << nmid << "x" << nslow << "\n";
  int ipfast = me % npfast;
  int ipmid = (me/npfast) % npmid;
  int ipslow = me / (npfast*npmid); 

  ilo = (int) 1.0*ipfast*nfast/npfast;
  ihi = (int) 1.0*(ipfast+1)*nfast/npfast - 1;
  jlo = (int) 1.0*ipmid*nmid/npmid;
  jhi = (int) 1.0*(ipmid+1)*nmid/npmid - 1;
  klo = (int) 1.0*ipslow*nslow/npslow;
  khi = (int) 1.0*(ipslow+1)*nslow/npslow - 1; 

  // setup FFT, could replace with tune() 

  int fftsize,sendsize,recvsize;
  // fft->setup(nfast,nmid,nslow,
  //            ilo,ihi,jlo,jhi,klo,khi,ilo,ihi,jlo,jhi,klo,khi,
  //            0,fftsize,sendsize,recvsize);

  // tune FFT, could replace with setup() 

  fft->tune(nfast,nmid,nslow,
           ilo,ihi,jlo,jhi,klo,khi,ilo,ihi,jlo,jhi,klo,khi,
           0,fftsize,sendsize,recvsize,0,5,10.0,0);

  // initialize each proc's local grid
  // global initialization is specific to proc count 

  FFT_SCALAR *work = (FFT_SCALAR *) malloc(2*fftsize*sizeof(FFT_SCALAR)); 

  int n = 0;
  for (int k = klo; k <= khi; k++) {
    for (int j = jlo; j <= jhi; j++) {
      for (int i = ilo; i <= ihi; i++) {
        work[n] = (double) n;
        n++;
        work[n] = (double) n;
        n++;
      }
    }
  } 


  double elapsed;//total_time[ITERATIONS];
  int inx = 0;
  std::vector<double> total_time;
  // perform 2 FFTs
  double fft_time = 0.0;
  fft_time = MPI_Wtime();

  for (int i =0; i<iterations; i++){
    int n = 0;
    for (int k = klo; k <= khi; k++) {
      for (int j = jlo; j <= jhi; j++) {
        for (int i = ilo; i <= ihi; i++) {
          work[n] = (double) n;
          n++;
          work[n] = (double) n;
          n++;
        }
      }
    }
    elapsed = MPI_Wtime();
    fft->compute(work,work,1);        // forward FFT
    fft->compute(work,work,-1);       // backward FFT
    elapsed = MPI_Wtime()-elapsed;
    total_time.push_back(elapsed);
    inx ++;
  }
  fft_time = MPI_Wtime() - fft_time;

  if (me == 0) {
    printf("Two %dx%dx%d FFTs per iteration on %d procs as %dx%dx%d grid, %d iterations\n",
           nfast,nmid,nslow,nprocs,npfast,npmid,npslow,iterations);
  }
  double *max_times, *tmp, ttime;
  tmp = (double*) malloc ( sizeof(double)*iterations);
  max_times= (double*) malloc ( sizeof(double)*iterations);
  std::copy(total_time.begin(),total_time.end(),tmp);

  MPI_Reduce(tmp, max_times, total_time.size(), MPI_DOUBLE, MPI_MAX, 0, world);
  if ( me == 0 ){
    ttime = 0;
    for (inx = 0; inx<iterations; inx++){
      printf("CPU time, iter no. %d = %g secs\n",inx,max_times[inx]);
      ttime += max_times[inx];
    }
    printf("Total FFT time = %lf\nAverage FFT time = %lf\n",ttime,ttime/(double)iterations);
  }
  // find largest difference between initial/final values
  // should be near zero 

  n = 0;
  double mydiff = 0.0;
  for (int k = klo; k <= khi; k++) {
    for (int j = jlo; j <= jhi; j++) {
      for (int i = ilo; i <= ihi; i++) {
        if (fabs(work[n]-n) > mydiff) mydiff = fabs(work[n]-n);
        n++;
        if (fabs(work[n]-n) > mydiff) mydiff = fabs(work[n]-n);
        n++;
      }
    }
  } 

  double alldiff;
  MPI_Allreduce(&mydiff,&alldiff,1,MPI_DOUBLE,MPI_MAX,world);  
  if (me == 0) printf("Max difference in initial/final values = %g\n",alldiff); 

  // clean up 

  free(work);
  free(tmp);
  free(max_times);
  delete fft;
  MPI_Finalize();
} 


