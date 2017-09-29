#include <stdio.h>
#include <xmp.h>
#pragma xmp nodes p(*)
#define SIZE 10
#define DIMS 1
int a[SIZE]:[*], b[SIZE]:[*], ret = 0;
#pragma acc declare create(a,b)

int main(){
  int start[DIMS], len[DIMS], stride[DIMS];
  int image = xmpc_this_image();

#pragma acc data present(a, b)
  {
#pragma acc parallel loop
    for(int i=0;i<SIZE;i++)
      b[i] = image;

    for(start[0]=0;start[0]<2;start[0]++){
      for(len[0]=1;len[0]<=SIZE;len[0]++){
	for(stride[0]=1;stride[0]<2;stride[0]++){
	  if(start[0]+(len[0]-1)*stride[0] < SIZE){

	    for(int i=0;i<SIZE;i++) a[i] = -1;
	    xmp_sync_all(NULL);

	    if(xmpc_this_image() == 0){
#pragma acc host_data use_device(a, b)
	      a[start[0]:len[0]:stride[0]] = b[start[0]:len[0]:stride[0]]:[1];

	      int num_err = 0;
#pragma acc parallel loop reduction(+:num_err)
	      for(int i=0;i<len[0];i++){
		int position = start[0]+i*stride[0];
		if(a[position] != 1){
		  num_err++;
		}
	      }

	      if(num_err > 0){
		fprintf(stderr, "a[%d:%d:%d] ", start[0], len[0], stride[0]);
		ret = -1;
		goto end;
	      }
	    }
	  }
	}
      }
    }

  end:
    xmp_sync_all(NULL);
#pragma xmp bcast(ret)
    if(xmpc_this_image() == 0)
      if(ret == 0) printf("PASS\n");
      else fprintf(stderr, "ERROR\n");

  }
  return ret;
}
