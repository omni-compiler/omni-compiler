#include <stdio.h>
#include <xmp.h>
#pragma xmp nodes p(*)
#define SIZE 10
#define DIMS 2
int a[SIZE][SIZE]:[*], b[SIZE][SIZE]:[*], ret = 0;

int main(){
  int start[DIMS], len[DIMS], stride[DIMS];
  int status;

  for(int i=0;i<SIZE;i++)
    for(int j=0;j<SIZE;j++)
      b[i][j] = xmp_node_num();

  for(start[0]=0;start[0]<2;start[0]++){
    for(len[0]=1;len[0]<=SIZE;len[0]++){
      for(stride[0]=1;stride[0]<4;stride[0]++){
	for(start[1]=0;start[1]<2;start[1]++){
	  for(len[1]=1;len[1]<=SIZE;len[1]++){
	    for(stride[1]=1;stride[1]<4;stride[1]++){

	      if(start[0]+(len[0]-1)*stride[0] < SIZE && start[1]+(len[1]-1)*stride[1] < SIZE){
		for(int i=0;i<SIZE;i++)
		  for(int j=0;j<SIZE;j++)
		    a[i][j] = -1;

		xmp_sync_all(&status);

		if(xmp_node_num() == 1)
		  a[start[0]:len[0]:stride[0]][start[1]:len[1]:stride[1]]:[2] 
		    = b[start[0]:len[0]:stride[0]][start[1]:len[1]:stride[1]];

		xmp_sync_all(&status);

		if(xmp_node_num() == 2){
		  for(int i=0;i<len[0];i++){
		    for(int j=0;j<len[1];j++){
		      int position0 = start[0]+i*stride[0];
		      int position1 = start[1]+j*stride[1];
		      if(a[position0][position1] != 1){
			fprintf(stderr, "a[%d:%d:%d][%d:%d:%d] ", 
				start[0], len[0], stride[0], start[1], len[1], stride[1]);
			ret = -1;
			goto end;
		      }
		    }
		  }
		}
	      }	    
	    }
	  }
	}
      }
    }
  }
 end:
  xmp_sync_all(&status);
#pragma xmp bcast(ret) from p(2)
  if(xmp_node_num() == 1)
    if(ret == 0) printf("PASS\n");
    else fprintf(stderr, "ERROR\n");

  return ret;
}

