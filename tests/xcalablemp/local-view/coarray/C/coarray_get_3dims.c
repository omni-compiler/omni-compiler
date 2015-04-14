#include <stdio.h>
#include <stdbool.h>
#include <xmp.h>
#pragma xmp nodes p(*)
#define SIZE 10
#define DIMS 3
int a[SIZE][SIZE][SIZE]:[*], b[SIZE][SIZE][SIZE]:[*], ret = 0;
int start[DIMS], len[DIMS], stride[DIMS];
int status;



int main(){
  for(int i=0;i<SIZE;i++)
    for(int j=0;j<SIZE;j++)
      for(int k=0;k<SIZE;k++)
	b[i][j][k] = xmp_node_num();

  for(start[0]=0;start[0]<2;start[0]++){
    for(len[0]=1;len[0]<=SIZE;len[0]++){
      for(stride[0]=1;stride[0]<4;stride[0]++){
	for(start[1]=0;start[1]<2;start[1]++){
	  for(len[1]=1;len[1]<=SIZE;len[1]++){
	    for(stride[1]=1;stride[1]<4;stride[1]++){
	      for(start[2]=0;start[2]<2;start[2]++){
		for(len[2]=1;len[2]<=SIZE;len[2]++){
		  for(stride[2]=1;stride[2]<4;stride[2]++){

		    if(start[0]+(len[0]-1)*stride[0] < SIZE && start[1]+(len[1]-1)*stride[1] < SIZE
		       && start[2]+(len[2]-1)*stride[2] < SIZE){

		      for(int i=0;i<SIZE;i++)
			for(int j=0;j<SIZE;j++)
			  for(int k=0;k<SIZE;k++)
			    a[i][j][k] = -1;

		      xmp_sync_all(&status);

		      if(xmp_node_num() == 1){
			a[start[0]:len[0]:stride[0]][start[1]:len[1]:stride[1]][start[2]:len[2]:stride[2]] 
			  = b[start[0]:len[0]:stride[0]][start[1]:len[1]:stride[1]][start[2]:len[2]:stride[2]]:[2];

			for(int i=0;i<len[0];i++){
			  int position0 = start[0]+i*stride[0];
			  for(int j=0;j<len[1];j++){
			    int position1 = start[1]+j*stride[1];
			    for(int k=0;k<len[2];k++){
			      int position2 = start[2]+k*stride[2];
			      if(a[position0][position1][position2] != 2){
				fprintf(stderr, "a[%d:%d:%d][%d:%d:%d][%d:%d:%d] ", 
					start[0], len[0], stride[0], start[1], len[1], stride[1], start[2], len[2], stride[2]);
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
	}
      }
    }
  }
 end:
  xmp_sync_all(&status);
#pragma xmp bcast(ret)
  if(xmp_node_num() == 1)
    if(ret == 0) printf("PASS\n");
    else fprintf(stderr, "ERROR\n");

  return ret;
}

