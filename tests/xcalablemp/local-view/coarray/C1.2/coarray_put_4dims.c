#include <stdio.h>
#include <stdbool.h>
#include <xmp.h>
#pragma xmp nodes p(*)
#define SIZE 7
#define DIMS 4
int a[SIZE][SIZE][SIZE][SIZE]:[*], b[SIZE][SIZE][SIZE][SIZE]:[*], ret = 0;
int start[DIMS], len[DIMS], stride[DIMS];
int status;



int main(){
  for(int i=0;i<SIZE;i++)
    for(int j=0;j<SIZE;j++)
      for(int k=0;k<SIZE;k++)
	for(int m=0;m<SIZE;m++)
	  b[i][j][k][m] = xmp_node_num();

  for(start[0]=0;start[0]<2;start[0]++){
    for(len[0]=1;len[0]<=SIZE;len[0]++){
      for(stride[0]=1;stride[0]<4;stride[0]++){
	for(start[1]=0;start[1]<2;start[1]++){
	  for(len[1]=1;len[1]<=SIZE;len[1]++){
	    for(stride[1]=1;stride[1]<4;stride[1]++){
	      for(start[2]=0;start[2]<2;start[2]++){
		for(len[2]=1;len[2]<=SIZE;len[2]++){
		  for(stride[2]=1;stride[2]<4;stride[2]++){
		    for(start[3]=0;start[3]<2;start[3]++){
		      for(len[3]=1;len[3]<=SIZE;len[3]++){
			for(stride[3]=1;stride[3]<4;stride[3]++){

			  if(start[0]+(len[0]-1)*stride[0] < SIZE && start[1]+(len[1]-1)*stride[1] < SIZE
			     && start[2]+(len[2]-1)*stride[2] < SIZE && start[3]+(len[3]-1)*stride[3] < SIZE){
			    
			    for(int i=0;i<SIZE;i++)
			      for(int j=0;j<SIZE;j++)
				for(int k=0;k<SIZE;k++)
				  for(int m=0;m<SIZE;m++)
				    a[i][j][k][m] = -1;
			    
			    xmp_sync_all(&status);
			    
			    if(xmp_node_num() == 1){
			      a[start[0]:len[0]:stride[0]][start[1]:len[1]:stride[1]][start[2]:len[2]:stride[2]][start[3]:len[3]:stride[3]]:[2] 
				= b[start[0]:len[0]:stride[0]][start[1]:len[1]:stride[1]][start[2]:len[2]:stride[2]][start[3]:len[3]:stride[3]];
			    }
			    
			    xmp_sync_all(&status);
			  
			    if(xmp_node_num() == 2){
			      for(int i=0;i<len[0];i++){
				int position0 = start[0]+i*stride[0];
				for(int j=0;j<len[1];j++){
				  int position1 = start[1]+j*stride[1];
				  for(int k=0;k<len[2];k++){
				    int position2 = start[2]+k*stride[2];
				    for(int m=0;m<len[3];m++){
				      int position3 = start[3]+m*stride[3];
				      if(a[position0][position1][position2][position3] != 1){
					fprintf(stderr, "a[%d:%d:%d][%d:%d:%d][%d:%d:%d][%d:%d:%d] ", 
						start[0], len[0], stride[0], start[1], len[1], stride[1], start[2], len[2], stride[2],
						start[3], len[3], stride[3]);
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

