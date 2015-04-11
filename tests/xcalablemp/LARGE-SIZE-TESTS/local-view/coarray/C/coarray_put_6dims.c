#include <stdio.h>
#include <stdbool.h>
#include <xmp.h>
#pragma xmp nodes p2(*)
#define SIZE 4
#define DIMS 5
int a[SIZE][SIZE][SIZE][SIZE][SIZE][SIZE]:[*], b[SIZE][SIZE][SIZE][SIZE][SIZE][SIZE]:[*];
int start[DIMS], len[DIMS], stride[DIMS];
int status, ret = 0;

int main(){
  for(int i=0;i<SIZE;i++)
    for(int j=0;j<SIZE;j++)
      for(int k=0;k<SIZE;k++)
	for(int m=0;m<SIZE;m++)
	  for(int n=0;n<SIZE;n++)
	    for(int p=0;p<SIZE;p++)
	      b[i][j][k][m][n][p] = xmp_node_num();

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
			  for(start[4]=0;start[4]<2;start[4]++){
			    for(len[4]=1;len[4]<=SIZE;len[4]++){
			      for(stride[4]=1;stride[4]<4;stride[4]++){
				for(start[5]=0;start[5]<2;start[5]++){
				  for(len[5]=1;len[5]<=SIZE;len[5]++){
				    for(stride[5]=1;stride[5]<4;stride[5]++){

				      if(start[0]+(len[0]-1)*stride[0] < SIZE && start[1]+(len[1]-1)*stride[1] < SIZE
					 && start[2]+(len[2]-1)*stride[2] < SIZE && start[3]+(len[3]-1)*stride[3] < SIZE
					 && start[4]+(len[4]-1)*stride[4] < SIZE && start[5]+(len[5]-1)*stride[5] < SIZE){
			    
					for(int i=0;i<SIZE;i++)
					  for(int j=0;j<SIZE;j++)
					    for(int k=0;k<SIZE;k++)
					      for(int m=0;m<SIZE;m++)
						for(int n=0;n<SIZE;n++)
						  for(int p=0;p<SIZE;p++)
						    a[i][j][k][m][n][p] = -1;
			    
					xmp_sync_all(&status);
			    
					if(xmp_node_num() == 1){
					  a[start[0]:len[0]:stride[0]][start[1]:len[1]:stride[1]][start[2]:len[2]:stride[2]][start[3]:len[3]:stride[3]][start[4]:len[4]:stride[4]][start[5]:len[5]:stride[5]]:[2] 
					    = b[start[0]:len[0]:stride[0]][start[1]:len[1]:stride[1]][start[2]:len[2]:stride[2]][start[3]:len[3]:stride[3]][start[4]:len[4]:stride[4]][start[5]:len[5]:stride[5]];
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
						  for(int n=0;n<len[4];n++){
						    int position4 = start[4]+n*stride[4];
						    for(int p=0;p<len[5];p++){
						      int position5 = start[5]+p*stride[5];
						      if(a[position0][position1][position2][position3][position4][position5] != 1){
							fprintf(stderr, "a[%d:%d:%d][%d:%d:%d][%d:%d:%d][%d:%d:%d][%d:%d:%d][%d:%d:%d] ", 
								start[0], len[0], stride[0], start[1], len[1], stride[1], start[2], len[2], stride[2],
								start[3], len[3], stride[3], start[4], len[4], stride[4], start[5], len[5], stride[5]);
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
#pragma xmp bcast(ret) from p2(2)
  if(xmp_node_num() == 1)
    if(ret == 0) printf("PASS\n");
    else fprintf(stderr, "ERROR\n");

  return ret;
}

