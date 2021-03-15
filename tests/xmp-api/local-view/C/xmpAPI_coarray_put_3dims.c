#include <stdio.h>
#include <stdbool.h>
#include <xmp.h>
// Fj start 202102
#include <xmp_api.h>
//#pragma xmp nodes p[*]
// Fj end 202102

#define SIZE 10
#define DIMS 3

// Fj start 202102
//int a[SIZE][SIZE][SIZE]:[*], b[SIZE][SIZE][SIZE]:[*], ret = 0;
xmp_desc_t a_desc,b_desc;
int (*a_p)[SIZE][SIZE][SIZE], (*b_p)[SIZE][SIZE][SIZE];
int ret = 0;
// Fj end 202102
int start[DIMS], len[DIMS], stride[DIMS];

// Fj start 202102
//int main(){
int main(int argc, char *argv[]){
  int img_dims[1];
  long a_dims[DIMS], b_dims[DIMS];
  xmp_array_section_t *a_section, *b_section;

  xmp_api_init(argc,argv);

  a_dims[0] = SIZE; a_dims[1] = SIZE; a_dims[2] = SIZE;
  b_dims[0] = SIZE; b_dims[1] = SIZE; b_dims[2] = SIZE;
  a_desc = xmp_new_coarray(sizeof(int), DIMS,a_dims,1,img_dims,(void **)&a_p);
  b_desc = xmp_new_coarray(sizeof(int), DIMS,b_dims,1,img_dims,(void **)&b_p);
// Fj end 202102

  for(int i=0;i<SIZE;i++)
    for(int j=0;j<SIZE;j++)
      for(int k=0;k<SIZE;k++)
        // Fj start 202102
	//b[i][j][k] = xmpc_this_image();
	(*b_p)[i][j][k] = xmpc_this_image();
        // Fj end 202102

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
                            // Fj start 202102
			    //a[i][j][k] = -1;
			    (*a_p)[i][j][k] = -1;
                            // Fj end 202102

		      xmp_sync_all(NULL);

		      if(xmpc_this_image() == 0){
                        // Fj start 202102
			//a[start[0]:len[0]:stride[0]][start[1]:len[1]:stride[1]][start[2]:len[2]:stride[2]]:[1] 
			//  = b[start[0]:len[0]:stride[0]][start[1]:len[1]:stride[1]][start[2]:len[2]:stride[2]];
                        a_section = xmp_new_array_section(3);
                        xmp_array_section_set_triplet(a_section,0,start[0],len[0],stride[0]);
                        xmp_array_section_set_triplet(a_section,1,start[1],len[1],stride[1]);
                        xmp_array_section_set_triplet(a_section,2,start[2],len[2],stride[2]);
                        b_section = xmp_new_array_section(3);
                        xmp_array_section_set_triplet(b_section,0,start[0],len[0],stride[0]);
                        xmp_array_section_set_triplet(b_section,1,start[1],len[1],stride[1]);
                        xmp_array_section_set_triplet(b_section,2,start[2],len[2],stride[2]);
                        img_dims[0] = 1;
                        xmp_coarray_put(img_dims,a_desc,a_section,b_desc,b_section);
                        xmp_free_array_section(a_section);
                        xmp_free_array_section(b_section);
                        // Fj end 202102
		      }

		      xmp_sync_all(NULL);

		      if(xmpc_this_image() == 1){
			for(int i=0;i<len[0];i++){
			  int position0 = start[0]+i*stride[0];
			  for(int j=0;j<len[1];j++){
			    int position1 = start[1]+j*stride[1];
			    for(int k=0;k<len[2];k++){
			      int position2 = start[2]+k*stride[2];
                              // Fj start 202102
			      //if(a[position0][position1][position2] != 0){
			      if((*a_p)[position0][position1][position2] != 0){
                              // Fj end 202102
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
  xmp_sync_all(NULL);
  // Fj start 202102
  //#pragma xmp bcast(ret) from p[1]
  // Fj end 202102
  if(xmpc_this_image() == 0)
    if(ret == 0) printf("PASS\n");
    else fprintf(stderr, "ERROR\n");

  // Fj start 202102
  xmp_coarray_deallocate(a_desc);
  xmp_coarray_deallocate(b_desc);

  xmp_api_finalize();
  // Fj end 202102

  return ret;
}

