#include <stdio.h>
#include <xmp.h>
#include <xmp_api.h>

// #pragma xmp nodes p[*]

#define SIZE 10
#define DIMS 2
// int a[SIZE][SIZE]:[*], b[SIZE][SIZE]:[*], ret = 0;

xmp_desc_t a_desc,b_desc;
int (*a_p)[SIZE][SIZE], (*b_p)[SIZE][SIZE];
int ret = 0;

int main(int argc, char *argv[]){
  int start[DIMS], len[DIMS], stride[DIMS];
  int img_dims[1];
  long a_dims[DIMS], b_dims[DIMS];
  xmp_array_section_t *a_section, *b_section;

  xmp_init_all(argc,argv);

  a_dims[0] = SIZE; a_dims[1] = SIZE;
  b_dims[0] = SIZE; b_dims[1] = SIZE;
  a_desc = xmp_new_coarray(sizeof(int), DIMS,a_dims,1,img_dims,(void **)&a_p);
  b_desc = xmp_new_coarray(sizeof(int), DIMS,b_dims,1,img_dims,(void **)&b_p);
  
  for(int i=0;i<SIZE;i++)
    for(int j=0;j<SIZE;j++)
      (*b_p)[i][j] = xmpc_this_image();

  for(start[0]=0;start[0]<2;start[0]++){
    for(len[0]=1;len[0]<=SIZE;len[0]++){
      for(stride[0]=1;stride[0]<4;stride[0]++){
	for(start[1]=0;start[1]<2;start[1]++){
	  for(len[1]=1;len[1]<=SIZE;len[1]++){
	    for(stride[1]=1;stride[1]<4;stride[1]++){

	      if(start[0]+(len[0]-1)*stride[0] < SIZE && start[1]+(len[1]-1)*stride[1] < SIZE){
		for(int i=0;i<SIZE;i++)
		  for(int j=0;j<SIZE;j++)
		    (*a_p)[i][j] = -1;

		xmp_sync_all(NULL);

		if(xmpc_this_image() == 0){  // if node == 0
		  // a[start[0]:len[0]:stride[0]][start[1]:len[1]:stride[1]] 
		  // = b[start[0]:len[0]:stride[0]][start[1]:len[1]:stride[1]]:[1];

		  a_section = xmp_new_array_section(2);
		  xmp_array_section_set_triplet(a_section,0,start[0],len[0],stride[0]);
		  xmp_array_section_set_triplet(a_section,1,start[1],len[1],stride[1]);
		  b_section = xmp_new_array_section(2);
		  xmp_array_section_set_triplet(b_section,0,start[0],len[0],stride[0]);
		  xmp_array_section_set_triplet(b_section,1,start[1],len[1],stride[1]);
		  img_dims[0] = 1;
		  xmp_coarray_get(img_dims,b_desc,b_section,a_desc,a_section);
		  xmp_free_array_section(a_section);
		  xmp_free_array_section(b_section);

		  for(int i=0;i<len[0];i++){
		    for(int j=0;j<len[1];j++){
		      int position0 = start[0]+i*stride[0];
		      int position1 = start[1]+j*stride[1];
		      if(/*a[position0][position1]*/ (*a_p)[position0][position1] != 1){
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
  xmp_sync_all(NULL);

  // #pragma xmp bcast(ret)
  xmp_barrier();

  if(xmpc_this_image() == 0)
    if(ret == 0) printf("PASS\n");
    else fprintf(stderr, "ERROR\n");

  xmp_coarray_deallocate(a_desc);
  xmp_coarray_deallocate(b_desc);

  xmp_finalize_all();
  return ret;
}
