#include <stdio.h>
#include <xmp.h>
// Fj start 202102
#include <xmp_api.h>
//#pragma xmp nodes p[*]
// Fj end 202102

#define SIZE 10
#define DIMS 1

// Fj start 202102
//int a[SIZE]:[*], b[SIZE]:[*], ret = 0;
xmp_desc_t a_desc,b_desc;
int (*a_p)[SIZE], (*b_p)[SIZE];
int ret = 0;
// Fj end 202102

// Fj start 202102
//int a[SIZE]:[*], b[SIZE]:[*], ret = 0;
//int main(){
int main(int argc, char *argv[]){
// Fj end 202102
  int start[DIMS], len[DIMS], stride[DIMS];
  // Fj start 202102
  int img_dims[1];
  long a_dims[DIMS], b_dims[DIMS];
  xmp_array_section_t *a_section, *b_section;

  xmp_init_all(argc,argv);

  a_dims[0] = SIZE;
  b_dims[0] = SIZE;
  a_desc = xmp_new_coarray(sizeof(int), DIMS,a_dims,1,img_dims,(void **)&a_p);
  b_desc = xmp_new_coarray(sizeof(int), DIMS,b_dims,1,img_dims,(void **)&b_p);
  // Fj end 202102

  for(int i=0;i<SIZE;i++)
    // Fj start 202102
    //b[i] = xmpc_this_image();
    (*b_p)[i] = xmpc_this_image();
    // Fj end 202102

  for(start[0]=0;start[0]<2;start[0]++){
    for(len[0]=1;len[0]<=SIZE;len[0]++){
      for(stride[0]=1;stride[0]<3;stride[0]++){
	if(start[0]+(len[0]-1)*stride[0] < SIZE){

          // Fj start 202102
	  //for(int i=0;i<SIZE;i++) a[i] = -1;
	  for(int i=0;i<SIZE;i++) (*a_p)[i] = -1;
          // Fj end 202102
	  xmp_sync_all(NULL);

	  if(xmpc_this_image() == 0){
            // Fj start 202102
	    //a[start[0]:len[0]:stride[0]] = b[start[0]:len[0]:stride[0]]:[1];

            a_section = xmp_new_array_section(1);
            xmp_array_section_set_triplet(a_section,0,start[0],len[0],stride[0]);
            b_section = xmp_new_array_section(1);
            xmp_array_section_set_triplet(b_section,0,start[0],len[0],stride[0]);
            img_dims[0] = 1;
            xmp_coarray_get(img_dims,b_desc,b_section,a_desc,a_section);
            xmp_free_array_section(a_section);
            xmp_free_array_section(b_section);
            // Fj end 202102

	    for(int i=0;i<len[0];i++){
	      int position = start[0]+i*stride[0];
              // Fj start 202102
	      //if(a[position] != 1){
	      if((*a_p)[position] != 1){
              // Fj end 202102
		fprintf(stderr, "a[%d:%d:%d] ", start[0], len[0], stride[0]);
		ret = -1;
		goto end;
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
  //#pragma xmp bcast(ret)
  // Fj end 202102

  if(xmpc_this_image() == 0)
    if(ret == 0) printf("PASS\n");
    else fprintf(stderr, "ERROR\n");

  // Fj start 202102
  //xmp_coarray_deallocate(a_desc);
  //xmp_coarray_deallocate(b_desc);

  xmp_finalize_all();
  // Fj end 202102

  return ret;
}

