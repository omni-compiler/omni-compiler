#include <stdio.h>
#include <xmp.h>
#include <xmp_api.h>

#define SIZE 10
#define DIMS 1

xmp_desc_t a_desc;
int (*a_p)[SIZE];

xmp_local_array_t *b_local;
int b[SIZE];

int main(int argc, char *argv[])
{
  xmp_array_section_t *a_sec, *b_sec;
  long a_dims[DIMS], b_dims[DIMS];
  int img_dims[1];
  int i, my_image;

  xmp_api_init(argc,argv);

  a_dims[0] = SIZE;
  a_desc = xmp_new_coarray(sizeof(int), DIMS,a_dims,1,img_dims,(void **)&a_p);
  a_sec = xmp_new_array_section(1);

  b_dims[0] = SIZE;
  b_local = xmp_new_local_array(sizeof(int),DIMS, b_dims, (void *)b);
  b_sec = xmp_new_array_section(1);
  
  my_image = xmpc_this_image();

  for(i = 0; i < SIZE; i++) (*a_p)[i] = 0;
  for(i = 0; i < SIZE; i++) b[i] = i;
    
  xmp_sync_all(NULL);
  
  if(my_image == 0) {
      xmp_array_section_set_triplet(a_sec,0, 0, 8, 1);
      xmp_array_section_set_triplet(b_sec,0, 0, 8, 1);
      img_dims[0] = 1;

      //! local array b of image 0 -> coarray a of image 1
      xmp_coarray_put_local(img_dims,a_desc,a_sec,b_local,b_sec);
  }

  xmp_sync_all(NULL);
  
  if(my_image == 1){
    if((*a_p)[3] == 3 && (*a_p)[5] == 5 && (*a_p)[7] == 7)
      printf("PASS\n");
    else fprintf(stderr, "ERROR\n");
  }

  xmp_free_array_section(a_sec);
  xmp_free_array_section(b_sec);

  xmp_coarray_deallocate(a_desc);
  xmp_free_local_array(b_local);

  xmp_api_finalize();
}
