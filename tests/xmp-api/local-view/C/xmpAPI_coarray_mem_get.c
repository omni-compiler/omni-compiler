#include <stdio.h>
#include <string.h>
#include <xmp.h>
#include <xmp_api.h>

#define SIZE 10

xmp_desc_t v_desc;
int *v_p;
int v1, v2;

xmp_desc_t s_desc;
char *s_p;

char s1[SIZE], s2[SIZE];

int main(int argc, char *argv[])
{
  int img_dims[1];
  int i, my_image;

  xmp_api_init(argc,argv);

  v1 = 100; v2 = 200;
  strncpy(s1,"123456789",10);
  strncpy(s2,"abcdefghi",10);

  v_desc = xmp_new_coarray_mem(sizeof(int), 1, img_dims,(void **)&v_p);
  *((int *)v_p) = 200;

  s_desc = xmp_new_coarray_mem(SIZE, 1, img_dims,(void **)&s_p);
  strncpy(s_p,"abcdefghi",10);


  xmp_sync_all(NULL);
  
  my_image = xmpc_this_image();
  
  if(my_image == 1) {
      img_dims[0] = 0;
      xmp_coarray_mem_get(img_dims,v_desc,sizeof(int),&v1);
      xmp_coarray_mem_get(img_dims,s_desc,5,s1);
  }

  xmp_sync_all(NULL);
  
  if(my_image == 1){
    // printf("v1: %d\n",v1);
    // printf("s1: %s\n",s1);
    if(v1 == v2 && strncmp(s1,s2,5) == 0)
      printf("PASS\n");
    else fprintf(stderr, "ERROR\n");
  }

  xmp_coarray_deallocate(v_desc);
  xmp_coarray_deallocate(s_desc);

  xmp_api_finalize();
}
