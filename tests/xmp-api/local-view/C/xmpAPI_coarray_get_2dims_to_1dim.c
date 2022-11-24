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

void get1(xmp_desc_t snd_desc, xmp_desc_t rcv_desc, int bufsize);

int main(int argc, char *argv[])
{
  int start[DIMS], len[DIMS], stride[DIMS];
  int img_dims[1];
  long a_dims[DIMS], b_dims[DIMS];
  // xmp_array_section_t *a_section, *b_section;

  xmp_api_init(argc,argv);

  a_dims[0] = SIZE; a_dims[1] = SIZE;
  b_dims[0] = SIZE; b_dims[1] = SIZE;
  a_desc = xmp_new_coarray(sizeof(int), DIMS,a_dims,1,img_dims,(void **)&a_p);
  b_desc = xmp_new_coarray(sizeof(int), DIMS,b_dims,1,img_dims,(void **)&b_p);

  ret = 0;
  for(int i=0;i<SIZE;i++)
    for(int j=0;j<SIZE;j++)
      (*b_p)[i][j] = xmpc_this_image();

  for(int i=0;i<SIZE;i++)
    for(int j=0;j<SIZE;j++)
      (*a_p)[i][j] = -1;

  xmp_sync_all(NULL);

  get1(b_desc,a_desc,SIZE*SIZE);

  xmp_coarray_deallocate(a_desc);
  xmp_coarray_deallocate(b_desc);

  xmp_api_finalize();
}

void get1(xmp_desc_t src_desc, xmp_desc_t dst_desc, int bufsize)
{
  int ret;
  int start = 0;
  int len = bufsize;
  int stride = 1;
  long int dims[1];
  int img_dims[1];
  int (*snd_p)[], (*rcv_p)[];
  xmp_desc_t snd_desc, rcv_desc;
  xmp_array_section_t *snd_sec, *rcv_sec;

  ret = 0;
  dims[0] = bufsize;
  snd_desc = xmp_reshape_coarray(src_desc, sizeof(int), 1,dims,1,img_dims,(void **)&snd_p);
  rcv_desc = xmp_reshape_coarray(dst_desc, sizeof(int), 1,dims,1,img_dims,(void **)&rcv_p);
  // printf("get1 reshape  snd_p=%p, rcv_p=%p...\n",snd_p, rcv_p);

  if(xmpc_this_image() == 0){
    /* bufrcv[start:len:stride] = bufsnd[start:len:stride]:[2]; */
    rcv_sec = xmp_new_array_section(1);
    xmp_array_section_set_triplet(rcv_sec,0,0,bufsize,1);
    snd_sec = xmp_new_array_section(1);
    xmp_array_section_set_triplet(snd_sec,0,0,bufsize,1);

    img_dims[0] = 1;
    xmp_coarray_get(img_dims,snd_desc,snd_sec, rcv_desc,rcv_sec);

    xmp_free_array_section(rcv_sec);
    xmp_free_array_section(snd_sec);
  }

  if(xmpc_this_image() == 0){
    for(int i = 0; i < bufsize; i++){
      if((*rcv_p)[i] != 1) {
	ret = -1;
	goto end;
      }
    }
  }

 end:
  if(xmpc_this_image() == 0)
    if(ret == 0) printf("PASS\n");
    else fprintf(stderr, "ERROR\n");
  xmp_barrier();
  
  xmp_coarray_deallocate(rcv_desc);
  xmp_coarray_deallocate(snd_desc);
}
