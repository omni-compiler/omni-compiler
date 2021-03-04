#include <stdio.h>
#include <stdlib.h>
// Fj start 202102
#include <xmp_api.h>
// Fj end 202102

#define I 10
#define J 20
#define K 30
#define TRUE 1
#define FALSE 0

// Fj start 202102
//int a[I]:[*], b[I][J]:[*], c[I][J][K]:[*];
int (*a_p)[I], (*b_p)[I][J], (*c_p)[I][J][K];
xmp_desc_t a_desc, b_desc, c_desc;
xmp_local_array_t *a_local, *b_local, *c_local;
int ret = 0;
int img_dims[1];
// Fj end 202102
int a_normal[I], b_normal[I][J], c_normal[I][J][K];

void test_ok(int flag)
{
  if(flag){
    printf("PASS\n");
  }
  else{
    printf("ERROR\n");
    exit(1);
  }
}

void initialize()
{
  for(int i=0;i<I;i++){
    // Fj start 202102
    //a[i] = a_normal[i] = i;
    (*a_p)[i] = a_normal[i] = i;
    // Fj end 202102
    for(int j=0;j<J;j++){
      // Fj start 202102
      //b[i][j] = b_normal[i][j] = (i * I) + j;
      (*b_p)[i][j] = b_normal[i][j] = (i * I) + j;
      // Fj end 202102
      for(int k=0;k<J;k++){
        // Fj start 202102
	//c[i][j][k] = c_normal[i][j][k] = (i * I * J) + (j * J) + k;
        (*c_p)[i][j][k] = c_normal[i][j][k] = (i * I * J) + (j * J) + k;
        // Fj end 202102
      }
    }
  }  
}

int scalar_put()
{
  // Fj start 202102
  int start, len, stride;
  xmp_array_section_t *a_section, *b_section, *c_section;
  xmp_array_section_t *a_l_section, *b_l_section, *c_l_section;

  a_section = xmp_new_array_section(1);
  b_section = xmp_new_array_section(2);
  c_section = xmp_new_array_section(3);
  a_l_section = xmp_new_array_section(1);

  //a[0]:[0]            = a_normal[0];
  start = 0; len = 1; stride = 1;
  xmp_array_section_set_triplet(a_section,0,start,len,stride);
  xmp_array_section_set_triplet(a_l_section,0,start,len,stride);
  img_dims[0] = 0;
  xmp_coarray_put_local(img_dims,a_desc,a_section,a_local,a_l_section);

  //a[3]:[0]            = b[1][2];
  start = 3; len = 1; stride = 1;
  xmp_array_section_set_triplet(a_section,0,start,len,stride);
  start = 1; //len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,0,start,len,stride);
  start = 2; //len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,1,start,len,stride);
  xmp_coarray_put(img_dims,a_desc,a_section,b_desc,b_section);

  //a[4:2:2]:[0]        = b[2][2];
  start = 4; len = 2; stride = 2;
  xmp_array_section_set_triplet(a_section,0,start,len,stride);
  start = 2; len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,0,start,len,stride);
  start = 2; //len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,1,start,len,stride);
  xmp_coarray_put(img_dims,a_desc,a_section,b_desc,b_section);

  //b[0][3]:[0]         = a_normal[0];
  start = 0; len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,0,start,len,stride);
  start = 3; //len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,1,start,len,stride);
  start = 0; //len = 1; stride = 1;
  xmp_array_section_set_triplet(a_l_section,0,start,len,stride);
  xmp_coarray_put_local(img_dims,b_desc,b_section,a_local,a_l_section);

  //b[3][2]:[0]         = c[1][2][2];
  start = 3; len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,0,start,len,stride);
  start = 2; //len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,1,start,len,stride);
  start = 1; //len = 1; stride = 1;
  xmp_array_section_set_triplet(c_section,0,start,len,stride);
  start = 2; //len = 1; stride = 1;
  xmp_array_section_set_triplet(c_section,1,start,len,stride);
  start = 2; //len = 1; stride = 1;
  xmp_array_section_set_triplet(c_section,2,start,len,stride);
  xmp_coarray_put(img_dims,b_desc,b_section,c_desc,c_section);

  //b[4:2:2][2:2:4]:[0] = c[2][2][1];
  start = 4; len = 2; stride = 2;
  xmp_array_section_set_triplet(b_section,0,start,len,stride);
  start = 2; len = 2; stride = 4;
  xmp_array_section_set_triplet(b_section,1,start,len,stride);
  start = 2; len = 1; stride = 1;
  xmp_array_section_set_triplet(c_section,0,start,len,stride);
  start = 2; //len = 1; stride = 1;
  xmp_array_section_set_triplet(c_section,1,start,len,stride);
  start = 1; //len = 1; stride = 1;
  xmp_array_section_set_triplet(c_section,2,start,len,stride);
  xmp_coarray_put(img_dims,b_desc,b_section,c_desc,c_section);

  xmp_free_array_section(a_section);
  xmp_free_array_section(b_section);
  xmp_free_array_section(c_section);
  xmp_free_array_section(a_l_section);
  // Fj end 202102

  int flag = TRUE;
  // Fj start 202102
  //if(a[0] != a_normal[0]) flag = FALSE;
  if((*a_p)[0] != a_normal[0]) flag = FALSE;
  //if(a[3] != b[1][2]) flag = FALSE;
  if((*a_p)[3] != (*b_p)[1][2]) flag = FALSE;
  //if(a[4] != b[2][2] || a[6] != b[2][2]) flag = FALSE;
  if((*a_p)[4] != (*b_p)[2][2] || (*a_p)[6] != (*b_p)[2][2]) flag = FALSE;
  //if(b[0][3] != a_normal[0]) flag = FALSE;
  if((*b_p)[0][3] != a_normal[0]) flag = FALSE;
  //if(b[3][2] != c[1][2][2]) flag = FALSE;
  if((*b_p)[3][2] != (*c_p)[1][2][2]) flag = FALSE;
  //if(b[4][2] != c[2][2][1] || b[4][6] != c[2][2][1] ||
  //   b[6][2] != c[2][2][1] || b[6][6] != c[2][2][1]) flag = FALSE;
  if((*b_p)[4][2] != (*c_p)[2][2][1] || (*b_p)[4][6] != (*c_p)[2][2][1] ||
     (*b_p)[6][2] != (*c_p)[2][2][1] || (*b_p)[6][6] != (*c_p)[2][2][1])
    flag = FALSE;
  // Fj end 202102

  return flag;
}

int vector_put()
{
  // Fj start 202102
  int start, len, stride;
  xmp_array_section_t *a_section, *b_section, *c_section;
  xmp_array_section_t *a_l_section;

  a_section = xmp_new_array_section(1);
  b_section = xmp_new_array_section(2);
  c_section = xmp_new_array_section(3);
  a_l_section = xmp_new_array_section(1);

  //a[0:5]:[0]      = a_normal[2:5];
  start = 0; len = 5; stride = 1;
  xmp_array_section_set_triplet(a_section,0,start,len,stride);
  start = 2; len = 5; stride = 1;
  xmp_array_section_set_triplet(a_l_section,0,start,len,stride);
  img_dims[0] = 0;
  xmp_coarray_put_local(img_dims,a_desc,a_section,a_local,a_l_section);

  //b[0:4][3]:[0]   = a[5:4];
  start = 0; len = 4; stride = 1;
  xmp_array_section_set_triplet(b_section,0,start,len,stride);
  start = 3; len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,1,start,len,stride);
  start = 5; len = 4; stride = 1;
  xmp_array_section_set_triplet(a_section,0,start,len,stride);
  xmp_coarray_put(img_dims,b_desc,b_section,a_desc,a_section);

  //b[3][2:2:2]:[0] = c[1:2:3][2][2];
  start = 3; len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,0,start,len,stride);
  start = 2; len = 2; stride = 2;
  xmp_array_section_set_triplet(b_section,1,start,len,stride);
  start = 1; len = 2; stride = 3;
  xmp_array_section_set_triplet(c_section,0,start,len,stride);
  start = 2; len = 1; stride = 1;
  xmp_array_section_set_triplet(c_section,1,start,len,stride);
  start = 2; len = 1; stride = 1;
  xmp_array_section_set_triplet(c_section,2,start,len,stride);
  xmp_coarray_put(img_dims,b_desc,b_section,c_desc,c_section);
  // Fj end 202102

  int flag = TRUE;
  for(int i=0;i<5;i++)
    // Fj start 202102
    //if(a[i] != a_normal[2+i])
    if((*a_p)[i] != a_normal[2+i])
    // Fj end 202102
      flag = FALSE;

  for(int i=0;i<4;i++)
    // Fj start 202102
    //if(b[i][3] != a[5+i])
    if((*b_p)[i][3] != (*a_p)[5+i])
    // Fj end 202102
      flag = FALSE;

  // Fj start 202102
  //if(b[3][2] != c[1][2][2] || b[3][4] != c[4][2][2])
  if((*b_p)[3][2] != (*c_p)[1][2][2] || (*b_p)[3][4] != (*c_p)[4][2][2])
  // Fj end 202102
    flag = FALSE;

  return flag;
}

// Fj start 202102
//int main()
int main(int argc, char *argv[])
{
  long a_dims[1], b_dims[2], c_dims[3];
  xmp_array_section_t *a_section, *b_section, *c_section;

  xmp_api_init(argc,argv);

  a_dims[0] = I;
  b_dims[0] = I; b_dims[1] = J;
  c_dims[0] = I; c_dims[1] = J; c_dims[2] = K;
  a_desc = xmp_new_coarray(sizeof(int), 1,a_dims,1,img_dims,(void **)&a_p);
  b_desc = xmp_new_coarray(sizeof(int), 2,b_dims,1,img_dims,(void **)&b_p);
  c_desc = xmp_new_coarray(sizeof(int), 3,c_dims,1,img_dims,(void **)&c_p);
  a_local = xmp_new_local_array(sizeof(int), 1,a_dims,(void **)&a_normal);
// Fj end 202102

  initialize();
  test_ok(scalar_put());

  initialize();
  test_ok(vector_put());

  // Fj start 202102
  xmp_coarray_deallocate(a_desc);
  xmp_coarray_deallocate(b_desc);
  xmp_coarray_deallocate(c_desc);
  xmp_free_local_array(a_local);

  xmp_api_finalize();
  // Fj end 202102

  return 0;
}
