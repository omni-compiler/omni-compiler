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

int scalar_get()
{
  // Fj start 202102
  int start, len, stride;
  xmp_array_section_t *a_section, *b_section, *c_section;
  xmp_array_section_t *a_l_section, *b_l_section, *c_l_section;

  a_section = xmp_new_array_section(1);
  b_section = xmp_new_array_section(2);
  c_section = xmp_new_array_section(3);
  a_l_section = xmp_new_array_section(1);

  //a_normal[0]        = a[0]:[0];
  start = 0; len = 1; stride = 1;
  xmp_array_section_set_triplet(a_section,0,start,len,stride);
  xmp_array_section_set_triplet(a_l_section,0,start,len,stride);
  img_dims[0] = 0;
  xmp_coarray_get_local(img_dims,a_desc,a_section,a_local,a_l_section);

  //b[1][2]            = a[3]:[0];
  start = 3; len = 1; stride = 1;
  xmp_array_section_set_triplet(a_section,0,start,len,stride);
  start = 1; //len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,0,start,len,stride);
  start = 2; //len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,1,start,len,stride);
  xmp_coarray_get(img_dims,a_desc,a_section,b_desc,b_section);

  //b[2:3:2][2]        = a[4]:[0];
  start = 4; len = 1; stride = 1;
  xmp_array_section_set_triplet(a_section,0,start,len,stride);
  start = 2; len = 3; stride = 2;
  xmp_array_section_set_triplet(b_section,0,start,len,stride);
  start = 2; len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,1,start,len,stride);
  xmp_coarray_get(img_dims,a_desc,a_section,b_desc,b_section);

  //a_normal[1]        = b[0][3]:[0];
  start = 0; len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,0,start,len,stride);
  start = 3; //len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,1,start,len,stride);
  start = 1; //len = 1; stride = 1;
  xmp_array_section_set_triplet(a_l_section,0,start,len,stride);
  xmp_coarray_get_local(img_dims,b_desc,b_section,a_local,a_l_section);

  //c[1][2][2]         = b[3][2]:[0];
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
  xmp_coarray_get(img_dims,b_desc,b_section,c_desc,c_section);
  
  //c[2:2:2][2:2:3][1] = b[4][2]:[0];
  start = 4; len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,0,start,len,stride);
  start = 2; //len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,1,start,len,stride);
  start = 2; len = 2; stride = 2;
  xmp_array_section_set_triplet(c_section,0,start,len,stride);
  start = 2; len = 2; stride = 3;
  xmp_array_section_set_triplet(c_section,1,start,len,stride);
  start = 1; len = 1; stride = 1;
  xmp_array_section_set_triplet(c_section,2,start,len,stride);
  xmp_coarray_get(img_dims,b_desc,b_section,c_desc,c_section);

  xmp_free_array_section(a_section);
  xmp_free_array_section(b_section);
  xmp_free_array_section(c_section);
  xmp_free_array_section(a_l_section);
  // Fj end 202102
  
  int flag = TRUE;
  // Fj start 202102
  //if(a_normal[0] != a[0]) flag = FALSE;
  if(a_normal[0] != (*a_p)[0]) flag = FALSE;
  //if(b[1][2]     != a[3]) flag = FALSE;
  if((*b_p)[1][2]     != (*a_p)[3]) flag = FALSE;
  //if(b[2][2] != a[4] || b[4][2] != a[4] || b[6][2] != a[4]) flag = FALSE;
  if((*b_p)[2][2] != (*a_p)[4] || (*b_p)[4][2] != (*a_p)[4] ||
     (*b_p)[6][2] != (*a_p)[4]) flag = FALSE;
  //if(a_normal[1] != b[0][3]) flag = FALSE;
  if(a_normal[1] != (*b_p)[0][3]) flag = FALSE;
  //if(c[1][2][2] != b[3][2]) flag = FALSE;
  if((*c_p)[1][2][2] != (*b_p)[3][2]) flag = FALSE;
  //if(c[2][2][1] != b[4][2] || c[2][5][1] != b[4][2] || 
  //   c[4][2][1] != b[4][2] || c[4][5][1] != b[4][2]) flag = FALSE;
  if((*c_p)[2][2][1] != (*b_p)[4][2] || (*c_p)[2][5][1] != (*b_p)[4][2] || 
     (*c_p)[4][2][1] != (*b_p)[4][2] || (*c_p)[4][5][1] != (*b_p)[4][2])
    flag = FALSE;
  // Fj end 202102

  return flag;
}

int vector_get()
{
  // Fj start 202102
  int start, len, stride;
  xmp_array_section_t *a_section, *b_section, *c_section;
  xmp_array_section_t *a_l_section;

  a_section = xmp_new_array_section(1);
  b_section = xmp_new_array_section(2);
  c_section = xmp_new_array_section(3);
  a_l_section = xmp_new_array_section(1);

  //a_normal[2:5]  = a[0:5]:[0];
  start = 0; len = 5; stride = 1;
  xmp_array_section_set_triplet(a_section,0,start,len,stride);
  start = 2; len = 5; stride = 1;
  xmp_array_section_set_triplet(a_l_section,0,start,len,stride);
  img_dims[0] = 0;
  xmp_coarray_get_local(img_dims,a_desc,a_section,a_local,a_l_section);

  //a[5:4]         = b[0:4][3]:[0];
  start = 0; len = 4; stride = 1;
  xmp_array_section_set_triplet(b_section,0,start,len,stride);
  start = 3; len = 1; stride = 1;
  xmp_array_section_set_triplet(b_section,1,start,len,stride);
  start = 5; len = 4; stride = 1;
  xmp_array_section_set_triplet(a_section,0,start,len,stride);
  xmp_coarray_get(img_dims,b_desc,b_section,a_desc,a_section);

  //c[1:2:3][2][2] = b[3][2:2:2]:[0];
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
  xmp_coarray_get(img_dims,b_desc,b_section,c_desc,c_section);
  // Fj end 202102

  int flag = TRUE;
  for(int i=0;i<5;i++)
    // Fj start 202102
    //if(a_normal[2+i] != a[i])
    if(a_normal[2+i] != (*a_p)[i])
    // Fj end 202102
      flag = FALSE;

  for(int i=0;i<4;i++)
    // Fj start 202102
    //if(a[5+i] != b[i][3])
    if((*a_p)[5+i] != (*b_p)[i][3])
    // Fj end 202102
      flag = FALSE;

  // Fj start 202102
  //if(c[1][2][2] != b[3][2] || c[4][2][2] != b[3][4])
  if((*c_p)[1][2][2] != (*b_p)[3][2] || (*c_p)[4][2][2] != (*b_p)[3][4])
  // Fj end 202102
    flag = FALSE;

  // Fj start 202102
  xmp_free_array_section(a_section);
  xmp_free_array_section(b_section);
  xmp_free_array_section(c_section);
  xmp_free_array_section(a_l_section);
  // Fj end 202102

  return flag;
}

// Fj start 202102
//int main()
//{
int main(int argc, char *argv[]){
  //int img_dims[1];
  long a_dims[1], b_dims[2], c_dims[3];
  xmp_array_section_t *a_section, *b_section, *c_section;

  xmp_init_all(argc,argv);

  a_dims[0] = I;
  b_dims[0] = I; b_dims[1] = J;
  c_dims[0] = I; c_dims[1] = J; c_dims[2] = K;
  a_desc = xmp_new_coarray(sizeof(int), 1,a_dims,1,img_dims,(void **)&a_p);
  b_desc = xmp_new_coarray(sizeof(int), 2,b_dims,1,img_dims,(void **)&b_p);
  c_desc = xmp_new_coarray(sizeof(int), 3,c_dims,1,img_dims,(void **)&c_p);
  a_local = xmp_new_local_array(sizeof(int), 1,a_dims,(void **)&a_normal);
// Fj end 202102

  initialize();
  test_ok(scalar_get());

  // Fj start 202102
  initialize();
  // Fj end 202102
  test_ok(vector_get());

  // Fj start 202102
  xmp_coarray_deallocate(a_desc);
  xmp_coarray_deallocate(b_desc);
  xmp_coarray_deallocate(c_desc);
  xmp_free_local_array(a_local);

  xmp_finalize_all();
  // Fj end 202102

  return 0;
}
