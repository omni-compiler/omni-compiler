struct test{int n;};
typedef struct test __attribute__((aligned)) test_t;
extern int hoge(test_t a);

#include <pthread.h>
