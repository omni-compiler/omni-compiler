
#define OFFSET 10

int printf(const char *, ...);
int omp_get_thread_num();
void usleep(unsigned long);

void f(int n, int a[n])
{
    int i, t;
    int ok = 1;

    #pragma omp parallel firstprivate(a) private(i, t)
    {
        t = omp_get_thread_num();
        //printf("%d: a=0x%08x, n=%d\n", t, (unsigned int)a, n);
        for(i = 0; i < n; ++i) {
            a[i] += t;
            usleep(10);
        }

        #pragma omp barrier

        for(i = 0; i < n; ++i) {
            usleep(10);
            if(a[i] != t + OFFSET) {
                ok = 0;
            }
        }
    }

    if(ok) {
        printf("vla 004 : SUCCESS\n");
    } else {
        printf("vla 004 : FAILED\n");
    }
}

int main()
{
    int i, n = 100;
    int a[n];

    for(i = 0; i < n; ++i)
        a[i] = OFFSET;

    f(n, a);
    return 0;
}

