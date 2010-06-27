# 1 "struct.c"
struct point {
int x;
int * y;
};
struct point * p;
int main()
{
(*(& p->x)) = (0);
(p->x) = (0);
(*(*(& p->y))) = (0);
return 0;
}

