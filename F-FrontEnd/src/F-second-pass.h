#ifndef _F_SECOND_PASS_H
#define _F_SECOND_PASS_H

#define SP_ERR_NONE 0
#define SP_ERR_UNDEF_TYPE_VAR 1
#define SP_ERR_CHAR_LEN 2
#define SP_ERR_FATAL 3
#define SP_ERR_UNDEF_TYPE_FUNC 4

extern void second_pass_init();
extern int second_pass();
extern void sp_link_id(ID id, int err_no, lineno_info *line);
extern void sp_link_expr(expr ep, int err_no, lineno_info *line);

#endif
