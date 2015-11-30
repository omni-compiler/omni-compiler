#ifndef _F_SECOND_PASS_H
#define _F_SECOND_PASS_H

extern void second_pass_init();
extern int second_pass();
extern void sp_link_id(ID id, int err_no, lineno_info *line);
extern void sp_link_expr(expr ep, int err_no, lineno_info *line);

#endif
