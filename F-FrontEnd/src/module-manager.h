
/**
 * \file module-manager.h
 *
 * This file should include module manager interfaces.
 * Don't include compiler funtions.
 */

#ifndef _MODULE_MANAGER_H_
#define _MODULE_MANAGER_H_

#include "C-expr.h"  // SYMBOL
#include "F-ident.h" // ID

/**
 * list of module name.
 */
struct depend_module {
    struct depend_module * next;
    SYMBOL module_name;
};

#define MOD_DEP_NEXT(dep) ((dep)->next)
#define MOD_DEP_NAME(dep) ((dep)->module_name)


/**
 * fortran module.
 */
struct module {
    struct module * next;
    struct {
        struct depend_module * head;
        struct depend_module * last;
    } depend;                 /* list of module name which this module depends on. */
    SYMBOL name;              /* name of this module. */
    ID head;                  /* public elements of this module. */
    ID last;
    int is_intrinsic;         /* TRUE if this module is an intrinsic module. */
};

#define MODULE_NEXT(mod)          ((mod)->next)
#define MODULE_NAME(mod)          ((mod)->name)
#define MODULE_DEPEND_HEAD(mod)   ((mod)->depend.head)
#define MODULE_DEPEND_LAST(mod)   ((mod)->depend.last)
#define MODULE_ID_LIST(mod)       ((mod)->head)
#define MODULE_ID_LIST_LAST(mod)  ((mod)->last)
#define MODULE_IS_INTRINSIC(mod)  ((mod)->is_intrinsic)



/**
 * import module form module manager.
 */
int import_module(const SYMBOL, struct module **);

/**
 * export public identifiers to module-manager.
 */
int export_module(const SYMBOL, ID, expv);

#endif /* _MODULE_MANAGER_H_ */
