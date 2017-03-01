
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
    SYMBOL submodule_name;    /* for submodule, name of this submodle */
    ID head;                  /* public elements of this module. */
    ID last;
    int is_intrinsic;         /* TRUE if this module is an intrinsic module. */
    int for_submodule;        /* module for submodule */
};

#define MODULE_IS_MODULE(mod)     ((mod)->submodule_name == NULL)

#define MODULE_NEXT(mod)          ((mod)->next)
#define MODULE_NAME(mod)          ((mod)->name)
#define MODULE_DEPEND_HEAD(mod)   ((mod)->depend.head)
#define MODULE_DEPEND_LAST(mod)   ((mod)->depend.last)
#define MODULE_ID_LIST(mod)       ((mod)->head)
#define MODULE_ID_LIST_LAST(mod)  ((mod)->last)
#define MODULE_IS_INTRINSIC(mod)  ((mod)->is_intrinsic)
#define MODULE_IS_FOR_SUBMODULE(mod)  ((mod)->for_submodule)

#define MODULE_IS_SUBMODULE(mod)  ((mod)->submodule_name != NULL)

#define SUBMODULE_NAME(mod)       ((mod)->submodule_name)
#define SUBMODULE_ANCESTOR(mod)   ((mod)->name)


/**
 * import module form module manager.
 */
int import_module(const SYMBOL, struct module **);

/**
 * import submodule form module manager.
 */
int import_submodule(const SYMBOL, const SYMBOL, struct module **);

/**
 * export public identifiers in the module to module-manager.
 */
int export_module(const SYMBOL, ID, expv);

/**
 * export public identifiers in the submodule to module-manager.
 */
int export_submodule(const SYMBOL, const SYMBOL, ID, expv);

#endif /* _MODULE_MANAGER_H_ */
