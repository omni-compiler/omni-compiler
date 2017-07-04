#include "F-front.h"
#include "F-output-xcodeml.h"
#include "F-input-xmod.h"
#include "module-manager.h"

/**
 * \file module-manager.c
 */

/**
 * collection of fortran modules.
 */
struct module_manager {
    struct module * head;
    struct module * tail;
} MODULE_MANAGER;

static struct module *
find_module(const SYMBOL module_name,
            const SYMBOL submodule_name,
            int for_submodule)
{
    struct module * mp;
    assert(module_name != NULL);
    /* submodule_name may be NULL */
    for (mp = MODULE_MANAGER.head; mp != NULL; mp = mp->next) {
        if (MODULE_NAME(mp) == module_name &&
            SUBMODULE_NAME(mp) == submodule_name &&
            MODULE_IS_FOR_SUBMODULE(mp) == for_submodule) {
            return mp;
        }
    }
    return NULL;
}


static void
add_module(struct module * mod)
{
    if (find_module(MODULE_NAME(mod),
                    SUBMODULE_NAME(mod),
                    MODULE_IS_FOR_SUBMODULE(mod)) != NULL) {
        return;
    }

    if(MODULE_MANAGER.head == NULL) {
        MODULE_MANAGER.head = mod;
    }else {
        MODULE_MANAGER.tail->next = mod;
    }
    MODULE_MANAGER.tail = mod;
    MODULE_MANAGER.tail->next = NULL;
}


static void
add_module_id(struct module * mod, ID id)
{
    // NOTE: 'mid' stand for module id.
    TYPE_DESC tp;
    ID mid = XMALLOC(ID, sizeof(*mid));
    *mid = *id;

    // HACK: it seems too dirty
    if (ID_CLASS(mid) == CL_PARAM && ID_STORAGE(mid) == STG_UNKNOWN)
        ID_STORAGE(mid) = STG_SAVE;

    if (mid->use_assoc == NULL) {
        mid->use_assoc = XMALLOC(struct use_assoc_info *, sizeof(*(id->use_assoc)));
        mid->use_assoc->module = mod;
        mid->use_assoc->module_name = MODULE_NAME(mod);
        mid->use_assoc->original_name = id->name;
    }

    /* NOTE: dirty code, but
     *       tagname of type may differs from one in LOCAL_SYMBOLS.
     *       So this code rescues the derived-type comparison.
     */
    tp = ID_TYPE(id);
    while (tp != NULL) {
        if (TYPE_TAGNAME(tp)) {
            ID tagname = TYPE_TAGNAME(tp);
            if (tagname->use_assoc == NULL) {
                tagname->use_assoc = XMALLOC(struct use_assoc_info *, sizeof(*(id->use_assoc)));
                tagname->use_assoc->module = mod;
                tagname->use_assoc->module_name = MODULE_NAME(mod);
                tagname->use_assoc->original_name = tagname->name;
            }
        }
        tp = TYPE_REF(tp);
    }

    ID_LINK_ADD(mid, MODULE_ID_LIST(mod), MODULE_ID_LIST_LAST(mod));
}

#define AVAILABLE_ID(id)                           \
    (ID_CLASS(id)   == CL_NAMELIST ||              \
    (TRUE                                          \
    && (      ID_CLASS(id)   == CL_VAR             \
           || ID_CLASS(id)   == CL_ENTRY           \
           || ID_CLASS(id)   == CL_PARAM           \
           || ID_CLASS(id)   == CL_CONTAINS        \
           || ID_CLASS(id)   == CL_TAGNAME         \
           || ID_CLASS(id)   == CL_UNKNOWN         \
           || ID_CLASS(id)   == CL_GENERICS        \
           || ID_CLASS(id)   == CL_PROC            \
       ) \
    && (      ID_STORAGE(id) != STG_NONE           \
    )))

struct module *
generate_current_module(SYMBOL mod_name, SYMBOL submod_name, ID ids, expv use_decls, int for_submodule)
{
    ID id;
    list lp;
    struct depend_module * dep;
    struct module * mod = XMALLOC(struct module *, sizeof(struct module));
    extern int flag_do_module_cache;

    *mod = (struct module){0};
    MODULE_NAME(mod) = mod_name;
    if (submod_name) {
        SUBMODULE_NAME(mod) = submod_name;
    }
    MODULE_IS_FOR_SUBMODULE(mod) = for_submodule;

    /* add public id */
    FOREACH_ID(id, ids) {
        if(AVAILABLE_ID(id) &&
           (for_submodule || (ID_TYPE(id) && !TYPE_IS_PRIVATE(ID_TYPE(id)))))
            add_module_id(mod, id);
    }

    /* make the list of module name which this module uses. */
    FOR_ITEMS_IN_LIST(lp, use_decls) {
        dep = XMALLOC(struct depend_module *, sizeof(struct depend_module));
        MOD_DEP_NAME(dep) = EXPR_SYM(LIST_ITEM(lp));
        if (MODULE_DEPEND_LAST(mod) == NULL) {
            MODULE_DEPEND_HEAD(mod) = dep;
            MODULE_DEPEND_LAST(mod) = dep;
        } else {
            MOD_DEP_NEXT(MODULE_DEPEND_LAST(mod)) = dep;
            MODULE_DEPEND_LAST(mod) = dep;
        }
    }

    if (flag_do_module_cache == TRUE)
        add_module(mod);

    if (nerrors == 0) {
        return mod;
    } else {
        return NULL;
    }
}


static void
intermediate_file_name(char * filename,
                       size_t len,
                       const struct module * mod,
                       const char * extension)
{
    char tmp[FILE_NAME_LEN];
    extern char *modincludeDirv;
    if (modincludeDirv) {
        snprintf(filename, strnlen(modincludeDirv, FILE_NAME_LEN)+2,
                 "%s/", modincludeDirv);
    }
    if (MODULE_IS_MODULE(mod)) {
        snprintf(tmp, sizeof(tmp), "%s.%s",
                 SYM_NAME(MODULE_NAME(mod)),
                 extension
                 );
    } else { /* mod is submodule */
        snprintf(tmp, sizeof(tmp), "%s:%s.%s",
                 SYM_NAME(SUBMODULE_ANCESTOR(mod)),
                 SYM_NAME(SUBMODULE_NAME(mod)),
                 extension
                 );
    }
    strncat(filename, tmp, len - strnlen(filename, len) - 1);
}


void
xmod_file_name(char * filename, size_t len, const struct module * mod)
{
    intermediate_file_name(filename, len, mod, "xmod");
}


void
xsmod_file_name(char * filename, size_t len, const struct module * mod)
{
    intermediate_file_name(filename, len, mod, "xsmod");
}


/**
 * export public identifiers
 */
int
export_xmod(struct module * mod)
{
    char filename[FILE_NAME_LEN] = {0};
    xmod_file_name(filename, sizeof(filename), mod);
    if (mod != NULL) {
        return output_module_file(mod, filename);
    } else {
        return FALSE;
    }
}


/**
 * export public/private identifiers
 */
int
export_xsmod(struct module * mod)
{
    char filename[FILE_NAME_LEN] = {0};
    xsmod_file_name(filename, sizeof(filename), mod);
    if (mod != NULL) {
        return output_module_file(mod, filename);
    } else {
        return FALSE;
    }
}


/**
 * export public identifiers to module-manager.
 */
int
export_module(SYMBOL sym, ID ids, expv use_decls)
{
    int ret = FALSE;
    int has_submodule = FALSE;
    ID id;
    struct module * mod = generate_current_module(sym, NULL, ids, use_decls, FALSE);
    if (mod) {
        ret = export_xmod(mod);
    } else {
        return FALSE;
    }
    if (ret == FALSE) {
        return FALSE;
    }

    /* NOTE:
     *  If the module doesn't have the module function/subroutine,
     *  the module is closed and has no submodules.
     *  So don't make xsmod.
     */
    FOREACH_ID(id, ids) {
        if (ID_TYPE(id) &&
            TYPE_IS_MODULE(ID_TYPE(id))) {
            has_submodule = TRUE;
        }
    }

    if (!has_submodule) {
        return TRUE;
    }

    mod = generate_current_module(sym, NULL, ids, use_decls, TRUE);
    if (mod) {
        return export_xsmod(mod);
    } else {
        return FALSE;
    }
}


/**
 * export public identifiers to module-manager.
 */
int
export_submodule(SYMBOL submod_name, SYMBOL mod_name, ID ids, expv use_decls)
{
    struct module * mod;
    mod = generate_current_module(mod_name, submod_name, ids, use_decls, TRUE);
    if (mod) {
        return export_xsmod(mod);
    } else {
        return FALSE;
    }
}


static int
import_intermediate_file(const SYMBOL name,
                         const SYMBOL submodule_name,
                         struct module ** pmod,
                         int as_for_submodule)
{
    struct module * mod;
    extern int flag_do_module_cache;
    const char * extension;

    if (!as_for_submodule) {
        extension = "xmod";
    } else {
        extension = "xsmod";
    }

    if ((*pmod = find_module(name, submodule_name, as_for_submodule)) != NULL) {
        return TRUE;
    }

    if (input_intermediate_file(name, submodule_name, &mod, extension)) {
        *pmod = mod;
        if (flag_do_module_cache == TRUE)
            add_module(mod);
        return TRUE;
    }

    error("failed to import module '%s'", SYM_NAME(name));
    exit(1);
}


/**
 * import public identifiers from module-manager.
 */
int
import_module(const SYMBOL name,
              struct module ** pmod)
{
    return import_intermediate_file(name, NULL, pmod, FALSE);
}

/**
 * import public identifiers from module-manager.
 */
int
import_submodule(const SYMBOL module_name,
                 const SYMBOL submodule_name,
                 struct module ** pmod)
{
    return import_intermediate_file(module_name, submodule_name, pmod, TRUE);
}

void include_module_file(FILE *fp, SYMBOL mod_name)
{
    struct module *mod;
    FILE *mod_fp;
    int ch;

    mod = find_module(mod_name, NULL, FALSE);
    if(mod == NULL || mod->filepath == NULL) return;
    if((mod_fp = fopen(mod->filepath,"r")) == NULL){
        fatal("cannon open xmod file '%s'",mod->filepath);
    }
    while((ch = getc(mod_fp)) != EOF)
        putc(ch,fp);
    return;
}
