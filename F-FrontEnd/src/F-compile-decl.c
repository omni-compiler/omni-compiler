/**
 * file F-compile-decl.c
 */

#include "F-front.h"
#include "module-manager.h"
#include "F-second-pass.h"
#include <math.h>

#define ROUND(a,b)    (b * ( (a+b-1)/b))

static SYMBOL blank_common_symbol;
int order_sequence = 0;

static void     declare_dummy_args _ANSI_ARGS_((expr l,
                                                enum name_class class));
static int      markAsSave _ANSI_ARGS_((ID id));


/* for module and use statement */
extern char line_buffer[];
extern SYMBOL current_module_name;

/*
 * FIXME:
 *	SUPER BOGUS FLAG ALERT !
 */
int is_in_kind_compilation_flag_for_declare_ident = FALSE;
int is_in_struct_member_initializer_compilation_flag_for_declare_ident = FALSE;

ID
declare_function_result_id(SYMBOL s, TYPE_DESC tp) {
    ID retId = find_ident(s);

    if (retId != NULL) {
        return retId;
    }

    retId = declare_ident(s, CL_VAR);
    if (retId == NULL) {
        return NULL; /* error */
    }
    declare_id_type(retId, tp);
    return declare_variable(retId);
}

/*
 * Check id in the parent program unit conflicts with a function/subroutine in
 * the contain block
 *
 * MODULE m
 *   public          :: f  ! not conflict
 *   integer         :: g  ! conflict
 *   integer, public :: h  ! NOT CONFLICT
 *   external        :: i  ! conflict
 *   save            :: j  ! conflict? (SAVE cannot be set to function/subroutine,
 *                         !            should be checked before)
 *
 *   interface
 *     function k() ! conflict
 *     [...]
 *   end inteface
 *
 *   _ = l()        ! not conflict
 *   call m()       ! not conflict
 *
 *  [...]
 *
 *   interface
 *     module function n() ! conflict
 *     [...]
 *   end inteface
 * contains
 *   module function n() ! not confilict (ignore type mismatch)
 *     [...]
 *
 *
 *   call other_function(o) ! not conflict (as a high order function)
 *
 */
static int
conflict_parent_vs_sub_program_unit(ID parent_id)
{
    if (parent_id == NULL)
        return FALSE;

    if (debug_flag) {
        fprintf(debug_fp, "checking conflict\n");
    }

    if (ID_CLASS(parent_id) != CL_UNKNOWN) {
        if (debug_flag) fprintf(debug_fp, "parent_id has explicit class\n");

        if (ID_CLASS(parent_id) == CL_PROC && ID_TYPE(parent_id) != NULL) {
            if (IS_VOID(FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(parent_id))) ||
                FUNCTION_TYPE_HAS_IMPLICIT_RETURN_TYPE(ID_TYPE(parent_id)) ||
                FUNCTION_TYPE_HAS_UNKNOWN_RETURN_TYPE(ID_TYPE(parent_id))) {
                if (debug_flag) fprintf(debug_fp, "parent_id may be used as a function/subroutine\n");
                /* continue checking */
            } else if (TYPE_IS_MODULE(ID_TYPE(parent_id)) &&
                       !FUNCTION_TYPE_IS_DEFINED(ID_TYPE(parent_id))) {
                if (debug_flag) fprintf(debug_fp, "parent_id seems a module function/subroutine\n");
                return FALSE;
            } else if (PROC_CLASS(parent_id) == P_UNDEFINEDPROC &&
                       IS_PROCEDURE_TYPE(ID_TYPE(parent_id))) {
                if (debug_flag) fprintf(debug_fp, "parent_id seems a undefined function/subroutine\n");
                return FALSE;

            } else {
                if (debug_flag) fprintf(debug_fp, "parent_id is an explicit function\n");
                return TRUE;
            }
        } else {
            if (ID_TYPE(parent_id) == NULL || TYPE_IS_IMPLICIT(ID_TYPE(parent_id))) {
                if (debug_flag) fprintf(debug_fp, "parent_id is used as an implicit declared variable\n");
                /* continue checking */
            } else {
                if (debug_flag) fprintf(debug_fp, "parent_id be used as a non-function/subroutine\n");
                return TRUE;
            }
        }
    }

    if ((ID_TYPE(parent_id) && !TYPE_IS_IMPLICIT(ID_TYPE(parent_id))) &&
        !TYPE_HAS_ACCESSIBILITY_FLAGS(parent_id) && TYPE_HAS_NON_ACCESSIBILITY_FLAGS(parent_id)) {
        if (debug_flag) fprintf(debug_fp, "parent_id has non-accessibility attribute\n");
        return TRUE;
    }

    if (ID_TYPE(parent_id) != NULL) {
        assert (ID_TYPE(parent_id) != NULL);

        if (!TYPE_IS_IMPLICIT(ID_TYPE(parent_id)) &&
            !TYPE_HAS_ACCESSIBILITY_FLAGS(parent_id) &&
            TYPE_HAS_NON_ACCESSIBILITY_FLAGS(parent_id)) {
            if (debug_flag) fprintf(debug_fp, "parent_id has non-accessibility attribute\n");
            return TRUE;
        }

        if (ID_TYPE(parent_id) && TYPE_IS_SAVE(ID_TYPE(parent_id))) {
            if (debug_flag) fprintf(debug_fp, "parent_id has non-accessibility attribute\n");
            error("SAVE attirubte conflicts to the procedure");
            return TRUE;
        }

        if (IS_SUBR(ID_TYPE(parent_id))) {
            if (debug_flag) fprintf(debug_fp, "parent_id seems a subroutine\n");
            if (ID_TYPE(parent_id) &&
                !TYPE_IS_IMPLICIT(ID_TYPE(parent_id)) && TYPE_HAS_NON_ACCESSIBILITY_FLAGS(parent_id)) {
                if (debug_flag) fprintf(debug_fp, "parent_id seems an external subroutine\n");
                return TRUE;
            }

        } else if (IS_FUNCTION_TYPE(ID_TYPE(parent_id))) {
            TYPE_DESC return_type = FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(parent_id));
            if (debug_flag) fprintf(debug_fp, "parent_id seems a function\n");
            if (return_type != NULL &&
                !TYPE_IS_IMPLICIT(return_type) &&
                TYPE_BASIC_TYPE(return_type) != TYPE_UNKNOWN) {
                if (debug_flag) fprintf(debug_fp, "ID_TYPE(parent_id) has explicit type\n");
                if (!TYPE_HAS_ACCESSIBILITY_FLAGS(parent_id) &&
                    !TYPE_HAS_ACCESSIBILITY_FLAGS(ID_TYPE(parent_id))) {
                    if (debug_flag) fprintf(debug_fp, "parent_id and ID_TYPE(parent_id) don't have accessibility attributes\n");
                    return TRUE;
                }
            }

        } else if (ID_IS_DECLARED(parent_id)) {
            if (debug_flag) fprintf(debug_fp, "parent_id seems like to be not declared\n");

        } else {
            if (debug_flag) fprintf(debug_fp, "parent_id seems like a variable\n");
            if (!TYPE_IS_IMPLICIT(ID_TYPE(parent_id)) &&
                TYPE_BASIC_TYPE(ID_TYPE(parent_id)) != TYPE_UNKNOWN) {
                if (debug_flag) fprintf(debug_fp, "ID_TYPE(parent_id) has explicit type\n");
                if (!TYPE_HAS_ACCESSIBILITY_FLAGS(parent_id) &&
                    !TYPE_HAS_ACCESSIBILITY_FLAGS(ID_TYPE(parent_id))) {
                    if (debug_flag) fprintf(debug_fp, "parent_id and ID_TYPE(parent_id) don't have accessibility attributes\n");
                    return TRUE;
                }
            }
        }
    }

    if (debug_flag) fprintf(debug_fp, "NOT conflict\n");

    return FALSE;
}

static void
link_parent_defined_by(ID id)
{
    SYMBOL sym = ID_SYM(id);
    ID parent;

    if (sym == NULL)
        return;
    /* search parents local symbols,
       and if my name is found, then link it */
    if (unit_ctl_level > 0) {
        int is_contain_proc = FALSE;
        if (PARENT_STATE == INCONT) {
            /* If the procedure exists in CONTAINS block,
               the procedure is a module procedure.       */
            is_contain_proc = TRUE;
        }

        parent = find_ident_block_parent(sym);
        if (parent == NULL) {
            parent = find_ident_head(sym, PARENT_LOCAL_SYMBOLS);
        }
        if (parent == NULL)
            return;

        if (ID_CLASS(parent) == CL_PROC &&
            ID_TYPE(parent) != NULL &&
            TYPE_IS_MODULE(ID_TYPE(parent)) &&
            (TYPE_IS_MODULE(id) || TYPE_IS_MODULE(ID_TYPE(id)))) {
            /*
             * both are module function/subroutine, check it later
             */
            return;
        }

        if (is_contain_proc) {
            if (conflict_parent_vs_sub_program_unit(parent)) {

                if (ID_CLASS(parent) == CL_PROC &&
                    ID_TYPE(parent) != NULL &&
                    IS_GENERIC_PROCEDURE_TYPE(ID_TYPE(parent))) {

                    ID next = ID_NEXT(parent);
                    id_multilize(parent);
                    ID_NEXT(parent) = MULTI_ID_LIST(parent);
                    ID_NEXT(ID_NEXT(parent)) = next;
                    return;
                } else {
                    error("%s has an explicit interface before", ID_NAME(parent));
                    return;
                }
            }
        }

        if (ID_CLASS(parent) == CL_UNKNOWN ||
            (ID_CLASS(parent) == CL_VAR && ID_IS_DECLARED(id) == FALSE)) { /* my used as a argument */
            ID_CLASS(parent) = CL_PROC;
            ID_STORAGE(parent) = STG_EXT;
            PROC_CLASS(parent) = P_EXTERNAL;
        } else if (ID_IS_DECLARED(id) == FALSE) {
            ID_CLASS(parent) = CL_PROC;
        }

        /* Conditions below is written to make test programs to pass. */
        /* And it is not derived from the specification. So condition */
        /* may be not enough. */
        if (ID_CLASS(parent) == CL_PROC &&
            (PROC_CLASS(parent) == P_UNDEFINEDPROC ||
             PROC_CLASS(parent) == P_EXTERNAL ||
             IS_TYPE_PUBLICORPRIVATE(parent) ||
             (ID_TYPE(parent) != NULL &&
              (IS_TYPE_PUBLICORPRIVATE(ID_TYPE(parent)))))) {
            ID_DEFINED_BY(parent) = CURRENT_PROCEDURE;
        } else {
            error("%s is defined as a variable before", ID_NAME(parent));
        }
    }
}

/*
 * define main program or block data, subroutine, functions
 */
void
declare_procedure(enum name_class class,
                  expr name, TYPE_DESC type, expr args,
                  expr prefix_spec, expr result_opt, expr bind_opt)
{
    SYMBOL s = NULL;
    ID id = NULL, ip, last = NULL;
    EXT_ID ep;
    int recursive = FALSE;
    int pure = FALSE;
    int elemental = FALSE;
    int module = FALSE;
    int impure = FALSE;
    int is_separate_definition = FALSE;
    list lp;

    if (name) {
        if(EXPR_CODE(name) != IDENT) abort();
        s = EXPR_SYM(name);
    }

    if (IS_PROCEDURE_TYPE(type) && TYPE_IS_MODULE(type)) {
        is_separate_definition = TRUE;
    }

    if (type != NULL &&
        ((type_is_nopolymorphic_abstract(type)) ||
         (FUNCTION_TYPE_RETURN_TYPE(type) != NULL &&
          type_is_nopolymorphic_abstract(FUNCTION_TYPE_RETURN_TYPE(type))))) {
        error("%s returns ABSTRACT TYPE", s?SYM_NAME(s):"function");
    }

    if (class != CL_ENTRY){
        CURRENT_PROC_CLASS = class;
        if (name) {
            CURRENT_PROC_NAME = s;
            ep = find_ext_id_parent(CURRENT_PROC_NAME);
            if (ep != NULL && EXT_IS_DEFINED(ep) &&
                 EXT_PROC_IS_MODULE_PROCEDURE(ep) == FALSE &&
                (unit_ctl_level == 0 || PARENT_STATE != ININTR) &&
                !is_separate_definition) {
                /* error("same name is already defined in parent"); */
                /* return; */
                warning("A host-associated procedure is overridden.");
            }
            else if (unit_ctl_level > 0 && PARENT_STATE != ININTR &&
                     !is_separate_definition) {
                ep = find_ext_id(CURRENT_PROC_NAME);
                if (ep != NULL && EXT_IS_DEFINED(ep) &&
                    EXT_PROC_IS_MODULE_PROCEDURE(ep) == FALSE) {
                    error("same name is already defined");
                    return;
                }
            }
        }
    }

    FOR_ITEMS_IN_LIST(lp, prefix_spec) {
        switch (EXPR_CODE(LIST_ITEM(lp))) {
        case F95_RECURSIVE_SPEC:
            if (class != CL_PROC) {
                error("invalid recursive prefix");
                return;
            }
            recursive = TRUE;
            break;

        case F95_PURE_SPEC:
            if (class != CL_PROC) {
                error("invalid pure prefix");
                return;
            }
            pure = TRUE;
            break;

        case F95_ELEMENTAL_SPEC:
            if (class != CL_PROC) {
                error("invalid elemental prefix");
                return;
            }
            elemental = TRUE;
            break;

        case F08_MODULE_SPEC:
            if (class != CL_PROC) {
                error("invalid module prefix");
                return;
            }
            module = TRUE;
            break;

        case F08_IMPURE_SPEC:
            if (class != CL_PROC) {
                error("invalid pure prefix");
                return;
            }
            impure = TRUE;
            break;

        default:
            error("unknown prefix");
        }
    }

    switch(class){

    case CL_MAIN:
      if (debug_flag)
          fprintf(diag_file,"  MAIN %s:\n",(name ? SYM_NAME(s): ""));

      // Delete line because of [Xmp-dev:1896]
      // CURRENT_EXT_ID = declare_external_id(find_symbol(
      //   		   name ? SYM_NAME(s): "main"), STG_EXT, TRUE);
      CURRENT_EXT_ID = declare_external_id(s, STG_EXT, TRUE);
      EXT_PROC_CLASS(CURRENT_EXT_ID) = EP_PROGRAM;
      EXT_PROC_TYPE(CURRENT_EXT_ID) = program_type();
      if (name) {
          /* set line_no */
          EXT_LINE(CURRENT_EXT_ID) = EXPR_LINE(name);
          id = declare_ident(s,CL_MAIN);
      }
      break;

    case CL_BLOCK:
        if (debug_flag)
            fprintf(diag_file,"  BLOCK DATA %s:\n",name ? SYM_NAME(s): "");
        CURRENT_EXT_ID = declare_external_id(find_symbol(
            name ? SYM_NAME(s): "no__name__blkdata__"), STG_COMMON, TRUE);
        if (CURRENT_EXT_ID) {
            EXT_IS_BLANK_NAME(CURRENT_EXT_ID) = (name ? FALSE : TRUE);
            if(name) {
                id = declare_ident(s,CL_BLOCK);
                ID_LINE(id) = EXPR_LINE(name); /* set line_no */
                EXT_LINE(CURRENT_EXT_ID) = EXPR_LINE(name); /* set line_no */
            }
        }
        break;

    case CL_PROC: { /* subroutine or functions */
        if (debug_flag)
            fprintf(diag_file,"   %s:\n",SYM_NAME(s));

        if (unit_ctl_level > 0 && PARENT_STATE == ININTR) {
            ID pid = find_ident_outer_scope(s);
            if (pid != NULL && ID_STORAGE(pid) == STG_ARG) {
                /*
                 * The s is declared in an interface statement in
                 * a dummy argument list.
                 */
                PROC_IS_DUMMY_ARG(pid) = TRUE;
            }
        }

        /* make local entry */
        id = declare_ident(s, CL_PROC);
        if (ID_ORDER(id) == 0)
            ID_ORDER(id) = order_sequence++;

        if (result_opt != NULL) {
            PROC_RESULTVAR(id) = result_opt;
        }

        if (type != NULL) {
            if (!IS_PROCEDURE_TYPE(type) || IS_PROCEDURE_POINTER(type)) {
                type = function_type(type);
            }
            declare_id_type(id, type);
            TYPE_SET_FOR_FUNC_SELF(ID_TYPE(id));
        } else {
            declare_id_type(id, new_type_desc());
        }
        if (result_opt != NULL) {
            FUNCTION_TYPE_RESULT(ID_TYPE(id)) = EXPR_SYM(result_opt);
        }

        ID_LINE(id) = EXPR_LINE(name); /* set line_no */

        PROC_ARGS(id) = args;
        FOR_ITEMS_IN_LIST(lp, args) {
            ip = new_ident_desc(EXPR_SYM(LIST_ITEM(lp)));
            ID_STORAGE(ip) = STG_ARG;
            ID_LINK_ADD(ip, FUNCTION_TYPE_ARGS(ID_TYPE(id)), last);
        }
        FUNCTION_TYPE_HAS_EXPLICIT_ARGS(ID_TYPE(id)) = TRUE;

        if (recursive == TRUE) {
            PROC_IS_RECURSIVE(id) = recursive;
            if (result_opt != NULL) {
                PROC_CLASS(id) = P_DEFINEDPROC;
            } else {
                /* RESULT is not specified,
                 * so this function cannot be called recursively */
                PROC_CLASS(id) = P_THISPROC;
            }
            TYPE_SET_RECURSIVE(id);
            if (ID_TYPE(id) != NULL) {
                TYPE_SET_RECURSIVE(ID_TYPE(id));
            }
        } else {
            PROC_CLASS(id) = P_THISPROC;
        }

        if (pure && impure) {
            error("PURE and IMPURE are specified");
        }

        if (pure == TRUE) {
            PROC_IS_PURE(id) = pure; /* MAY NOT BE REQUIRED */
            TYPE_SET_PURE(id); /* MAY NOT BE REQUIRED */
            if (ID_TYPE(id) != NULL) {
                TYPE_SET_PURE(ID_TYPE(id));
            }
        }
        if (elemental == TRUE) {
            PROC_IS_ELEMENTAL(id) = elemental;
            TYPE_SET_ELEMENTAL(id);
            if (ID_TYPE(id) != NULL) {
                TYPE_SET_ELEMENTAL(ID_TYPE(id));
            }
        }
        if (impure == TRUE) {
            PROC_IS_PURE(id) = !impure;
            TYPE_SET_IMPURE(id);
            if (ID_TYPE(id) != NULL) {
                TYPE_SET_IMPURE(ID_TYPE(id));
            }
        }
        if (module == TRUE) {
            PROC_IS_MODULE(id) = module;
            TYPE_SET_MODULE(id);
            if (ID_TYPE(id) != NULL) {
                TYPE_SET_MODULE(ID_TYPE(id));
            }
        }

        if (bind_opt) {
            PROC_HAS_BIND(id) = TRUE;
            TYPE_SET_BIND(ID_TYPE(id));
            expr bind_name = EXPR_ARG1(bind_opt);

            if(bind_name){
                PROC_BIND(id) = bind_name;
                if(ID_TYPE(id) != NULL) {
                    TYPE_BIND_NAME(ID_TYPE(id)) = bind_name;
                }
            }
        }

        if (unit_ctl_level == 0 || PARENT_STATE != ININTR) {
            /* This function/subroutine has the definition */
            FUNCTION_TYPE_SET_DEFINED(type);
        }

        ID_STORAGE(id) = STG_EXT;
        declare_dummy_args(args, CL_PROC);
        CURRENT_PROCEDURE = id;

        /* make link before declare_current_procedure_ext_id() */
        link_parent_defined_by(id);
        (void)declare_current_procedure_ext_id();

	EXT_LINE(CURRENT_EXT_ID) = EXPR_LINE(name);

        break;
    }

    case CL_ENTRY: {
        EXT_ID ext_id = NULL;
        expv emitV = NULL;
        expv symV = NULL;
        expv resultV = NULL;
        TYPE_DESC tp = NULL;
        list lp;

        if (debug_flag)
            fprintf(diag_file,"   entry %s:\n",SYM_NAME(s));

        id = declare_ident(s, CL_ENTRY);
        if (IS_SUBR(ID_TYPE(CURRENT_PROCEDURE))) {
            /*
             * If the current procedure is a subroutine,
             * the entry is also a subroutine.
             */
            tp = subroutine_type();
        } else {
            type = ID_TYPE(CURRENT_PROCEDURE);
            if (type == NULL) {
                fatal("%s: return type is NOT determined.",
                      __func__);
                return;
            }
            /*
             * Copy the type of the current procedure.
             */
            tp = new_type_desc();
            *tp = *type;
            TYPE_ATTR_FLAGS(tp) = 0;
            FUNCTION_TYPE_RETURN_TYPE(tp) = new_type_desc();
            *FUNCTION_TYPE_RETURN_TYPE(tp) = *FUNCTION_TYPE_RETURN_TYPE(type);
        }
        declare_id_type(id, tp);

        ID_LINE(id) = EXPR_LINE(name); /* set line_no */
        PROC_CLASS(id) = P_THISPROC;
        PROC_ARGS(id) = args;
        ID_STORAGE(id) = STG_EXT;

        PROC_ARGS(id) = args;
        FOR_ITEMS_IN_LIST(lp, args) {
            ip = new_ident_desc(EXPR_SYM(LIST_ITEM(lp)));
            ID_LINK_ADD(ip, FUNCTION_TYPE_ARGS(ID_TYPE(id)), last);
        }
        FUNCTION_TYPE_HAS_EXPLICIT_ARGS(ID_TYPE(id)) = TRUE;

        declare_dummy_args(args, CL_ENTRY);
        /* If an entry statement appears in execute part,
           then each arguments of it should have type. */
        if(CURRENT_STATE == INEXEC) {
            FOR_ITEMS_IN_LIST(lp,args) {
                expr x = LIST_ITEM(lp);
                SYMBOL s = EXPR_SYM(x);
                ID arg = find_ident_local(s);
                if(ID_TYPE(arg) == NULL) {
                    implicit_declaration(arg);
                }
                if(ID_TYPE(arg) == NULL) {
                    error("'%s' has no implicit type.", SYM_NAME(s));
                    abort();
                }
            }
        }

        if (result_opt != NULL) {
            if (IS_SUBR(ID_TYPE(id))) {
                error("subroutine with RESULT.");
                return;
            }

            SYMBOL resS = EXPR_SYM(result_opt);
            ID resId = declare_function_result_id(resS,
                                                  FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(id)));
            if (resId == NULL) {
                fatal("%s: can't declare result identifier '%s'.",
                      __func__, SYM_NAME(resS));
                return;
            }
            //TODO: resId also required implicit declaration like args
	    // to be resolved
            resultV = expv_sym_term(F_VAR, ID_TYPE(resId), ID_SYM(resId));

            PROC_CLASS(id) = P_DEFINEDPROC;
        }

        ext_id = define_external_function_id(id);
        EXT_PROC_CLASS(ext_id) = EP_ENTRY;
        EXT_PROC_RESULTVAR(ext_id) = resultV;
        symV = expv_sym_term(F_FUNC, ID_TYPE(id), ID_SYM(id));
        EXPV_ENTRY_EXT_ID(symV) = ext_id;
        PROC_EXT_ID(id) = ext_id;
        EXT_PROC_ID_LIST(ext_id) = id;
        unset_save_attr_in_dummy_args(ext_id);
        emitV = expv_cons(F_ENTRY_STATEMENT,
                          ID_TYPE(id), symV, EXT_PROC_ARGS(ext_id));

        EXPV_LINE(emitV) = EXPR_LINE(name);

        output_statement(emitV);

        break;
    }

    case CL_SUBMODULE: /* submodule */ /* fall through */
    case CL_MODULE: /* modules */ {
        extern int mcLn_no;
        current_module_name = s;
        /* should print in module compile mode.  */
        if (mcLn_no == -1)
        if (debug_flag)
            fprintf(diag_file,"   %s %s:\n",
                    class == CL_MODULE?"module":"submodule",
                    SYM_NAME(current_module_name));
        id = declare_ident(s,class);
        declare_id_type(id,type);
        ID_LINE(id) = EXPR_LINE(name); /* set line_no */
        ID_STORAGE(id) = STG_EXT;
        CURRENT_PROCEDURE = id;
        (void)declare_current_procedure_ext_id();
        break;
    }

    default:
        fatal("%s: unknown class", __func__);
    }


    if(unit_ctl_level > 0 && PARENT_STATE == ININTR) {
        EXT_ID interface;
        int arg_len;
        list lp;

        if (id != NULL && ID_TYPE(id) != NULL) {
            FUNCTION_TYPE_SET_INTERFACE(ID_TYPE(id));
        }

        arg_len = 0;
        FOR_ITEMS_IN_LIST(lp, args) {
            arg_len++;
        }

        if (class != CL_PROC) {
            error("unexpected statement in interface block");
            abort();
        }

        interface = CURRENT_INTERFACE;

        assert(interface != NULL);

        switch(EXT_PROC_INTERFACE_CLASS(interface)) {
            case INTF_ASSIGNMENT: {
                if(IS_SUBR(type) == FALSE) {
                    error("unexpected FUNCTION in assignment(=) INTERFACE");
                    return;
                }
                if(arg_len != 2) {
                    error("wrong number of argument.");
                    return;
                }
            } break;

            case INTF_OPERATOR: {
                if(IS_SUBR(type)) {
                    error("unexpected SUBROUTINE in operator INTERFACE");
                    return;
                }
            } break;

            case INTF_USEROP: {
                if(IS_SUBR(type)) {
                    error("unexpected SUBROUTINE in operator INTERFACE");
                    return;
                }
                if(arg_len != 1 && arg_len != 2) {
                    error("wrong number of argument.");
                    return;
                }
            } break;

            case INTF_GENERICS: {
                if(IS_SUBR(type)) {
                    EXT_PROC_INTERFACE_CLASS(interface) = INTF_GENERIC_SUBR;
                } else {
                    EXT_PROC_INTERFACE_CLASS(interface) = INTF_GENERIC_FUNC;
                }
            } break;

            case INTF_GENERIC_FUNC: {
                if(IS_SUBR(type)) {
                    error("unexpected SUBROUTINE in FUNCTION generics INTERFACE");
                    return;
                }
            } break;

            case INTF_GENERIC_SUBR: {
                if(IS_SUBR(type) == FALSE) {
                    error("unexpected FUNCTION in SUBROUTINE generics INTERFACE");
                    return;
                }
            } break;

            case INTF_DECL:
            default:
                break;
        }
    }
}


EXT_ID
declare_current_procedure_ext_id()
{
    EXT_ID ep;
    assert(CURRENT_PROCEDURE);

    ep = CURRENT_EXT_ID = define_external_function_id(CURRENT_PROCEDURE);
    assert(ep);

    return ep;
}

static void
declare_dummy_args(expr l, enum name_class class)
{
    list lp;
    expr x;
    SYMBOL s;
    ID id;

    FOR_ITEMS_IN_LIST(lp,l) {
        x = LIST_ITEM(lp);
        if (EXPR_CODE(x) != IDENT) {
            fatal("%s: not IDENT", __func__);
        }
        s = EXPR_SYM(x);
        id = declare_ident(s, CL_UNKNOWN);
        ID_COULD_BE_IMPLICITLY_TYPED(id) = TRUE;
        if (ID_STORAGE(id) == STG_UNKNOWN) {
            ID_STORAGE(id) = STG_ARG;
        } else if (ID_STORAGE(id) != STG_ARG) {
            if (ID_STORAGE(id) == STG_AUTO &&
                IS_ARRAY_TYPE(ID_TYPE(id)) &&
                (!TYPE_IS_POINTER(ID_TYPE(id))) &&
                class == CL_ENTRY &&
                is_array_size_adjustable(ID_TYPE(id))) {
                /*
                 * Local adjustable array is about to be used as a
                 * dummy arg for ENTRY statement. Currently unsupported.
                 */
                error_at_node(l,
                              "A local adjustable array '%s' is about to "
                              "be used as a dummy argument for an "
                              "ENTRY statement, not supported yet.",
                              ID_NAME(id));
            } else if (class != CL_ENTRY) {
                error_at_node(l,
                              "illegal dummy argument '%s'", SYM_NAME(s));
            }
        }
    }
}


TYPE_DESC
copy_type_shallow(TYPE_DESC tp)
{
    TYPE_DESC ret = new_type_desc();
    *ret = *tp;
    return ret;
}


static void
copy_parent_type(ID id)
{
    ID parent_id;

    if (TYPE_IS_OVERRIDDEN(id)) { /* second time */
        return;
    }
    if (ID_TYPE(id) != NULL) {
        return;
    }
    parent_id = find_ident_outer_scope(ID_SYM(id));
    if (parent_id == NULL) {
        return;
    }
    ID_DEFINED_BY(id) = parent_id;
    if (ID_CLASS(parent_id) == CL_PROC) {
        PROC_CLASS(id) = PROC_CLASS(parent_id);
    }
    declare_id_type(id, ID_TYPE(parent_id));
    id->use_assoc = parent_id->use_assoc;
    TYPE_SET_OVERRIDDEN(id);
}

/*
 * identifier management
 */

void
implicit_declaration(ID id)
{
    TYPE_DESC tp;
    char c;

    /* if (ID_CLASS(id) == CL_MAIN || */
    /*     (ID_CLASS(id) == CL_PROC && PROC_CLASS(id) == P_INTRINSIC)) { */
    /*     return; */
    /* } */

    if (ID_CLASS(id) == CL_MAIN) return;

    if (CURRENT_STATE == INEXEC && !ID_COULD_BE_IMPLICITLY_TYPED(id)) {
        /* use parent type as is */
        copy_parent_type(id);
    }
    tp = ID_TYPE(id);
    if (IS_SUBR(tp)) {
        /* implicit declaration does not work */
        return;
    }

    if (tp != NULL && TYPE_IS_PROCEDURE(tp) && TYPE_REF(tp)) {
        /* id is a procedure pointer, its type isn't determeind implicitly */
        return;
    }

    if (IS_FUNCTION_TYPE(tp)) {
        tp = FUNCTION_TYPE_RETURN_TYPE(tp);
    }

    if (tp != NULL && TYPE_IS_INTRINSIC(tp)) {
        /* don't set a type to intrinsic procedures */
        return;
    }

    if (tp == NULL || TYPE_IS_NOT_FIXED(tp) ||
        (IS_ARRAY_TYPE(tp) && array_element_type(tp) == NULL)) {
        c = ID_NAME(id)[0];
        if (isupper((int)c)) {
            c = tolower((int)c);
        }
        tp = IMPLICIT_TYPES[c-'a'];
        if (tp == NULL) {
            if (ID_CLASS(id) == CL_VAR) {
/* FEAST delete start */
                /* error("attempt to use undefined type variable, %s", ID_NAME(id)); */
                sp_link_id(id, SP_ERR_UNDEF_TYPE_VAR, current_line);
/* FEAST delete  end  */
                return;
            }
            else return;
        }
        /*
         * OK, here we make a new TYPE_DESC.
         */
        assert(tp != NULL);
        tp = wrap_type(tp);
        assert(tp != NULL);

        if (current_proc_state == P_SAVE) {
            if (!IS_PROCEDURE_TYPE(ID_TYPE(id))) {
                TYPE_SET_SAVE(tp);
            }
        }

        declare_id_type(id, tp);
    }
}

/* variable declaration */
ID
declare_variable(ID id)
{
    expv v;

    if (ID_CLASS(id) == CL_MAIN) return id; /* don't care */

    if (ID_CLASS(id) == CL_NAMELIST) {
        return id;
    }

    if (ID_CLASS(id) == CL_UNKNOWN) {
        ID_CLASS(id) = CL_VAR;
    } else if (ID_CLASS(id) != CL_VAR &&
               ID_CLASS(id) != CL_PROC &&
               ID_CLASS(id) != CL_ENTRY &&
               ID_CLASS(id) != CL_PARAM &&
               ID_CLASS(id) != CL_ELEMENT &&
               ID_CLASS(id) != CL_TYPE_PARAM) {
        error("used as variable, %s", ID_NAME(id));
        return NULL;
    }

    if(ID_IS_DECLARED(id)){
        if(ID_ADDR(id) == NULL) return NULL;    /* error recovery */
        return id;
    } else ID_IS_DECLARED(id) = TRUE;

    implicit_declaration(id);
    if(ID_TYPE(id) == NULL) return NULL; /* error */

    if (ID_STORAGE(id) == STG_UNKNOWN) {
        if (TYPE_IS_SAVE(id) || (ID_TYPE(id) && TYPE_IS_SAVE(ID_TYPE(id))) ||
            VAR_INIT_LIST(id) ||
            CURRENT_PROC_CLASS == CL_MODULE) {
            ID_STORAGE(id) = STG_SAVE;
        } else {
            int isSubprogram = CURRENT_EXT_ID &&
                (EXT_TAG(CURRENT_EXT_ID) == STG_EXT &&
                EXT_PROC_IS_PROGRAM(CURRENT_EXT_ID) == FALSE);

            if (IS_ARRAY_TYPE(ID_TYPE(id)) &&
                (!TYPE_IS_POINTER(ID_TYPE(id))) &&
                (!TYPE_IS_ALLOCATABLE(ID_TYPE(id))) &&
                isSubprogram == FALSE &&
                is_array_size_adjustable(ID_TYPE(id)) &&
                !is_array_implicit_shape(ID_TYPE(id)) &&
                !XMP_flag) { // For XMP, local adjustable array seems to be supported, because of LOCAL_ALIAS.
                error("'%s' looks like a local adjustable array, "
                      "not supported yet.",
                      ID_NAME(id));
                /* not reached. */
                return NULL;
            }
            else if (IS_ARRAY_TYPE(ID_TYPE(id)) &&
                     (TYPE_IS_POINTER(ID_TYPE(id)) ||
                      TYPE_IS_ALLOCATABLE(ID_TYPE(id))) &&
                     !is_array_shape_assumed(ID_TYPE(id))) {
                error("'%s' has the allocatable or pointer attribute, "
                      "but is not a dererred-shape array.",
                      ID_NAME(id));
                /* not reached. */
                return NULL;
            }
            else {
                ID_STORAGE(id) = STG_AUTO;
            }
        }
    }

    switch(ID_STORAGE(id)){
    case STG_SAVE:
    case STG_ARG: /* dummy argument */
    case STG_AUTO:
    case STG_EXT:
    case STG_EQUIV:
    case STG_COMEQ:
    case STG_COMMON:
    case STG_TYPE_PARAM:
    case STG_INDEX:
        v = expv_sym_term(F_VAR, ID_TYPE(id), ID_SYM(id));
        if (PROC_CLASS(id) == P_THISPROC) {
            EXPV_TYPE(v) = FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(id));
        }
        ID_ADDR(id) = v;
        break;
    default:
        fatal("declare_variable: unknown class");
    }
    return id;
}

ID
declare_function(ID id)
{
    if (ID_CLASS(id) == CL_MULTI) {
        id = multi_find_class(id , CL_PROC);
        if (id == NULL) {
            fatal("multi class id bug");
            return id;
        }
    }

    if (ID_CLASS(id) == CL_UNKNOWN) {
        /* if name class is unknown, define it as CL_PROC */
        ID_CLASS(id) = CL_PROC;
        if (ID_STORAGE(id) == STG_UNKNOWN) {
            if (is_intrinsic_function(id)) {
                PROC_CLASS(id) = P_INTRINSIC;
                TYPE_SET_INTRINSIC(id);
                ID_STORAGE(id) = STG_NONE;
                ID_IS_DECLARED(id) = TRUE;
                return id;
            } else {
                /* it may be undefined function. */
                TYPE_DESC tp;
                tp = ID_TYPE(id);
                if (tp == NULL ||
                   (TYPE_IS_IMPLICIT(tp) && !IS_TYPE_PUBLICORPRIVATE(tp))) {
                    if (tp == NULL) {
                        tp = function_type(new_type_desc());
                        TYPE_SET_NOT_FIXED(FUNCTION_TYPE_RETURN_TYPE(tp));
                        TYPE_BASIC_TYPE(FUNCTION_TYPE_RETURN_TYPE(tp)) = TYPE_GNUMERIC_ALL;
                        ID_TYPE(id) = tp;
                    } else if (!IS_PROCEDURE_TYPE(tp) || (TYPE_IS_PROCEDURE(tp) && TYPE_REF(tp) != NULL)) {
                        ID_TYPE(id) = function_type(tp);
                        /*
                         * For F_Front, There is no difference between an
                         * explicit SAVE attribute and an implicit SAVE
                         * attribute
                         *
                         *   REAL, SAVE func
                         *
                         * and
                         *
                         *   REAL func
                         *   SAVE ! set a SAVE attribute to func
                         *
                         * But if `func` is a funciton/subroutine, the compiler
                         * should raise error against an explicit SAVE
                         * attribute, and shouldn't set an implicit SAVE
                         * attribute to `func`.  So the following
                         * TYPE_UNSET_SAVE causes false positive behaviour
                         * (accepts a function with an explicit SAVE attribute).
                         */
                        TYPE_UNSET_SAVE(FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(id)));
                    }
                    PROC_CLASS(id) = P_UNDEFINEDPROC;
                } else {
                    if (!TYPE_IS_PROCEDURE(tp) || (TYPE_IS_PROCEDURE(tp) && TYPE_REF(tp) != NULL)) {
                        ID_TYPE(id) = function_type(tp);
                        TYPE_UNSET_SAVE(FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(id)));
                    }

                    if (IS_TYPE_PUBLICORPRIVATE(FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(id)))) {
                        /* function is module procedure */
                        PROC_CLASS(id) = P_DEFINEDPROC;
                    } else {
                        PROC_CLASS(id) = P_EXTERNAL;
                    }

                    if (TYPE_BASIC_TYPE(FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(id))) == TYPE_UNKNOWN) {
                        TYPE_SET_NOT_FIXED(FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(id)));
                    }
                }
            }
        } else if (ID_STORAGE(id) == STG_ARG) {
#if 0
            /* This seems wrong, just used as an argument */
            if (VAR_IS_USED_AS_FUNCTION(id) == FALSE) {
                warning("Dummy procedure not declared EXTERNAL. "
                        "Code may be wrong.");
            }
#endif
            if (ID_TYPE(id) && !IS_PROCEDURE_TYPE(ID_TYPE(id))) {
                ID_TYPE(id) = function_type(ID_TYPE(id));
            }
            PROC_CLASS(id) = P_UNDEFINEDPROC;
        } else if (ID_STORAGE(id) != STG_EXT /* maybe interface */) {
            fatal("%s: bad storage '%s'", __func__, ID_NAME(id));
        }
    } else if (ID_CLASS(id) != CL_PROC &&
               ID_CLASS(id) != CL_ENTRY &&
               !(ID_CLASS(id) == CL_VAR && ID_TYPE(id) != NULL 
               && IS_PROCEDURE_TYPE(ID_TYPE(id)))) 
    {
        if(SYM_TYPE(ID_SYM(id)) == S_INTR) {
            // Not user defined function and intrinsic function used 
            // but not declared yet. So let's declare it.
            ID_CLASS(id) = CL_PROC;
            PROC_CLASS(id) = P_INTRINSIC;
            TYPE_SET_INTRINSIC(id);
            ID_STORAGE(id) = STG_NONE;
            ID_IS_DECLARED(id) = TRUE;
        } else {
            error("identifier '%s' is used as a function", ID_NAME(id));
            return NULL;
        }
    }

    if (ID_STORAGE(id) == STG_UNKNOWN) {
        ID_STORAGE(id) = STG_EXT;
    }

    if (ID_IS_DECLARED(id)) {
        return id;
    }

    if (PROC_CLASS(id) != P_UNDEFINEDPROC) {
        ID_IS_DECLARED(id) = TRUE;
        if (ID_TYPE(id) == NULL) {
            implicit_declaration(id);
        }
    }

    if (ID_ADDR(id) == NULL) {
        /* fix stoarge */
        /* NOTE that function address's type is ignored.
         * don't need keep track of function type in Fortran.
         */
        expv v;
        if (ID_STORAGE(id) != STG_EXT) {
            if (ID_STORAGE(id) != STG_ARG &&
                !(ID_CLASS(id) == CL_VAR && IS_PROCEDURE_TYPE(ID_TYPE(id)))) {
                fatal("%s: unknown storage", __func__);
            }
        }

        if (ID_CLASS(id) != CL_VAR &&
            PROC_CLASS(id) != P_INTRINSIC &&
            PROC_CLASS(id) != P_UNDEFINEDPROC) {
            EXT_ID ep = PROC_EXT_ID(id);
            if(ep == NULL) {
                ep = declare_external_proc_id(ID_SYM(id), ID_TYPE(id), FALSE);
                PROC_EXT_ID(id) = ep;
            }
        }

        v = expv_sym_term(F_FUNC, NULL, ID_SYM(id));
        EXPV_TYPE(v) = ID_TYPE(id);
        if (ID_TYPE(id)) TYPE_UNSET_SAVE(ID_TYPE(id));
        ID_ADDR(id) = v;
    }
    return id;
}


void
declare_statement_function(id,args,body)
     ID id;
     expr args,body;
{
    ID ip;
    list lp;
    expr x;

    if(ID_CLASS(id) != CL_UNKNOWN)
      fatal("declare_statement_function: not CL_UNKNOWN");

    check_NOT_INBLOCK();

    if (CURRENT_PROC_CLASS == CL_MODULE ||
        CURRENT_PROC_CLASS == CL_SUBMODULE) {
        error("misplaced statement function");
        return;
    }

    ID_CLASS(id) = CL_PROC;
    PROC_CLASS(id) = P_STFUNCT;
    ID_STORAGE(id) = STG_NONE;

    implicit_declaration(id);

    /* check argument */
    FOR_ITEMS_IN_LIST(lp,args){
        x = LIST_ITEM(lp);
        if(EXPR_CODE(x) != IDENT && EXPR_CODE(x) != F_ARRAY_REF){
            error("non-variable argument in statement function definition");
            return;
        }
        ip = declare_ident(EXPR_SYM(x),CL_UNKNOWN);

        /* fix type of paramter */
        implicit_declaration(ip);

        if (ID_TYPE(ip) == NULL) {
            fatal("statement function call: unknown type parameter");
            /* not reached. */
            return;
        }
    }
    PROC_STBODY(id) = body;
    PROC_ARGS(id) = args;
}


ID
find_label_from_block(int st_no, BLOCK_ENV block)
{
    ID label = NULL;
    BLOCK_ENV bp;

    if (block == NULL) {
        return NULL;
    }

    FOREACH_ID(label, BLOCK_LOCAL_LABELS(block)) {
        if (LAB_ST_NO(label) == st_no) {
            return label;
        }
    }

    FOREACH_BLOCKS(bp, BLOCK_CHILDREN(block)) {
        label = find_label_from_block(st_no, bp);
        if (label != NULL) {
            return label;
        }
    }

    return NULL;
}


ID
declare_label(int st_no,LABEL_TYPE type,int def_flag)
{
    ID ip,last_ip;
    char name[10];
    CTL cp;
    int in_block = FALSE;
    int not_same_block = FALSE;

    if(st_no <= 0){
        error("illegal label %d", st_no);
        return NULL;
    }

    last_ip = NULL;
    FOREACH_ID(ip, LOCAL_LABELS) {
        if(LAB_ST_NO(ip) == st_no) goto found;
        last_ip = ip;
    }

    FOR_CTLS_BACKWARD(cp) {
        if (CTL_TYPE(cp) == CTL_BLK ||
            CTL_TYPE(cp) == CTL_FORALL ||
            CTL_TYPE(cp) == CTL_ASSOCIATE) {
            in_block = TRUE;
            if (CTL_LABELS(cp) == LOCAL_LABELS) {
                continue;
            }
            FOREACH_ID(ip, CTL_LABELS(cp)) {
                if(LAB_ST_NO(ip) == st_no) {
                    goto found;
                }
            }
        }
    }
    if (in_block) {
        FOREACH_ID(ip, UNIT_CTL_LOCAL_LABELS(CURRENT_UNIT_CTL)) {
            if(LAB_ST_NO(ip) == st_no) {
                goto found;
            }
        }
    }
    if ((ip = find_label_from_block(st_no, LOCAL_BLOCKS)) != NULL) {
        not_same_block = TRUE;
        goto found;
    }
    if ((ip = find_label_from_block(st_no,
                                    UNIT_CTL_LOCAL_BLOCKS(CURRENT_UNIT_CTL))) != NULL) {
        not_same_block = TRUE;
        goto found;
    }

    /* if not found, make label entry */
    sprintf(name,"%05d",st_no);
    ip = new_ident_desc(find_symbol(name));
    ID_LINK_ADD(ip, LOCAL_LABELS, last_ip);

    ID_ADDR(ip) = expv_sym_term(IDENT,NULL,ID_SYM(ip));
    ID_CLASS(ip) = CL_LABEL;
    LAB_ST_NO(ip) = st_no;

 found:
    if(def_flag){

        if(LAB_IS_DEFINED(ip)){
            error("label %d already defined", st_no);
            return ip;
        }
        if(type == LAB_EXEC){
            if(not_same_block == TRUE) {
                error("label %d is not in the same block",st_no);
            }
            if(LAB_IS_USED(ip) && LAB_TYPE(ip) != LAB_FORMAT
               && LAB_BLK_LEVEL(ip) < CURRENT_BLK_LEVEL)
                warning("there is a branch to label %d from outside block",
                      st_no);
            LAB_BLK_LEVEL(ip) = CURRENT_BLK_LEVEL;
            if(LAB_TYPE(ip) == LAB_FORMAT)
                error("label %d is referenced as format number",st_no);
        } else if(type == LAB_FORMAT){
            if(LAB_TYPE(ip) == LAB_EXEC)
                error("format number %d is referenced as label",st_no);
        }
        /* define label */
        LAB_TYPE(ip) = type;
        LAB_IS_DEFINED(ip) = TRUE;
    } else {
        if(not_same_block == TRUE) {
            error("label %d is not in the same block",st_no);
        }

        LAB_IS_USED(ip) = TRUE;         /* referenced */
        if(LAB_TYPE(ip) == LAB_UNKNOWN) LAB_TYPE(ip) = type;

        if(type == LAB_EXEC){
            if(LAB_CANNOT_JUMP(ip))
                warning("illegal branch to inner block, statement %d",st_no);
            if(!LAB_IS_DEFINED(ip))
                LAB_BLK_LEVEL(ip) = CURRENT_BLK_LEVEL;
            if(LAB_TYPE(ip) == LAB_FORMAT)
                error("may not branch to a format");
        } else if(type == LAB_FORMAT){
            if(LAB_TYPE(ip) == LAB_EXEC) error("bad format number");
        }
    }
    return ip;
}

EXT_ID
declare_external_proc_id(SYMBOL s, TYPE_DESC tp, int def_flag)
{
    /* allocate external symbol */
    EXT_ID ep = declare_external_id(s, STG_EXT, def_flag);
    assert(ep != NULL);

    if (def_flag == TRUE || EXT_PROC_TYPE(ep) == NULL) {
        if(tp != NULL 
            && (TYPE_BASIC_TYPE(tp) == TYPE_FUNCTION 
            || TYPE_BASIC_TYPE(tp) == TYPE_SUBR 
            || TYPE_BASIC_TYPE(tp) ==  TYPE_MODULE))
        {
            // Do not set non-function type to PROC
            EXT_PROC_TYPE(ep) = tp;
        }
    } else if(EXT_IS_DEFINED(ep)) {
        /* avoid overwriting EXT_ID already defined. */
        ep = new_external_id_for_external_decl(s, tp);
    }
    return ep;
}


/* 'intern' external symbol with tag.
 *  if def_flag, mark it as defined
 */
EXT_ID
declare_external_id(SYMBOL s, enum storage_class tag, int def_flag)
{
    EXT_ID ep, last_ep;

    if (tag == STG_COMEQ) {
        fatal("creating named common as equivalance??");
    }
    last_ep = NULL;
    for (ep = EXTERNAL_SYMBOLS; ep != NULL; ep = EXT_NEXT(ep)){
        if (EXT_SYM(ep) == s) {
            if (tag == STG_COMMON && EXT_TAG(ep) == tag){
                if (strcmp(SYM_NAME(s), "no__name__blkdata__")) {
                    error("block data '%s' is already defined.",SYM_NAME(s));
                } else {
                    error("unnamed block data is already defined.");
                }
                return NULL;
            }
            break; /* found */
        }
        last_ep = ep;
    }
    if (ep == NULL) {     /* not found */
        ep = new_external_id(s);
        if (last_ep == NULL) {
            EXTERNAL_SYMBOLS = ep;
        } else {
            EXT_NEXT(last_ep) = ep;
        }
        EXT_SYM(ep) = s;
    }
    if (EXT_TAG(ep) != STG_UNKNOWN) { /* not referenced yet */
#ifdef BUGFIX
        if (EXT_IS_DEFINED(ep) && EXT_TAG(ep) != tag) {
            /* defined, but not desired tag */
            error("external name is already used, '%s'",SYM_NAME(s));
            return NULL;
        }
#else
        if(tag == STG_EXT && (EXT_IS_DEFINED(ep) || EXT_TAG(ep) != tag)){
            error("external name is already used, '%s'",SYM_NAME(s));
            return NULL;
        }
#endif
        if (tag == STG_COMMON && EXT_TAG(ep) != tag){
            error("%s cannot be a common block name", SYM_NAME(s));
            return NULL;
        }
    }
    EXT_TAG(ep) = tag;
    if (!EXT_IS_DEFINED(ep)) {
        EXT_IS_DEFINED(ep) = def_flag;
    }
    EXT_IS_OFMODULE(ep) = FALSE;
    return ep;
}


ID
declare_ident(SYMBOL s, enum name_class class)
{
    ID ip,last_ip;
    ID predecl_ip = NULL;
    ID* symbols;
    int isInUseDecl = checkInsideUse();
    int isPreDecl = FALSE;
    int inTypeDecl = FALSE;
    int isInternalPrivate = FALSE; // For struct type
    char msg[2048];
    const char *fmt = "%s '%s' is already declared.";

    symbols = &LOCAL_SYMBOLS;

    /*
     * FIXME:
     *	SUPER BOGUS FLAG ALERT !
     */
    if (is_in_struct_member_initializer_compilation_flag_for_declare_ident ==
        FALSE) {
        if (CTL_TYPE(ctl_top) == CTL_STRUCT) {
            if (class == CL_TAGNAME) {
                isPreDecl = TRUE;
            } else
                /*
                 * FIXME:
                 *	SUPER BOGUS FLAG ALERT !
                 */
            if (is_in_kind_compilation_flag_for_declare_ident == FALSE) {
                TYPE_DESC struct_tp = CTL_STRUCT_TYPEDESC(ctl_top);
                if (CURRENT_STATE == IN_TYPE_PARAM_DECL) {
                    symbols = &TYPE_TYPE_PARAMS(struct_tp);
                    class = CL_TYPE_PARAM;
                } else if (CURRENT_STATE == IN_TYPE_BOUND_PROCS) {
                    isInternalPrivate = TYPE_IS_INTERNAL_PRIVATE(struct_tp);
                    symbols = &TYPE_MEMBER_LIST(struct_tp);
                    class = CL_TYPE_BOUND_PROC;
                } else {
                    isInternalPrivate = TYPE_IS_INTERNAL_PRIVATE(struct_tp);
                    symbols = &TYPE_MEMBER_LIST(struct_tp);
                    class = CL_ELEMENT;
                    inTypeDecl = TRUE;
                }
            }
        }
    }

    last_ip = NULL;
    FOREACH_ID(ip, *symbols) {
        if (ID_SYM(ip) == s) {
            /* if argument 'class' is CL_UNKNOWN, find id */
            if (ID_CLASS(ip) == class) {
                if (class == CL_TAGNAME) {
                    if(TYPE_IS_DECLARED(ID_TYPE(ip)) == FALSE) {
                        predecl_ip = ip;
                        continue;
                    }
                    snprintf(msg, 2048, fmt, "type", SYM_NAME(s));
                    if (isInUseDecl == FALSE) {
                        error(msg);
                        return NULL;
                    } else {
#if 1
                        warning(msg);
                        /*
                         * FIXME:
                         *	USE statement case:
                         *	Need to check struct compatibilities
                         *	and if the newly declared type is
                         *	compatible to the old one, return old
                         *	one's ID.
                         */
#endif
                    }
                }
                return ip;
            }
            if (ID_CLASS(ip) == CL_PROC && class == CL_VAR) {
                return ip;
            }

            if (ID_CLASS(ip) == CL_MULTI &&
                (class == CL_TAGNAME || class == CL_PROC)) {
                symbols = &MULTI_ID_LIST(ip);
                FOREACH_ID(ip, *symbols) {
                    last_ip = ip;
                }
                break;
            }

            if (class == CL_UNKNOWN) {
                return ip;
            }

            /* define name class */
            if (ID_CLASS(ip) == CL_UNKNOWN) {
                ID_CLASS(ip) = class;
            } else if (!(ID_IS_DUMMY_ARG(ip))) {
                snprintf(msg, 2048, fmt, "name", SYM_NAME(s));
                if (isInUseDecl == FALSE) {
                    error(msg);
                    return NULL;
                } else {
#if 1
                    warning(msg);
                    /*
                     * FIXME:
                     *	USE statement case:
                     *	Need to check the compatibilities of newly
                     *	declared ident.
                     */
#endif
                }
            }
            return ip;
        }
        last_ip = ip;
    }

    if(predecl_ip != NULL) {
        ip = predecl_ip;
        ID_ORDER(ip) = order_sequence++;
    } else {    
        ip = new_ident_desc(s);
        // Do not add intrinsic id to the type member list
        if(!(SYM_TYPE(s) == S_INTR && inTypeDecl)) {
            ID_LINK_ADD(ip, *symbols, last_ip);      
        }
        ID_SYM(ip) = s;
        ID_CLASS(ip) = class;
        if(isPreDecl == FALSE)
            ID_ORDER(ip) = order_sequence++;
        if(isInternalPrivate) {
            TYPE_SET_INTERNAL_PRIVATE(ip);
        }
        
        
    }

    switch (class) {
        case CL_TAGNAME:
            ID_STORAGE(ip) = STG_TAGNAME;
            break;
        case CL_TYPE_PARAM:
            ID_STORAGE(ip) = STG_TYPE_PARAM;
            break;
        default:
            ID_STORAGE(ip) = STG_UNKNOWN;
            break;
    }

    return ip;
}


ID
declare_common_ident(SYMBOL s)
{
    ID ip, last_ip;

    last_ip = NULL;
    FOREACH_ID(ip, LOCAL_COMMON_SYMBOLS) {
        if(ID_SYM(ip) == s)
            return ip;
        last_ip = ip;
    }

    ip = new_ident_desc(s);
    ID_LINK_ADD(ip, LOCAL_COMMON_SYMBOLS, last_ip);
    ID_SYM(ip) = s;
    ID_CLASS(ip) = CL_COMMON;
    ID_STORAGE(ip) = STG_UNKNOWN;
    return ip;
}


ID
find_ident_from_type_parameter(SYMBOL s, TYPE_DESC tp)
{
    ID ip;

    if (s == NULL || tp == NULL)
        return NULL;


    if (TYPE_TYPE_PARAMS(tp) != NULL) {
        ip = find_ident_head(s, TYPE_TYPE_PARAMS(tp));
        if (ip != NULL) {
            return ip;
        }
    }

    if (TYPE_PARENT(tp) && TYPE_PARENT_TYPE(tp)) {
        ip = find_ident_from_type_parameter(s, TYPE_PARENT_TYPE(tp));
        if (ip != NULL) {
            return ip;
        }
    }

    return NULL;
}


/*
 * Find identifier from the current scope.
 * If caller is in the block, search the local symbols in the block.
 * If caller is in the function and not in the block, search the local symbols
 * of the function.
 */
ID
find_ident_local(SYMBOL s)
{
    if (CTL_TYPE(ctl_top) == CTL_STRUCT) {
        ID ip;
        TYPE_DESC struct_tp = CTL_STRUCT_TYPEDESC(ctl_top);

        ip = find_ident_from_type_parameter(s, struct_tp);
        if (ip != NULL) return ip;
    }

    return find_ident_head(s, LOCAL_SYMBOLS);
}

/*
 * Find an identifier from scopes between outside of the current block and the
 * local scope of the current procedure/module.
 */
ID
find_ident_block_parent(SYMBOL s)
{
    CTL cp;
    ID ip = NULL;
    int in_block = FALSE;

    FOR_CTLS_BACKWARD(cp) {
        if (CTL_TYPE(cp) == CTL_BLK || \
            CTL_TYPE(cp) == CTL_FORALL || \
            CTL_TYPE(cp) == CTL_ASSOCIATE || \
            CTL_TYPE(cp) == CTL_DO || \
            CTL_TYPE(cp) == CTL_TYPE_GUARD) {
            in_block = TRUE;
            if (CTL_SYMBOLS(cp) == LOCAL_SYMBOLS) {
                continue;
            }
            ip = find_ident_head(s, CTL_SYMBOLS(cp));
            if (ip != NULL) {
                return ip;
            }
        }
    }

    if (in_block &&  UNIT_CTL_LOCAL_SYMBOLS(CURRENT_UNIT_CTL) != LOCAL_SYMBOLS) {
        ip = find_ident_head(s, UNIT_CTL_LOCAL_SYMBOLS(CURRENT_UNIT_CTL));
    }

    return ip;
}

/*
 * Find an identifier from the scopes of the parent program units.
 */
ID
find_ident_parent(SYMBOL s)
{
    ID ip;
    int lev_idx;

    for (lev_idx = unit_ctl_level - 1; lev_idx >= 0; --lev_idx) {
        ip = find_ident_head(s, UNIT_CTL_LOCAL_SYMBOLS(unit_ctls[lev_idx]));
        if (ip != NULL) {
            return ip;
        }
    }
    return NULL;
}

/*
 * Find an identifier from the scopes of the other subprogram units in the
 * current CONTAIN block.
 */
ID
find_ident_sibling(SYMBOL s)
{
    EXT_ID ep;
    ID ip;

    if (unit_ctl_level < 1) {
        return NULL;
    }
    FOREACH_EXT_ID(ep, PARENT_CONTAINS) {
        ip = find_ident_head(s, EXT_PROC_ID_LIST(ep));
        if (ip != NULL) {
            return ip;
        }
    }
    return NULL;
}


ID
find_external_ident_head(SYMBOL s)
{
    EXT_ID ep;
    int parent = !unit_ctl_level?unit_ctl_level:unit_ctl_level-1;
    if(parent == 0) {
        /* global function can be called.*/
        return NULL;
    }
    FOREACH_EXT_ID(ep, UNIT_CTL_LOCAL_EXTERNAL_SYMBOLS(unit_ctls[parent])) {
        if (EXT_SYM(ep) == s)
        switch(EXT_TAG(ep)) {
        case STG_EXT:
            if ((EXT_PROC_BODY(ep) || EXT_PROC_ID_LIST(ep)) &&
                (EXT_PROC_TYPE(ep) != NULL) &&
                (TYPE_BASIC_TYPE(EXT_PROC_TYPE(ep)) != TYPE_MODULE)) {
                ID ip = EXT_PROC_ID_LIST(ep);
                if(ID_SYM(ip) == EXT_SYM(ep))
                    return ip;
            }
            break;
        case STG_COMMON:
            /* TODO
             */
            break;
        default:
            break;
        }
    }
    return NULL;
}


ID
find_external_cont_ident_head(SYMBOL s)
{
    EXT_ID ep;
    int parent = !unit_ctl_level?unit_ctl_level:unit_ctl_level-1;
    FOREACH_EXT_ID(ep, UNIT_CTL_LOCAL_EXTERNAL_SYMBOLS(unit_ctls[parent])) {
        if (EXT_SYM(ep) == s)
        switch(EXT_TAG(ep)) {
        case STG_EXT:
            if ((EXT_PROC_IS_ENTRY(ep) == FALSE) &&
                (EXT_PROC_BODY(ep) || EXT_PROC_ID_LIST(ep)) &&
                (EXT_PROC_TYPE(ep) != NULL) &&
                (TYPE_BASIC_TYPE(EXT_PROC_TYPE(ep)) != TYPE_MODULE)) {
                ID ip = EXT_PROC_ID_LIST(ep);
                if(ID_SYM(ip) == EXT_SYM(ep))
                    return ip;
            }
            break;
        case STG_COMMON:
            /* TODO
             */
            break;
        default:
            break;
        }
    }
    return NULL;
}


/*
 * Find an identifier from the outiside of the current scope.
 */
ID
find_ident_outer_scope(SYMBOL s)
{
    ID ip;

    ip = find_ident_block_parent(s);
    if (ip != NULL) {
        return ip;
    }
    ip = find_ident_sibling(s);
    if (ip != NULL) {
        return ip;
    }
    ip = find_ident_parent(s);
    if (ip != NULL) {
        return ip;
    }
    ip = find_external_ident_head(s);
    return ip;
}


/*
 * Find an identifier from the all scopes thoses the caller can access.
 */
ID
find_ident(SYMBOL s)
{
    ID ip;

    ip = find_ident_local(s);
    if (ip != NULL) {
        return ip;
    }
    if (in_interface() && !in_module_procedure()) {
        return NULL;
    }
    return find_ident_outer_scope(s);
}



EXT_ID
find_ext_id_parent(SYMBOL s)
{
    EXT_ID ep;

    if (unit_ctl_level == 0)
        return NULL;
    ep = find_ext_id_head(s,
         UNIT_CTL_LOCAL_EXTERNAL_SYMBOLS(unit_ctls[unit_ctl_level - 1]));
    return ep;
}

EXT_ID
find_ext_id_sibling(SYMBOL s)
{
    EXT_ID ep;

    if (unit_ctl_level < 1) {
        return NULL;
    }

    if ((ep = find_ext_id_head(s, PARENT_INTERFACE)) != NULL)
        return ep;
    if (PARENT_EXT_ID != NULL &&
        (ep = find_ext_id_head(s, PARENT_CONTAINS)) != NULL)
        return ep;
    return NULL;
}

EXT_ID
find_ext_id(SYMBOL s)
{
    EXT_ID ep;

    ep = find_ext_id_head(s, EXTERNAL_SYMBOLS);
    if (ep != NULL) {
        return ep;
    }
    ep = find_ext_id_parent(s);
    if (ep != NULL) {
        return ep;
    }
    ep = find_ext_id_sibling(s);
    return ep;
}

ID
find_common_ident_parent(SYMBOL s)
{
    ID ip;
    int lev_idx;

    for (lev_idx = unit_ctl_level - 1; lev_idx >= 0; --lev_idx) {
        ip = find_ident_head(s,
                UNIT_CTL_LOCAL_COMMON_SYMBOLS(unit_ctls[lev_idx]));
        if (ip != NULL) {
            return ip;
        }
    }
    return NULL;
}

ID
find_common_ident_sibling(SYMBOL s)
{
    ID ip;
    EXT_ID ep;

    if (unit_ctl_level < 1) {
        return NULL;
    }
    FOREACH_EXT_ID(ep, PARENT_CONTAINS) {
        ip = find_ident_head(s, EXT_PROC_COMMON_ID_LIST(ep));
        if (ip != NULL) {
            return ip;
        }
    }
    return NULL;
}

ID
find_common_ident(SYMBOL s)
{
    ID ip;

    ip = find_ident_head(s, LOCAL_COMMON_SYMBOLS);
    if (ip != NULL) {
        return ip;
    }
    ip = find_common_ident_parent(s);
    if (ip != NULL) {
        return ip;
    }
    ip = find_common_ident_sibling(s);
    return ip;
}

static TYPE_DESC
declare_struct_type(expr ident)
{
    SYMBOL sp = EXPR_SYM(ident);
    ID id = NULL;
    TYPE_DESC tp0, tp = NULL, ttail;
    int isInUseDecl = checkInsideUse();
    char msg[2048];

    ttail = NULL;
    for (tp0 = LOCAL_STRUCT_DECLS; tp0 != NULL; tp0 = TYPE_SLINK(tp0)) {
        if (ID_SYM(TYPE_TAGNAME(tp0)) == sp) {
            if(!TYPE_IS_DECLARED(tp0)) {
                tp = tp0;
                continue;
            }
            snprintf(msg, 2048, "type '%s' is already declared.",
                     SYM_NAME(sp));
            if (isInUseDecl == FALSE) {
                error(msg);
                return NULL;
            } else {
#if 1
                warning(msg);
#endif
                return tp0;
            }
        }
        ttail = tp0;
    }

    if(tp == NULL) {
        if ((id = find_ident_local(sp)) == NULL) {
            id = declare_ident(sp, CL_TAGNAME);
        }else{
            if (ID_CLASS(id) == CL_UNKNOWN) {
                ID_CLASS(id) = CL_TAGNAME;
                ID_STORAGE(id) = STG_TAGNAME;

            } else if (ID_CLASS(id) == CL_PROC &&
                  ID_TYPE(id) != NULL &&
                  IS_GENERIC_PROCEDURE_TYPE(ID_TYPE(id))) {
                /*
                 * There is the generic procedure with the same name,
                 * so turn id into the multi class identifier.
                 */
                id_multilize(id);
                id = declare_ident(sp, CL_TAGNAME);
            } else {
                error("identifier '%s' is already used", ID_NAME(id));
            }
        }
        tp = struct_type(id);
        TYPE_SLINK_ADD(tp, LOCAL_STRUCT_DECLS, ttail);
        ID_TYPE(id) = tp;
    } else if(TYPE_IS_DECLARED(tp) == FALSE) {
        id = declare_ident(sp, CL_TAGNAME);
    }
    ID_LINE(id) = EXPR_LINE(ident);
    ID_ORDER(id) = order_sequence++;

    return tp;
}

/* declare struct type wihtout component. */
TYPE_DESC
declare_struct_type_wo_component(expr ident)
{
    SYMBOL sp = EXPR_SYM(ident);
    ID id;
    TYPE_DESC tp0, tp = NULL, ttail;

    if(ident == NULL)
        fatal("compile_struct_decl: F95 derived type name is NULL");
    if(EXPR_CODE(ident) != IDENT)
        fatal("compile_struct_decl: not IDENT");

    ttail = NULL;
    for (tp0 = LOCAL_STRUCT_DECLS; tp0 != NULL; tp0 = TYPE_SLINK(tp0)) {
        ttail = tp0;
    }

    id = declare_ident(sp, CL_TAGNAME);
    tp = struct_type(id);
    TYPE_SLINK_ADD(tp, LOCAL_STRUCT_DECLS, ttail);
    ID_TYPE(id) = tp;

    return tp;
}

/* declare type for F95 attributes */
/* check compatibility if id's type is already declared. */
static TYPE_DESC
declare_type_attributes(ID id, TYPE_DESC tp, expr attributes,
                        int ignoreDims, int ignoreCodims)
{
    expr v;
    list lp;

    // The ALLOCATABLE/POINTER attribute must be checked in advance.
    FOR_ITEMS_IN_LIST(lp, attributes){
        v = LIST_ITEM(lp);
        if (EXPR_CODE(v) == F95_ALLOCATABLE_SPEC){
            TYPE_SET_ALLOCATABLE(tp);
            break;
        }
        else if (EXPR_CODE(v) == F95_POINTER_SPEC){
            TYPE_SET_POINTER(tp);
            break;
        }
    }

    FOR_ITEMS_IN_LIST(lp, attributes) {
        v = LIST_ITEM(lp);

        if (EXPR_CODE(v) == F95_INTENT_SPEC ||
            EXPR_CODE(v) == F95_OPTIONAL_SPEC) {
            if (id != NULL && !(ID_IS_DUMMY_ARG(id))) {
                error_at_node(attributes, "\"%s\" is not a dummy argument.",
                              SYM_NAME(ID_SYM(id)));
                continue;
            }
        }

        if (debug_flag)
            fprintf(debug_fp,"<!-- %s -->\n", EXPR_CODE_NAME(EXPR_CODE(v)) );

        switch(EXPR_CODE(v)) {
        case F95_PARAMETER_SPEC:
            /* see compile_PARAMETER_decl() */
            TYPE_SET_PARAMETER(tp);
            break;
        case F95_ALLOCATABLE_SPEC:
            TYPE_SET_ALLOCATABLE(tp);
            break;
        case F95_DIMENSION_SPEC:
            /* see compile_dimensions() */
            if (ignoreDims == FALSE) {
                tp = compile_dimensions(tp, EXPR_ARG1(v));
            }
            break;
        case XMP_CODIMENSION_SPEC:
            if (ignoreDims) break;
            /* if (is_descendant_coindexed(tp)){ */
            /*   error_at_node(EXPR_ARG1(v), "The derived-type of the coindexed " */
            /*                 "object cannot have a coindexed member."); */
            /*   return NULL; */
            /* } */

            codims_desc *codesc = compile_codimensions(EXPR_ARG1(v),
                                                       TYPE_IS_ALLOCATABLE(tp));
            if (codesc)
                tp->codims = codesc;
            else {
                error_at_node(EXPR_ARG1(v), "Wrong codimension declaration.");
                return NULL;
            }
            break;
        case F95_EXTERNAL_SPEC:
            /* see compile_EXTERNAL_decl() */
            TYPE_SET_EXTERNAL(tp);
            if (TYPE_REF(tp)) {
                TYPE_SET_EXTERNAL(TYPE_REF(tp));
            }
            break;
        case F95_INTENT_SPEC:
            switch(EXPR_CODE(EXPR_ARG1(v))) {
            case F95_IN_EXTENT:
                TYPE_SET_INTENT_IN(tp);
                break;
            case F95_OUT_EXTENT:
                TYPE_SET_INTENT_OUT(tp);
                break;
            case F95_INOUT_EXTENT:
                TYPE_SET_INTENT_INOUT(tp);
                break;
            default:
                TYPE_UNSET_INTENT_IN(tp);
                TYPE_UNSET_INTENT_OUT(tp);
                TYPE_UNSET_INTENT_INOUT(tp);
                break;
            }
            break;
        case F95_INTRINSIC_SPEC:
            TYPE_SET_INTRINSIC(tp);
            FUNCTION_TYPE_SET_VISIBLE_INTRINSIC(tp);
            break;
        case F95_OPTIONAL_SPEC:
            TYPE_SET_OPTIONAL(tp);
            break;
        case F95_POINTER_SPEC:
            TYPE_SET_POINTER(tp);
            break;
        case F95_SAVE_SPEC:
            TYPE_SET_SAVE(tp);
            break;
        case F95_TARGET_SPEC:
            TYPE_SET_TARGET(tp);
            break;
        case F95_PUBLIC_SPEC:
            if (TYPE_IS_PRIVATE(tp)) {
                error_at_node(attributes, "'%s' is already specified as "
                              "private.", ID_NAME(id));
                return NULL;
            }
            TYPE_SET_PUBLIC(tp);
            TYPE_UNSET_PRIVATE(tp);
            break;
        case F95_PRIVATE_SPEC:
            if (TYPE_IS_PUBLIC(tp)) {
                error_at_node(attributes, "'%s' is already specified as "
                              "public.", ID_NAME(id));
                return NULL;
            }
            TYPE_UNSET_PUBLIC(tp);
            TYPE_SET_PRIVATE(tp);
            break;
        case F03_PROTECTED_SPEC:
            TYPE_SET_PROTECTED(tp);
            break;
        case F03_VOLATILE_SPEC:
            TYPE_SET_VOLATILE(tp);
            break;
        case F03_KIND_SPEC:
            if (CTL_TYPE(ctl_top) != CTL_STRUCT) {
                error_at_node(attributes,
                              "KIND attribute outside of TYPE declaration");
                return NULL;
            }
            if (TYPE_BASIC_TYPE(tp) != TYPE_INT) {
                /* kind arg must be integer type. */
                error_at_node(attributes,
                              "KIND to a no-integer type parameter '%s'",
                              ID_NAME(id));
                return NULL;
            }

            TYPE_SET_KIND(tp);
            break;
        case F03_LEN_SPEC:
            if (CTL_TYPE(ctl_top) != CTL_STRUCT) {
                error_at_node(attributes,
                              "LEN attribute outside of TYPE declaration");
                return NULL;
            }
            if (TYPE_BASIC_TYPE(tp) != TYPE_INT) {
                /* len arg must be integer type. */
                error_at_node(attributes,
                              "LEN to a no-integer type parameter '%s'",
                              ID_NAME(id));
                return NULL;
            }

            TYPE_SET_LEN(tp);
            break;
        case F03_BIND_SPEC:
            TYPE_SET_BIND(tp);
            if(EXPR_ARG1(v) != NULL) {
                // BIND(C, NAME='')
                TYPE_BIND_NAME(tp) = EXPR_ARG1(v);
            }
            break;
        case F03_VALUE_SPEC:
            TYPE_SET_VALUE(tp);
            break;
        case F03_ASYNCHRONOUS_SPEC:
            TYPE_SET_ASYNCHRONOUS(tp);
            break;
        case F08_CONTIGUOUS_SPEC:
            TYPE_SET_CONTIGUOUS(tp);
            break;

        default:
            error("incompatible type attribute , code: %d", EXPR_CODE(v));
        }
    }
    return tp;
}


/*
 * type handling
 */
/* declare type for id */
/* check compatibility if id's type is already declared. */
void
declare_id_type(ID id, TYPE_DESC tp)
{
    TYPE_DESC * id_type;
    TYPE_DESC tq, tpp;
    int isInUseDecl = checkInsideUse();

    if (tp == NULL) {
        return; /* nothing for TYPE_UNKNOWN */
    }

    if (ID_TYPE(id) != NULL &&
        ID_TYPE(id) != tp &&
        TYPE_IS_UNCHANGABLE(ID_TYPE(id))) {
        error("type of %s can not be changed",
              ID_NAME(id));
    }

    if (IS_MODULE(tp)) {      /* tp is module */
        ID_TYPE(id) = tp;
        return;
    }

    id_type = &(ID_TYPE(id));
    if (ID_CLASS(id) == CL_PROC ||
        IS_FUNCTION_TYPE(ID_TYPE(id))) {

        if (ID_TYPE(id) == NULL) {
            if (!IS_PROCEDURE_TYPE(tp) || IS_PROCEDURE_POINTER(tp)) {
                /* change the type into function type */
                *id_type = function_type(tp);
            } else {
                ID_TYPE(id) = tp;
                return;
            }
        }

        if (IS_FUNCTION_TYPE(ID_TYPE(id))) {
            /*
             * expects tp replace FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(id))
             */
            id_type = &(FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(id)));
        }
    }

    if (*id_type == tp) {
        return; /* nothing for TYPE_UNKNOWN */
    }

    if (IS_STRUCT_TYPE(tp)) {      /* tp is struct */
        replace_or_assign_type(id_type, tp);
        return;
    }

    tq = *id_type;
    if (tq != NULL && (TYPE_IS_IMPLICIT(tq) || TYPE_IS_NOT_FIXED(tq))) {
        /* override implicit declared type */
        TYPE_ATTR_FLAGS(tp) |= TYPE_ATTR_FLAGS(tq);
        TYPE_EXTATTR_FLAGS(tp) |= TYPE_EXTATTR_FLAGS(tq);
        replace_or_assign_type(id_type, tp);
        return;
    }

    if (IS_ARRAY_TYPE(tp)) {      /* tp is array */
        if (ID_CLASS(id) == CL_UNKNOWN) {
            ID_CLASS(id) = CL_VAR;
        } else if (ID_CLASS(id) != CL_VAR &&
                   ID_CLASS(id) != CL_ELEMENT &&
                   ID_CLASS(id) != CL_PROC
            ) {
            error("array must be a variable, a struct element or a function return value, %s",
                      ID_NAME(id));
            return;
        }
        if (tq != NULL) {
            if (IS_ARRAY_TYPE(tq)) {
                char msg[2048];
                snprintf(msg, 2048, "'%s' is already declared as an array.",
                         ID_NAME(id));
                if (isInUseDecl == FALSE) {
                    error(msg);
                } else {
#if 1
                    warning(msg);
                    /*
                     * FIXME:
                     *	USE statement case:
                     *	Need to check the compatibility of newly
                     *	declared array.
                     */
#endif
                }
                return;
            }
            /* declared as scalar type, then array declaration come later. */
            tpp = tp;
            while(TYPE_REF(tpp) != NULL && IS_ARRAY_TYPE(TYPE_REF(tpp)))
                tpp = TYPE_REF(tpp);
            if (TYPE_REF(tpp) != NULL && !type_is_soft_compatible(tq, TYPE_REF(tpp)))
                goto no_compatible;
            if (TYPE_REF(tpp) == NULL ||
                type_is_specific_than(tq, TYPE_REF(tpp)))
                TYPE_REF(tpp) = tq;
        }
        *id_type = tp;
        return;
    }

    /* tp is not ARRAY_TYPE */
    if(tq != NULL && IS_ARRAY_TYPE(tq)){
        /* already defined as array */
        while(TYPE_REF(tq) != NULL && IS_ARRAY_TYPE(TYPE_REF(tq)))
            tq = TYPE_REF(tq);
        if (TYPE_REF(tq) != NULL && !type_is_soft_compatible(TYPE_REF(tq),tp))
            goto no_compatible;
        if (TYPE_REF(tq) == NULL ||
            type_is_specific_than(tp, TYPE_REF(tq))) {
            TYPE_REF(tq) = tp;
            return;
        }
    }

    /* both are not ARRAY_TYPE */

    if(tq != NULL && IS_SUBR(tp) && ID_IS_DUMMY_ARG(id)) {
        /* if argument, may override with TYPE_SUBR ??? */
        replace_or_assign_type(id_type, tp);
        return;
    } else if(tq == NULL || type_is_soft_compatible(tq, tp)){
        replace_or_assign_type(id_type, tp);
        if (ID_CLASS(id) == CL_PROC &&
            (TYPE_IS_RECURSIVE(id) ||
             TYPE_IS_RECURSIVE(tp))) {
            /* copy recursive flag */
            PROC_IS_RECURSIVE(id) = TRUE;
        }
        if (ID_CLASS(id) == CL_PROC &&
            (TYPE_IS_PURE(id) ||
             TYPE_IS_PURE(tp))) {
            /* copy pure flag */
            PROC_IS_PURE(id) = TRUE;
        }
        return;
    }

 no_compatible:
    if(checkInsideUse() && ID_IS_OFMODULE(id)) {
        /* TODO:
         *  the type of id turns into ambiguous type.
         */
      // to be resolved
        return;
    }

    error("incompatible type declarations, %s", ID_NAME(id));
}


/**
 * create derived-type TYPE_DESC from type expression x.
 */
TYPE_DESC
compile_derived_type(expr x, int allow_predecl)
{
    SYMBOL sym = NULL;
    ID id;
    TYPE_DESC tp = NULL;

    int is_class = FALSE;
    int is_parameterized_type = FALSE;

    if (EXPR_CODE(x) == F03_CLASS) {
        x = EXPR_ARG1(x);

        if (x == NULL) {
            /*
             * for `CLASS(*)`
             */
            tp = struct_type(NULL);
            TYPE_SET_CLASS(tp);
            return tp;
        }

        is_class = TRUE;
    }

    if (EXPR_CODE(x) == IDENT) {
        sym = EXPR_SYM(x);
    } else if (EXPR_CODE(x) == F03_PARAMETERIZED_TYPE) {
        is_parameterized_type = TRUE;
        sym = EXPR_SYM(EXPR_ARG1(x));
    } else {
        error_at_node(x, "type %s is not a derived-type.",
                      SYM_NAME(EXPR_SYM(x)));
    }

    id = find_ident_local(sym);
    if(id != NULL && ID_IS_AMBIGUOUS(id)) {
        error_at_node(x, "an ambiguous reference to symbol '%s'", ID_NAME(id));
        return NULL;
    }

    tp = find_struct_decl(sym);
    if (tp == NULL) {
        if(allow_predecl && !is_parameterized_type) {
            tp = declare_struct_type_wo_component(x);
            if (tp == NULL) {
                return NULL;
            }
        } else {
            error_at_node(x, "type %s not found",
                          SYM_NAME(sym));
            return NULL;
        }
    } else if (EXPR_CODE(x) == IDENT &&
               type_param_values_required(tp)) {
        error_at_node(x, "struct type '%s' requires type parameter values",
                      SYM_NAME(sym));
        return NULL;
    }

    if(CTL_TYPE(ctl_top) == CTL_STRUCT) {
        /*
         * member of SEQUENCE struct must be SEQUENCE.
         */
        if (TYPE_IS_SEQUENCE(CTL_STRUCT_TYPEDESC(ctl_top)) &&
            TYPE_IS_SEQUENCE(tp) == FALSE) {
            error_at_node(x, "type %s does not have SEQUENCE attribute.",
                          SYM_NAME(sym));
            return NULL;
        }
    }

    if (is_parameterized_type) {
        tp = type_apply_type_parameter(tp, EXPR_ARG2(x));
    }

    tp = wrap_type(tp);

    if (is_class) {
        TYPE_SET_CLASS(tp);
    }

    return tp;
}


/* create TYPE_DESC from type expression x. */
/* x := (LIST basic_type leng_spec)
 * leng_spec = NULL | expr | (LIST)
 */
TYPE_DESC
compile_basic_type(expr x)
{
    expr r1, r2 = NULL;
    int kind = 0, isKindConst = 1, kindByLen = 0, charLen = 0;
    BASIC_DATA_TYPE t;
    TYPE_DESC tp = NULL;
    expr rkind = NULL, rcharLen = NULL;
    expv vkind = NULL, vkind2 = NULL, vcharLen = NULL, vcharLen1 = NULL;
    // expv org_vkind = NULL;
    int isInsideExpression = FALSE;

    if(x == NULL) return NULL;

    r1 = EXPR_ARG1(x);
    r2 = EXPR_ARG2(x);
    isInsideExpression = EXPR_HAS_ARG3(x)?TRUE:FALSE;

    if(r1 == NULL && r2) {
        if(r2) {
            error("invalid type-length");
            return NULL;
        }
    }

    if(EXPR_CODE(r1) != F_TYPE_NODE) {
        if (EXPR_CODE(r1) != F95_RECURSIVE_SPEC)
            fatal("compile_type: nor F_TYPE_NODE or F95_RECURSIVE_SPEC");
    }

    t = EXPR_TYPE(r1);
    if(r2) {
        if(EXPR_CODE(r2) == LIST) {
            expr r21 = expr_list_get_n(r2, 0);
            expr r22 = expr_list_get_n(r2, 1);
            if(r21 || r22) {
                if(r21 && EXPR_CODE(r21) == F95_KIND_SELECTOR_SPEC)
                    rkind = r21;
                else if(r22 && EXPR_CODE(r22) == F95_KIND_SELECTOR_SPEC)
                    rkind = r22;

                if(r21 && (EXPR_CODE(r21) == F95_LEN_SELECTOR_SPEC))
                    rcharLen = r21;
                else if(r22 && EXPR_CODE(r22) == F95_LEN_SELECTOR_SPEC)
                    rcharLen = r22;
            } else if(t == TYPE_CHAR) {
                charLen = CHAR_LEN_UNFIXED;
            }
        } else if(t == TYPE_CHAR) {
            rcharLen = r2;
        } else {
            rkind = r2;
            if(EXPR_CODE(r2) == F95_LEN_SELECTOR_SPEC)
                kindByLen = 1;
        }
    }

    if(rkind) {
        /*
         * FIXME:
         *  SUPER BOGUS FLAG ALERT !
         */
        is_in_kind_compilation_flag_for_declare_ident = TRUE;
        //org_vkind = vkind = compile_expression(rkind);
        vkind = compile_expression(rkind);
        is_in_kind_compilation_flag_for_declare_ident = FALSE;

        if(vkind == NULL)  return NULL;

        vkind2 = expv_reduce(vkind, FALSE);
        if(vkind2 != NULL)  vkind = vkind2;

        if(IS_INT_CONST_V(vkind)) {
            if(EXPV_CODE(vkind) == INT_CONSTANT) {
                kind = EXPV_INT_VALUE(vkind);
                if(kindByLen && t == TYPE_COMPLEX) {
                    if(kind % 2 == 1) {
                        error("complex*%d is not supported.", kind);
                        return NULL;
                    }
                    kind /= 2;
                    // org_vkind = vkind = expv_int_term(INT_CONSTANT, type_INT, kind);
                    vkind = expv_int_term(INT_CONSTANT, type_INT, kind);
                }
            } else if(kindByLen) {
                error("cannot reduce length parameter");
                return NULL;
            } else {
                isKindConst = 0;
            }

            if(isKindConst && kind <= 0) {
                error("kind parameter must be positive");
                return NULL;
            }
        } else if(kindByLen) {
            error("cannot reduce length parameter");
            return NULL;
        } else {
            /* could be any expression */
            isKindConst = 0;
        }
    }

    if(rcharLen) {
        /*
         * gfortran accepts a variable as `rchaLen`.
         * So don't check rcharLen is a constant or not.
         */
        if ((vcharLen = compile_expression(rcharLen)) == NULL)
            return NULL;

        if (EXPV_CODE(vcharLen) == F_ASTERISK) {
            charLen = CHAR_LEN_UNFIXED;
        } else if (EXPV_CODE(vcharLen) == F08_LEN_SPEC_COLON) {
            charLen = CHAR_LEN_ALLOCATABLE;
        } else {
            vcharLen1 = expv_reduce(vcharLen, TRUE);
            if (vcharLen1 != NULL && EXPV_CODE(vcharLen1) == INT_CONSTANT) {
                vcharLen = vcharLen1;
                charLen = EXPV_INT_VALUE(vcharLen1);
            } else if (vcharLen1 != NULL && expv_is_specification(vcharLen1)) {
                vcharLen = vcharLen1;
                charLen = 0;
            } else if (vcharLen1 != NULL && isInsideExpression) {
                /* NOTE:
                 *  If the type specifier is used inside an expression,
                 *  LENGHT can be a variable.
                 *
                 * ex)
                 *    INTEGER :: k = 8
                 *    CHARACTER(8), DIMENSION(1:3) c = (/CHARACTER(k) :: 'a', 'b', 'c'
                 *    !                                  ^^^^^^^^^^^^
                 *    ! `k` is not a parameter, but the above code works.
                 */
                TYPE_DESC tp = EXPV_TYPE(vcharLen1);
                if (tp != NULL &&
                    (TYPE_BASIC_TYPE(tp) != TYPE_INT &&
                     TYPE_BASIC_TYPE(tp) != TYPE_GNUMERIC &&
                     TYPE_BASIC_TYPE(tp) != TYPE_GNUMERIC_ALL)) {
                    error("unexpected type of length in the characater ");
                }
                vcharLen = vcharLen1;
                charLen = 0;
            } else if (!expv_is_specification(vcharLen1)) {
                error("unexpected type of length in the characater ");
            } else {
                /* unexpected expression, but ignore it */
                vcharLen = vcharLen1;
                charLen = CHAR_LEN_UNFIXED;
            }
        }
    } else if(charLen == 0) {
        charLen = 1;
    }

    switch(t) {

    case TYPE_CHAR:
        tp = type_char(charLen);
        TYPE_KIND(tp) = vkind;
        TYPE_LENG(tp) = vcharLen;
        break;

    case TYPE_INT:
        if(isKindConst) {
            if(kind == 0)
                tp = type_basic(TYPE_INT);
            else {
                switch(kind) {
                case 1:
                case 2:
                case 4:
                case 8:
                case 16:
                    break;
                default:
                    error("integer(%d) is not supported.", kind);
                    return NULL;
                }
            }
        }
        break;

    case TYPE_LOGICAL:
        if(isKindConst) {
            if(kind == 0)
                tp = type_basic(TYPE_LOGICAL);
            else {
                switch(kind) {
                case 1:
                case 2:
                case 4:
                case 8:
                case 16:
                    break;
                default:
                    error("logical(%d) is not supported.", kind);
                    return NULL;
                }
            }
        }
        break;

    case TYPE_REAL:
        if(isKindConst) {
            if(kind == 0)
                tp = type_basic(TYPE_REAL);
            else {
                switch(kind) {
                case 4:
                case 8:
                case 10: /* gfortran 32bit */
                case 16: /* ifort */
                    break;
                default:
                    error("real(%d) is not supported.", kind);
                    return NULL;
                }
            }
        }
        break;

    case TYPE_DREAL:
        if(vkind) {
            error("invalid length specification");
            return NULL;
        }
        tp = type_basic(TYPE_DREAL);
        break;

    case TYPE_COMPLEX:
        if(isKindConst) {
            if(kind == 0)
                tp = type_basic(TYPE_COMPLEX);
            else {
                switch(kind) {
                case 4:
                case 8:
                case 10: /* gfortran 32bit */
                case 16: /* ifort */
                    break;
                default:
                    error("complex(%d) is not supported.", kind);
                    return NULL;
                }
            }
        }
        break;

    case TYPE_DCOMPLEX:
        if (vkind) {
            error("invalid length specification");
            return NULL;
        }
        tp = type_basic(TYPE_DCOMPLEX);
        break;

    default:
        error("bad type name");
        return NULL;
    }

    if(tp == NULL) {
        tp = type_basic(t);
        //TYPE_KIND(tp) = org_vkind ? org_vkind : vkind;
        TYPE_KIND(tp) = vkind;
    }

    return tp;
}

TYPE_DESC
compile_type(expr x, int allow_predecl)
{
    if (x == NULL) {
        return NULL;
    } else if (EXPR_CODE(x) == IDENT ||
        EXPR_CODE(x) == F03_PARAMETERIZED_TYPE ||
        EXPR_CODE(x) == F03_CLASS) {
        return compile_derived_type(x, allow_predecl);

    } else {
        return compile_basic_type(x);
    }
}


void
compile_IMPLICIT_decl(expr type,expr l)
{
    TYPE_DESC tp = NULL;
    list lp;
    expr v, ty;

    if (UNIT_CTL_IMPLICIT_NONE(CURRENT_UNIT_CTL)) {
        error("IMPLICIT NONE is already set");
        return;
    }

    if(type == NULL){   /* DIMENSION */
        error("bad IMPLICIT declaration");
        return;
    }
    if (EXPR_CODE (type) == IDENT ||
        EXPR_CODE (type) == F03_PARAMETERIZED_TYPE ||
        EXPR_CODE (type) == F03_CLASS) {
        tp = compile_derived_type(type, FALSE);

    } else {
        ty = EXPR_ARG1 (type);
        if (EXPR_CODE (ty) != F_TYPE_NODE) {
          fatal ("compile_IMPLICIT_decl: not F_TYPE_NODE");
          return;
        }
        if ((BASIC_DATA_TYPE) EXPR_INT (ty) == TYPE_UNKNOWN) {
          set_implicit_type(NULL,'a','z');

          /* store implict none decl. */
          list_put_last(UNIT_CTL_IMPLICIT_DECLS(CURRENT_UNIT_CTL),
                        create_implicit_decl_expv(NULL, "a", "z"));
          return;
        } else {
            tp = compile_basic_type(type);
            if(tp == NULL) return;  /* error */
            if(l == NULL) {
                error("no implicit set");
                return;
            }
        }
    }

    if (type_is_nopolymorphic_abstract(tp)) {
        error("abstract types in IMPLICIT statement");
        return;
    }

    FOR_ITEMS_IN_LIST(lp,l){
        v = LIST_ITEM(lp);
        if(EXPR_CODE(v) == IDENT)
            set_implicit_type(tp,*(SYM_NAME(EXPR_SYM(v))),
                              * (SYM_NAME(EXPR_SYM(v))));
        else
            set_implicit_type(tp,*SYM_NAME(EXPR_SYM(EXPR_ARG1(v))),
                              *SYM_NAME(EXPR_SYM(EXPR_ARG2(v))));

        /* store implict decl. */
        EXPV_TYPE(v) = tp;
        list_put_last(UNIT_CTL_IMPLICIT_DECLS(CURRENT_UNIT_CTL), v);
    }
}

static void
set_implicit_type_declared_uc(UNIT_CTL uc, int c)
{
    int i = 1;
    i <<= (c - 'a');
    UNIT_CTL_IMPLICIT_TYPE_DECLARED(uc) |= i;
}

static int
is_implicit_type_declared_uc(UNIT_CTL uc, int c)
{
    int i = 1;
    i <<= (c - 'a');
    return UNIT_CTL_IMPLICIT_TYPE_DECLARED(uc) & i;
}

void
set_implicit_type_uc(UNIT_CTL uc, TYPE_DESC tp, int c1, int c2,
                     int ignore_declared_flag)
{
    int i;

    if (c1 == 0 || c2 == 0)
        return;

    if (c1 > c2) {
        error("characters out of order in IMPLICIT:%c-%c", c1, c2);
        return;
    }

    if (tp)
        TYPE_SET_IMPLICIT(tp);

    for (i = c1 ; i <= c2 ; ++i) {
        if (ignore_declared_flag) {
            UNIT_CTL_IMPLICIT_TYPES(uc)[i - 'a'] = tp;
        } else {
            if (!is_implicit_type_declared_uc(uc, i)) {
                UNIT_CTL_IMPLICIT_TYPES(uc)[i - 'a'] = tp;
                set_implicit_type_declared_uc(uc, i);
            } else {
                error("character '%c' already has IMPLICIT type", i);
            }
        }
    }
}

void
set_implicit_type(TYPE_DESC tp, int c1, int c2)
{
    set_implicit_type_uc(CURRENT_UNIT_CTL, tp, c1, c2, FALSE);
}

void
set_implicit_storage_uc(UNIT_CTL uc, enum storage_class stg,int c1,int c2)
{
    int i;

    if (c1 == 0 || c2 == 0)
        return;

    if (c1 > c2) {
        error("characters out of order in implicit:%c-%c", c1, c2);
    } else {
        for (i = c1 ; i<=c2 ; ++i)
          UNIT_CTL_IMPLICIT_STG(uc)[i - 'a'] = stg;
    }
}

void
set_implicit_storage(enum storage_class stg,int c1,int c2)
{
    set_implicit_storage_uc(CURRENT_UNIT_CTL, stg, c1, c2);
}


expv
expv_reduce_kind(expv v)
{
    expv ret = expv_reduce(v, TRUE); /* reduce parameter */
    
    if (EXPV_CODE(ret) == INT_CONSTANT) {
        return ret; // already reduced
    }

    switch (EXPV_CODE(ret)) {
    case FUNCTION_CALL: {
        char *name = NULL;
        SYMBOL s = EXPV_NAME(EXPR_ARG1(ret));
        ID fId = find_ident(s);
        
        if(fId != NULL && PROC_CLASS(fId) == P_INTRINSIC){
            name = SYM_NAME(ID_SYM(fId));
        } else if(SYM_TYPE(s) == S_INTR){
            name = SYM_NAME(s);
        }

        if (name == NULL) return NULL;
        if (strncasecmp("kind", name, 4) == 0 ||
            strncasecmp("selected_int_kind", name, 17) == 0 ||
	    strncasecmp("selected_char_kind", name, 18) == 0) {
            expv arg = expr_list_get_n(EXPR_ARG2(ret), 0);
            arg = expv_reduce_kind(arg);
            if(arg == NULL) return NULL;
            return expv_cons(FUNCTION_CALL, type_INT, 
                             EXPR_ARG1(ret), list1(LIST,arg));
        } else if (strncasecmp("selected_real_kind", name, 18) == 0) {
            expv arg = EXPR_ARG2(ret);
            int n = expr_list_length(arg);
            if(n == 1){
                expv arg1 = expr_list_get_n(arg, 0);
                if(arg1 == NULL) return NULL; // error
                return expv_cons(FUNCTION_CALL, type_INT, 
                                 EXPR_ARG1(ret), list1(LIST,arg1));
            } else if(n == 2){
                expv arg1 = expr_list_get_n(arg, 0);
                expv arg2 = expr_list_get_n(arg, 1);
            
                if(arg1 == NULL || arg2 == NULL) return NULL; // error
                arg1 = expv_reduce_kind(arg1);
                arg2 = expv_reduce_kind(arg2);
                return expv_cons(FUNCTION_CALL, type_INT, 
                                 EXPR_ARG1(ret), list2(LIST,arg1,arg2));
            } 
	}
        break;
    }
    default: 
        break;
    }

    if (EXPV_CODE(ret) != INT_CONSTANT &&
        EXPV_CODE(ret) != FLOAT_CONSTANT &&
	EXPV_CODE(ret) != STRING_CONSTANT) {
        ret = NULL;  // error
    }

    return ret;
}


static TYPE_DESC
max_kind(expv v0, TYPE_DESC t0, expv v1, TYPE_DESC t1)
{
    expv kv0 = NULL;
    expv kv1 = NULL;
    double d0 = 0;
    double d1 = 0;

    if (v0 != NULL) {
        kv0 = expv_reduce_kind(v0);
    }
    if (v1 != NULL) {
        kv1 = expv_reduce_kind(v1);
    }

    if (kv0 == NULL || kv1 == NULL) {
        return NULL;
    }

    if (EXPV_CODE(kv0) == INT_CONSTANT) {
        d0 = (double)EXPV_INT_VALUE(kv0);
    } else if (EXPV_CODE(kv0) == FLOAT_CONSTANT) {
        d0 = EXPV_FLOAT_VALUE(kv0);
    }

    if (EXPV_CODE(kv1) == INT_CONSTANT) {
        d1 = (double)EXPV_INT_VALUE(kv1);
    } else if (EXPV_CODE(kv1) == FLOAT_CONSTANT) {
        d1 = EXPV_FLOAT_VALUE(kv1);
    }

    return (d0 >= d1) ? t0 : t1;
}


/**
 * compare numeric type's kind parameter and
 * return maximum type
 */
static TYPE_DESC
max_numeric_kind_type(TYPE_DESC tp1, TYPE_DESC tp2)
{
    assert(TYPE_BASIC_TYPE(tp1) == TYPE_BASIC_TYPE(tp2));
    expv k1 = TYPE_KIND(tp1), k2 = TYPE_KIND(tp2);
    TYPE_DESC rT = NULL;

    if (k1 == NULL && k2 == NULL) {
        /* same kind.*/
        return tp1;
    } else if (k1 != NULL && k2 == NULL) {
        return tp1;
    } else if (k1 == NULL && k2 != NULL) {
        return tp2;
    } else if (k1 == k2 || EXPR_SYM(k1) == EXPR_SYM(k2)) {
        /* Also same kind. */
        return tp1;
    }

    rT = max_kind(k1, tp1, k2, tp2);
    if (rT != NULL) {
        return rT;
    }

    return type_basic(IS_COMPLEX(tp1) ? TYPE_GNUMERIC_ALL : TYPE_GNUMERIC);
}


TYPE_DESC
max_type(TYPE_DESC tp1, TYPE_DESC tp2)
{
    BASIC_DATA_TYPE t;

/* FEAST add start */
    if(!tp1 || !tp2) return NULL;
/* FEAST add  end  */
    t = TYPE_BASIC_TYPE(tp2);
    switch(TYPE_BASIC_TYPE(tp1)) {
    case TYPE_INT:
        switch(t) {
        case TYPE_INT:
            return max_numeric_kind_type(tp1, tp2);
        case TYPE_REAL: case TYPE_DREAL:
        case TYPE_COMPLEX: case TYPE_DCOMPLEX:
            return tp2;
        default:
            return tp1;
        }
    case TYPE_REAL:
        switch(t) {
        case TYPE_REAL:
            return max_numeric_kind_type(tp1, tp2);
        case TYPE_DREAL: case TYPE_COMPLEX: case TYPE_DCOMPLEX:
            return tp2;
        default:
            return tp1;
        }
    case TYPE_DREAL:
        switch(t) {
        case TYPE_DREAL:
            return max_numeric_kind_type(tp1, tp2);
        case TYPE_COMPLEX: case TYPE_DCOMPLEX:
            return tp2;
        default:
            return tp1;
        }
    case TYPE_COMPLEX:
        switch(t) {
        case TYPE_COMPLEX:
            return max_numeric_kind_type(tp1, tp2);
        case TYPE_DCOMPLEX:
            return tp2;
        default:
            return tp1;
        }
    case TYPE_DCOMPLEX:
        return tp1;
    case TYPE_CHAR:
        if(t == TYPE_CHAR)
            return tp1;
        break;
    case TYPE_GNUMERIC:
        switch(t) {
        case TYPE_INT:
        case TYPE_REAL:
        case TYPE_DREAL:
            return tp1;
        default:
            return tp2;
        }
        break;
    case TYPE_GNUMERIC_ALL:
        switch(t) {
        case TYPE_INT:
        case TYPE_REAL:
        case TYPE_DREAL:
        case TYPE_COMPLEX:
        case TYPE_DCOMPLEX:
            return tp1;
        default:
            return tp2;
        }
        break;
    case TYPE_GENERIC:
        return tp1;
    default:
        if(t == TYPE_BASIC_TYPE(tp2)) return(tp2);
        abort();
    }
    return NULL;
}

int
is_variable_shape(expv shape)
{
    expv x;
    list lp;

    FOR_ITEMS_IN_LIST(lp, shape) {
        x = LIST_ITEM(lp);

        if(EXPR_ARG1(x) == NULL ||
           EXPV_CODE(EXPR_ARG1(x)) != INT_CONSTANT)
            return TRUE;

        if(EXPR_ARG2(x) == NULL ||
           EXPV_CODE(EXPR_ARG2(x)) != INT_CONSTANT)
            return TRUE;

        if(EXPR_ARG3(x) != NULL &&
           EXPV_CODE(EXPR_ARG3(x)) != INT_CONSTANT)
            return TRUE;
    }

    return FALSE;
}


static int
is_pure_null_array_spec(expv shape) {
    int ret = FALSE;

    if (shape != NULL) {
        int i;
        expv v;
        for (i = 0; i < 3; i++) {
            v = expr_list_get_n(shape, i);
            if (v != NULL) {
                break;
            }
        }
        ret = (i == 3) ? TRUE : FALSE;
    } else {
        fatal("invalid array-spec. (NULL).");
    }

    return ret;
}


static int
calc_array_spec_size_by_spec(int lower, int upper, int step) {
    int ret = -1;

    if (lower != 0 &&
        upper != 0 &&
        step != 0) {
        ret = (upper - lower + step) / step;
        /* if (ret <= 0) { */
        /*     error("invalid array-spec: %d:%d:%d.", */
        /*           lower, upper, step); */
        /*     ret = -1; */
        /* } */
    }

    return ret;
}


static int
calc_array_spec_size(expv shape) {
    int vals[3];

    vals[0] = 0;
    vals[1] = 0;
    vals[2] = 0;

    if (shape != NULL) {
        int i;
        expv v;

        for (i = 0; i < 3; i++) {
            v = expv_reduce(expr_list_get_n(shape, i), TRUE);
            if (v != NULL && EXPR_CODE(v) == INT_CONSTANT) {
                vals[i] = EXPV_INT_VALUE(v);
            } else if (v == NULL) {
                switch (i) {
                    case 0: {
                        vals[i] = 1;
                        break;
                    }
                    case 1: {
                        break;
                    }
                    case 2: {
                        vals[i] = 1;
                        break;
                    }
                }
            }
        }
    }

    return calc_array_spec_size_by_spec(vals[0], vals[1], vals[2]);
}


static int
score_array_spec_element(expv e, int n) {
    /*
     * n:
     *	0:	lower
     *	1:	upper
     *	2:	step
     */
    int ret = 0;

    if (e != NULL) {
        e = expv_reduce(e, TRUE);
        if (e != NULL) {
            if (EXPR_CODE(e) == INT_CONSTANT) {
                if (n < 2) {
                    /*
                     * give it a high score, if it is lower or upper.
                     */
                    ret += 100;
                } else {
                    ret += 10;
                }
            } else {
                if (n < 2) {
                    ret += 30;
                } else {
                    ret += 3;
                }
            }
        }
    }

    /*
     * The highest score is 210.
     */
    return ret;
}


/**
 * \brief Returns a "score" of the given array-spec. The larger the score
 * the more specifc the shape.
 *
 *	@param shape    An array-spec.
 *
 *	@return A score. The highest score is 1210.
 */
int
score_array_spec(expv aSpec) {
    int ret = 0;

    if (aSpec != NULL) {
        int i = 0;
        expv e;

        if (calc_array_spec_size(aSpec) > 0) {
            /*
             * no matter how the array-spec consists of, this one is
             * highly usable no doubt.
             */
            ret += 1000;
        }
        for (i = 0; i < 3; i++) {
            e = expr_list_get_n(aSpec, i);
            ret += score_array_spec_element(e, i);
        }
        /*
         * The highest score is 1210.
         */
    }

    return ret;
}


/**
 * \brief Combine two given array-specs.
 *
 *	@param l    The left hand array-spec.
 *	@param r    The right hand array-spec.
 *
 *	@return A newly created array-spec or one of the given array-specs.
 */
expv
combine_array_specs(expv l, expv r) {
    expv ret = NULL;

    if (l != NULL && r != NULL) {
        int i;
        int lS, rS;
        expv lE, rE;

        ret = list0(LIST);

        for (i = 0; i < 3; i++) {
            lE = expr_list_get_n(l, i);
            rE = expr_list_get_n(r, i);
            lS = score_array_spec_element(lE, i);
            rS = score_array_spec_element(rE, i);

            if (lS == rS) {
                /*
                 * Use the left side element since the right side
                 * might be worn-out'ed by successive compilation.
                 */
                list_put_last(ret, lE);
            } else {
                list_put_last(ret,
                              (lS > rS) ? lE : rE);
            }
        }
    } else if (l != NULL && r == NULL) {
        ret = l;
    } else {
        ret = r;
    }

    return ret;
}


/**
 * \brief Calculate a size of an array-spec.
 *
 *	@param aSpec      An array-spec expv.
 *	@param idASpec   An array-spec of the variable definition
 *	                  (accept NULL).
 *	@param whichSPtr  If not NULL, a used array-spec is returned.
 *
 *	@return -1 if the size can't be determined statically.
 *	@return A size of the shape.
 */
int
array_spec_size(expv aSpec, expv idASpec, expv *whichSPtr) {
    if (aSpec == NULL) {
        fatal("the array-spec expression is NULL.");
        /* not reached. */
        return -1;
    } else {
        int ret = -1;
        expv which = NULL;

        if (is_pure_null_array_spec(aSpec) == TRUE) {
            /*
             * In this case, the aSpec is so called "assumed shape".
             */
            if (idASpec != NULL) {
                if (is_pure_null_array_spec(idASpec) == TRUE) {
                    /*
                     * Both the array-specs are the assumed shapes.
                     * Use the aShape.
                     */
                    which = aSpec;
                } else {
                    int dimSz = calc_array_spec_size(idASpec);
                    if (dimSz > 0) {
                        /*
                         * The size of aShape can't be determined but
                         * the dimASpec can be. Use dimASpec.
                         */
                        ret = dimSz;
                        which = idASpec;
                    } else {
                        which = aSpec;
                    }
                }
            } else {
                which = aSpec;
            }
        } else {
            /*
             * Use the aSpec no argue.
             */
            ret = calc_array_spec_size(aSpec);
            which = aSpec;
        }

        if (whichSPtr != NULL) {
            *whichSPtr = which;
        }

        return ret;
    }
}


/**
 * \brief Set a type of F_INDEX_RANGE expression.
 *
 *	@param v	an F_INDEX_RANGE expression.
 */
void
set_index_range_type(expv v) {
    if (EXPR_CODE(v) == F_INDEX_RANGE) {
        TYPE_DESC tp = compile_dimensions(type_INT, list1(LIST, v));
        fix_array_dimensions(tp);
        EXPV_TYPE(v) = tp;
    }
}


/**
 * If shapes are not same shape then return NULL .
 * If one shape is not array but scalar, then return the other shape.
 * If one shape is assumed shape and the other not, then return
 * assumed shape.
 * If two shape are same shape or are assumed shape,
 * return shape decided by parameter select.
 * If select = TRUE(1) then return lshape.
 *
 * @param  lshape
 * @param  rshape
 * @param  select
 * @return the larger shape than another one.
 *    <br>returns NULL if two shapes are obiously different.
 */
expv
max_shape(expv lshape, expv rshape, int select)
{
    list l_lp, r_lp;
    int l_upper, l_lower, l_step;
    int r_upper, r_lower, r_step;
    int l_size, r_size;

    if(EXPR_LIST(lshape) == NULL ||
       LIST_ITEM(EXPR_LIST(lshape)) == NULL)
        return rshape;

    if(EXPR_LIST(rshape) == NULL ||
       LIST_ITEM(EXPR_LIST(rshape)) == NULL)
        return lshape;

    if(is_variable_shape(lshape) &&
       is_variable_shape(rshape)) {
        goto ret;
    }

    if (is_variable_shape(lshape)) {
        return lshape;
    }

    if (is_variable_shape(rshape)) {
        return rshape;
    }

    for(l_lp = EXPR_LIST(lshape),
        r_lp = EXPR_LIST(rshape);

        l_lp != NULL && r_lp != NULL ;

        l_lp = LIST_NEXT(l_lp),
        r_lp = LIST_NEXT(r_lp)) {

	l_upper = EXPV_INT_VALUE(EXPR_ARG1(LIST_ITEM(l_lp)));
	l_lower = EXPV_INT_VALUE(EXPR_ARG2(LIST_ITEM(l_lp)));
	l_step = !EXPR_ARG3(LIST_ITEM(l_lp)) ?
	         1 : EXPV_INT_VALUE(EXPR_ARG3(LIST_ITEM(l_lp)));

	r_upper = EXPV_INT_VALUE(EXPR_ARG1(LIST_ITEM(r_lp)));
	r_lower = EXPV_INT_VALUE(EXPR_ARG2(LIST_ITEM(r_lp)));
	r_step = !EXPR_ARG3(LIST_ITEM(r_lp)) ?
	         1 : EXPV_INT_VALUE(EXPR_ARG3(LIST_ITEM(r_lp)));

	l_size = (l_upper - l_lower + l_step) / l_step;
	r_size = (r_upper - r_lower + r_step) / r_step;

	if (l_size != r_size) return NULL;

	// completely wrong.
/*         l_size = EXPV_INT_VALUE(l_upper) - EXPV_INT_VALUE(l_lower) + 1; */
/*         if(l_step != NULL) { */
/*             l_size = (EXPV_INT_VALUE(l_step)); */
/*         } */

/*         r_size = EXPV_INT_VALUE(r_upper) - EXPV_INT_VALUE(r_lower) + 1; */
/*         if(r_step != NULL) { */
/*             r_size = (EXPV_INT_VALUE(r_step)); */
/*         } */

/*         if(l_step != r_step) */
/*             return NULL; */
    }

ret:
    if(select)
        return lshape;
    else
        return rshape;
}


/* type = (LIST basic_type length)
 * decl_list = (LIST (LIST ident dims length codims) ...)
 * dims = (LIST dim ...)
 * dim = expr | (LIST expr expr)
 * attributes = (LIST attribute ...)
 */
void
compile_type_decl(expr typeExpr, TYPE_DESC baseTp,
                  expr decl_list, expr attributes)
{
    expr x, ident, dims, codims, leng, value;
    TYPE_DESC tp;
    TYPE_DESC tp0 = NULL;
    int len;
    list lp;
    ID id;
    SYMBOL resS = NULL;
    expr v;
    int hasDimsInAttr = FALSE;
    int hasPointerAttr = FALSE;
    int hasCodimsInAttr = FALSE;
    int hasTypeParamAttr = FALSE;
    int hasNonTypeParamAttr = FALSE;

    /* Check dimension spec in attribute. */
    if (attributes != NULL) {
        FOR_ITEMS_IN_LIST(lp, attributes) {
            x = LIST_ITEM(lp);
            switch (EXPR_CODE(x)) {
                case  F95_DIMENSION_SPEC:
                    hasDimsInAttr = TRUE;
                    hasNonTypeParamAttr = TRUE;
                    break;
                case F95_POINTER_SPEC:
                    hasPointerAttr = TRUE;
                    hasNonTypeParamAttr = TRUE;
                    break;
                case XMP_CODIMENSION_SPEC:
                    hasCodimsInAttr = TRUE;
                    hasNonTypeParamAttr = TRUE;
                    break;
                case F03_KIND_SPEC:
                case F03_LEN_SPEC:
                    hasTypeParamAttr = TRUE;
                    break;
                default:
                    hasNonTypeParamAttr = TRUE;
                    break;
            }
        }
    }

    if (hasTypeParamAttr) {
        if (hasNonTypeParamAttr) {
            error_at_node(decl_list, "Invalid type attributes");
            return;
        }
    }

    if (!hasTypeParamAttr &&
        CTL_TYPE(ctl_top) == CTL_STRUCT &&
        CURRENT_STATE == IN_TYPE_PARAM_DECL ) {
        CURRENT_STATE = INDCL;
    }

    if (typeExpr != NULL &&
        baseTp != NULL) {
        fatal("%s: invalid args.", __func__);
        /* not reached. */
        return;
    }

    if (typeExpr != NULL) {
        tp0 = compile_type(typeExpr, hasPointerAttr);
        if (tp0 == NULL) {
            error_at_node(typeExpr, "Invalid type spec");
            return;
        }
    } else if (baseTp != NULL) {
        tp0 = baseTp;
    }

    /*
     * Check result var
     */
    if (CURRENT_PROCEDURE != NULL &&
        ctl_top == ctl_base && /* not in BLOCK */
        PROC_RESULTVAR(CURRENT_PROCEDURE) != NULL) {
        expr resX = PROC_RESULTVAR(CURRENT_PROCEDURE);
        resS = EXPR_SYM(resX);
    }

    FOR_ITEMS_IN_LIST(lp, decl_list) {

        x = LIST_ITEM(lp);
        ident  = EXPR_ARG1(x);
        dims   = EXPR_ARG2(x);
        leng   = EXPR_ARG3(x);
        value  = EXPR_ARG4(x);
        codims = EXPR_ARG5(x);

        if (ident == NULL) {
            continue; /* error in parser ? */
        }

        if (EXPR_CODE(ident) != IDENT) {
            fatal("%s: not IDENT", __func__);
        }

        if(CTL_TYPE(ctl_top) == CTL_STRUCT) {
            TYPE_DESC struct_tp = CTL_STRUCT_TYPEDESC(ctl_top);
            if (TYPE_PARENT(struct_tp) &&
                (ID_SYM(TYPE_PARENT(struct_tp)) == EXPR_SYM(ident) ||
                 find_struct_member(TYPE_PARENT_TYPE(struct_tp),
                                    EXPR_SYM(ident)))) {
                error_at_node(typeExpr,
                              "component %s already exists in the parent type.",
                              SYM_NAME(EXPR_SYM(ident)));
            }

            id = find_ident_from_type_parameter(EXPR_SYM(ident), struct_tp);
            if (hasTypeParamAttr) {
                if (id == NULL) {
                    error_at_node(decl_list,
                                  "'%s' is not in the type parameter.",
                                  SYM_NAME(EXPR_SYM(ident)));
                } else if (ID_TYPE(id) != NULL) {
                    error_at_node(decl_list, "'%s' is already declared.",
                                  ID_NAME(id));
                }
            } else {
                if (id != NULL && CURRENT_STATE != IN_TYPE_PARAM_DECL) {
                    error_at_node(decl_list, "'%s' is already used as a type parameter.", ID_NAME(id));
                    continue;
                }
            }

            id = declare_ident(EXPR_SYM(ident), CL_UNKNOWN);
            if (id == NULL) {
                id = find_ident_head(EXPR_SYM(ident), TYPE_MEMBER_LIST(struct_tp));
            }

            if (id == NULL) {
                CURRENT_STATE = INDCL;
                id = declare_ident(EXPR_SYM(ident), CL_UNKNOWN);
            }
        } else {
            id = find_ident_local(EXPR_SYM(ident));
            if(id != NULL && ID_IS_OFMODULE(id)) {
                error_at_node(decl_list, "setting a type to USE-associated symbol '%s'.", ID_NAME(id));
                continue;
            } else if (id != NULL && ID_IS_AMBIGUOUS(id)) {
                error_at_node(decl_list, "an ambiguous reference to symbol '%s'", ID_NAME(id));
                return;
	    } else if(id != NULL && ID_TYPE(id) && TYPE_IS_FOR_FUNC_SELF(ID_TYPE(id))) {
	      ;
            } else if(id == NULL || !(ID_IS_DUMMY_ARG(id))) {
                id = declare_ident(EXPR_SYM(ident), CL_UNKNOWN);
            } else if (id != NULL && ID_IS_DUMMY_ARG(id)) {
                /* update order from one set in declare_dummy_args */
                ID_ORDER(id) = order_sequence++;
            }
            if (ID_TYPE(id) != NULL && TYPE_IS_MODIFIED(ID_TYPE(id))) {
                /* If VOLATILE or ASYNCHRONOUS statements occurred */
                ID_TYPE(id) = NULL;
            }
            TYPE_DESC t = tp0 ? tp0 : ID_TYPE(id);
            if (t && TYPE_LENG(t) && IS_INT_CONST_V(TYPE_LENG(t)) == FALSE)
                ID_ORDER(id) = order_sequence++;
        }

        if (tp0 == NULL) {
            tp = ID_TYPE(id);
            if (tp == NULL) {
                /*
                 * TYPE_DESC for the ident is not declared, compile
                 * this at end_declaration(), so copy all specs into
                 * the ID.
                 */
                VAR_UNCOMPILED_DECL(id) = list2(LIST, x, attributes);
                VAR_IS_UNCOMPILED(id) = TRUE;
                if (hasDimsInAttr == TRUE || dims != NULL){
                    VAR_IS_UNCOMPILED_ARRAY(id) = TRUE;
                }
                ID_CLASS(id) = CL_VAR;
                ID_LINE(id) = EXPR_LINE(decl_list);
                ID_COULD_BE_IMPLICITLY_TYPED(id) = TRUE;
                continue;
            }
        } else {
            tp = tp0;
        }

        if (leng != NULL) {
            if (EXPR_CODE(leng) == LIST) {
                len = CHAR_LEN_UNFIXED;
            } else if (EXPR_CODE(leng) == F95_LEN_SELECTOR_SPEC) {
                if (EXPR_ARG1(leng) == NULL) {
                    len = CHAR_LEN_UNFIXED;
                } else {
                    if ((v = compile_int_constant(EXPR_ARG1(leng))) == NULL) {
                        continue;
                    }
                    goto gotV;
                }
            } else {
                if ((v = compile_int_constant(leng)) == NULL) {
                    continue;     /* error */
                }
                gotV:
                if (EXPV_CODE(v) == INT_CONSTANT) {
                    len = EXPV_INT_VALUE(v);
                } else {
                    len = CHAR_LEN_UNFIXED;
                }
            }
            if (TYPE_BASIC_TYPE(tp) != TYPE_CHAR) {
                error_at_node(decl_list,
                              "length specification to non character");
                continue;       /* error */
            } else {
                tp = type_char(len);
            }
        }

        /*
         * Create new TYPE_DESC, ALWAYS.  Since we need it. Otherwise
         * identifier-origin attribute are corrupted.
         */
        if (!TYPE_IS_CLASS(tp)) {
            tp = wrap_type(tp);
        }

        if (attributes != NULL) {
            int ignoreDimsInAttr = FALSE;
            int ignoreCodimsInAttr = FALSE;
            if (dims != NULL) {
                if (hasDimsInAttr == TRUE) {
                    warning_at_node(decl_list,
                                    "Both the attributes and '%s' have "
                                    "a dimension spec.",
                                    SYM_NAME(EXPR_SYM(ident)));
                }
                ignoreDimsInAttr = TRUE;
            }

            if (codims) {
                if (hasCodimsInAttr){
                    warning_at_node(decl_list,
                                    "Both the attributes and '%s' have "
                                    "a codimension spec.",
                                    SYM_NAME(EXPR_SYM(ident)));
                }
                ignoreCodimsInAttr = TRUE;
            }

            tp = declare_type_attributes(id, tp, attributes,
                                         ignoreDimsInAttr,
                                         ignoreCodimsInAttr);

            if (!tp) return;

        }

        if (dims != NULL) {
            /*
             * Always use dimension spec specified with identifier.
             */
            tp = compile_dimensions(tp, dims);
            /* fix_array_dimensions() is now called in end_declaration() */
            ID_ORDER(id) = order_sequence++;
        }

        if (codims) {
            if (CTL_TYPE(ctl_top) == CTL_STRUCT && !TYPE_IS_ALLOCATABLE(tp)){
                error_at_node(codims, "A coarray component must be allocatable.");
                return;
            }

            /* if (is_descendant_coindexed(tp)){ */
            /*     error_at_node(codims, "The codimension attribute cannnot be nested."); */
            /*     return; */
            /* } */

            codims_desc *codesc = compile_codimensions(codims,
                                                       TYPE_IS_ALLOCATABLE(tp));
            if (codesc) {
                tp->codims = codesc;
            } else {
                error_at_node(codims, "Wrong codimension declaration.");
                return;
            }
        }

	TYPE_DESC tp1 = IS_ARRAY_TYPE(tp) ? bottom_type(tp) : tp;
	if (has_coarray_component(tp1)){
	  if (TYPE_IS_POINTER(tp1) || TYPE_IS_ALLOCATABLE(tp1) || IS_ARRAY_TYPE(tp) ||
	      TYPE_IS_COINDEXED(tp1)){
	    error_at_node(decl_list, "An entity or data component whose type has a coarray"
			  " ultimate component shall be a nonpointer nonallocatable scalar"
			  " and shall not be a coarray.");
	    return;
	  }
	}
	
        if (id != NULL) {
             declare_id_type(id, tp);
             if (!ID_LINE(id) || PROC_CLASS(id) == P_THISPROC) ID_LINE(id) = EXPR_LINE(decl_list);
             if (TYPE_IS_PARAMETER(tp)) {
                 ID_CLASS(id) = CL_PARAM;
             }
             else if (TYPE_IS_INTRINSIC(tp)){
                 ID_CLASS(id) = CL_PROC;
                 PROC_CLASS(id) = P_INTRINSIC;
             }
             if (TYPE_IS_EXTERNAL(ID_TYPE(id)) &&
                 (!IS_PROCEDURE_TYPE(ID_TYPE(id)) || IS_PROCEDURE_POINTER(ID_TYPE(id)))) {
                 ID_TYPE(id) = function_type(ID_TYPE(id));
                 ID_CLASS(id) = CL_PROC;
                 PROC_CLASS(id) = P_EXTERNAL;
                 tp = ID_TYPE(id);
             }

        }

        if (value != NULL && EXPR_CODE(value) != F_DATA_DECL) {

            /*
             * to be checked for function results.
             */
            if (ID_IS_DUMMY_ARG(id) || ID_CLASS(id) == CL_PROC ||
                TYPE_IS_ALLOCATABLE(tp) ||
                TYPE_IS_EXTERNAL(tp) || TYPE_IS_INTRINSIC(tp) || ID_SYM(id) == resS) {
                error_at_node(decl_list, "%s cannot have an initializer.", ID_NAME(id));
                return;
            }

            /*
             * FIXME:
             *	SUPER BOGUS FLAG ALERT !
             */
            is_in_struct_member_initializer_compilation_flag_for_declare_ident
                = TRUE;
            VAR_INIT_VALUE(id) = compile_expression(value);
            ID_ORDER(id) = order_sequence++;
            is_in_struct_member_initializer_compilation_flag_for_declare_ident
                = FALSE;
        }

        if (TYPE_IS_PARAMETER(tp)) {
           /* handle the attribute for parameter, calc its value.
             ident
             value
             => list1(LIST, lis1(LIST, list2(LIST, ident, value)))
           */
            expr const_list = list0(LIST);
            const_list
               = list_put_last(const_list, list2(LIST, ident, value));
            compile_PARAM_decl(const_list);
        }

    } /* end FOR_ITEMS_IN_LIST */
}

TYPE_DESC
find_struct_decl_block_parent(SYMBOL s)
{
    CTL cp;
    TYPE_DESC tp = NULL;
    int in_block = FALSE;

    FOR_CTLS_BACKWARD(cp) {
        if (CTL_TYPE(cp) == CTL_BLK ||
            CTL_TYPE(cp) == CTL_FORALL ||
            CTL_TYPE(cp) == CTL_ASSOCIATE || \
            CTL_TYPE(cp) == CTL_DO || \
            CTL_TYPE(cp) == CTL_TYPE_GUARD) {
            in_block = TRUE;
            if (CTL_STRUCT_DECLS(cp) == LOCAL_STRUCT_DECLS) {
                continue;
            }
            tp = find_struct_decl_head(s, CTL_STRUCT_DECLS(cp));
            if (tp != NULL) {
                return tp;
            }
        }
    }

    if (in_block) {
        tp = find_struct_decl_head(s,
            UNIT_CTL_LOCAL_STRUCT_DECLS(CURRENT_UNIT_CTL));
    }

    return tp;
}


TYPE_DESC
find_struct_decl_parent(SYMBOL s)
{
    int lev_idx;
    TYPE_DESC tp;

    for (lev_idx = unit_ctl_level - 1; lev_idx >= 0; --lev_idx) {
        tp = find_struct_decl_head(s,
                UNIT_CTL_LOCAL_STRUCT_DECLS(unit_ctls[lev_idx]));
        if (tp != NULL) {
            return tp;
        }
    }
    return NULL;
}

TYPE_DESC
find_struct_decl_sibling(SYMBOL s)
{
    TYPE_DESC tp;
    EXT_ID ep;

    if (unit_ctl_level < 1) {
        return NULL;
    }
    FOREACH_EXT_ID(ep, PARENT_CONTAINS) {
        tp = find_struct_decl_head(s, EXT_PROC_STRUCT_DECLS(ep));
        if (tp != NULL) {
            return tp;
        }
    }
    return NULL;
}

TYPE_DESC
find_struct_decl(SYMBOL s)
{
    TYPE_DESC tp;

    tp = find_struct_decl_head(s, LOCAL_STRUCT_DECLS);
    if (tp != NULL) {
        return tp;
    }
    if (in_interface() && !in_module_procedure() && !has_import_all()) {
        return NULL;
    }
    tp = find_struct_decl_block_parent(s);
    if (tp != NULL) {
        return tp;
    }
    tp = find_struct_decl_parent(s);
    if (tp != NULL) {
        return tp;
    }
    tp = find_struct_decl_sibling(s);
    return tp;
}

/* compile type statement */
void
compile_struct_decl(expr ident, expr type, expr type_params)
{
    TYPE_DESC tp;
    expv v;
    list lp = NULL;
    int has_access_spec = FALSE;
    int has_extends_spec = FALSE;
    int has_abstract_spec = FALSE;
    int has_bind_spec = FALSE;
    ID last = NULL;

    if(ident == NULL)
        fatal("compile_struct_decl: F95 derived type name is NULL");
    if(EXPR_CODE(ident) != IDENT)
        fatal("compile_struct_decl: not IDENT");
    tp = declare_struct_type(ident);
    //TYPE_IS_DECLARED(tp) = TRUE;
    v = list0(F95_TYPEDECL_STATEMENT);
    EXPV_TYPE(v) = tp;

    FOR_ITEMS_IN_LIST(lp, type) {
        expr x = LIST_ITEM(lp);
        if (x == NULL) {
            continue;
        }

        switch(EXPR_CODE(x)) {
            case F95_PUBLIC_SPEC:
            case F95_PRIVATE_SPEC:
            case F03_PROTECTED_SPEC:
                if (has_access_spec) {
                    error("unexpected access spec");
                } else {
                    has_access_spec = TRUE;
                }
                break;
            case F03_EXTENDS_SPEC:
                if (has_extends_spec) {
                    error("Dupulicate EXTENDS spec");
                } else {
                    has_extends_spec = TRUE;
                }
                break;
            case F03_ABSTRACT_SPEC:
                if (has_abstract_spec) {
                    error("Dupulicate ABSTRACT spec");
                } else {
                    has_abstract_spec = TRUE;
                }
                break;
            case F03_BIND_SPEC:
                if (has_bind_spec) {
                    error("Dupulicate BIND spec");
                } else {
                    has_bind_spec = TRUE;
                }
                break;
            default:
                break;
        }

        if (has_abstract_spec && has_bind_spec) {
            error("non-extensible derived type should not be ABSTRACT");
        }

        switch(EXPR_CODE(x)) {
            case F95_PUBLIC_SPEC:
                TYPE_SET_PUBLIC(tp);
                TYPE_UNSET_PRIVATE(tp);
                TYPE_UNSET_PROTECTED(tp);
                break;
            case F95_PRIVATE_SPEC:
                TYPE_SET_PRIVATE(tp);
                TYPE_UNSET_PUBLIC(tp);
                TYPE_UNSET_PROTECTED(tp);
                break;
            case F03_PROTECTED_SPEC:
                TYPE_SET_PROTECTED(tp);
                TYPE_UNSET_PRIVATE(tp);
                TYPE_UNSET_PUBLIC(tp);
                break;
            case F03_EXTENDS_SPEC: {
                TYPE_DESC parent_type;
                parent_type = find_struct_decl(EXPR_SYM(EXPR_ARG1(x)));
                if (parent_type == NULL) {
                    error("derived-type %s does not exist",
                          SYM_NAME(EXPR_SYM(EXPR_ARG1(x))));
                    break;
                }
                if (TYPE_IS_SEQUENCE(parent_type)) {
                    error("derived-type %s is not an extensible type",
                          SYM_NAME(EXPR_SYM(EXPR_ARG1(x))));
                    break;
                }
                TYPE_PARENT(tp) = new_ident_desc(EXPR_SYM(EXPR_ARG1(x)));
                TYPE_PARENT_TYPE(tp) = parent_type;
            }; break;
            case F03_ABSTRACT_SPEC:
                TYPE_SET_ABSTRACT(tp);
                break;
            case F03_BIND_SPEC:
                TYPE_SET_BIND(tp);
                break;
            default:
                break;
        }
    }


    push_ctl(CTL_STRUCT);
    CTL_BLOCK(ctl_top) = v;
    FOR_ITEMS_IN_LIST(lp, type_params) {
        ID id = declare_ident(EXPR_SYM(LIST_ITEM(lp)), CL_TYPE_PARAM);
        ID_LINK_ADD(id, TYPE_TYPE_PARAMS(tp), last);
    }
}


static void
check_type_parameters_declared(CTL ctl)
{
    ID ip;
    if (CTL_TYPE(ctl_top) != CTL_STRUCT)
        return;

    FOREACH_ID(ip, TYPE_TYPE_PARAMS(CTL_STRUCT_TYPEDESC(ctl))) {
        if (ID_TYPE(ip) == NULL ||
            !(TYPE_IS_KIND(ID_TYPE(ip)) || TYPE_IS_LEN(ID_TYPE(ip)))) {
            error("Type parameter '%s' is not declared", ID_NAME(ip));
        }
    }
}

/*
 * Check the bindings of the type-bound generic appear before END TYPE statement.
 */
static void
check_type_bound_generics(TYPE_DESC stp)
{
    ID mem;
    ID binding;
    ID bindto;

    FOREACH_TYPE_BOUND_GENERIC(mem, stp) {
        FOREACH_ID(binding, TBP_BINDING(mem)) {
            bindto = find_struct_member(stp, ID_SYM(binding));
            if (bindto == NULL ||
                (ID_CLASS(mem) == CL_TYPE_BOUND_PROC &&
                 !(TBP_BINDING_ATTRS(mem) & TYPE_BOUND_PROCEDURE_IS_GENERIC))) {
                error("generic should bound to type-bound procedures");
            }
        }
    }
}

/*
 * Check the bindings of the type-bound generic appear before END TYPE statement.
 */
static void
check_no_abstract_type_bound_procedure(const TYPE_DESC stp)
{
    ID mem;
    ID p;
    TYPE_DESC tp;

    if (stp == NULL) {
        return;
    }

    for (tp = stp; tp != NULL; tp = TYPE_PARENT(tp)?TYPE_PARENT_TYPE(tp):NULL) {
        if (TYPE_IS_ABSTRACT(tp)) {
            FOREACH_TYPE_BOUND_PROCEDURE(mem, tp) {
                if (ID_CLASS(mem) == CL_TYPE_BOUND_PROC && TBP_IS_DEFERRED(mem)) {
                    p = find_struct_member_allow_private(stp, ID_SYM(mem), TRUE);
                    if (TBP_IS_DEFERRED(p)) {
                        error("%s is not implemented", SYM_NAME(ID_SYM(mem)));
                    }
                }
            }
        }
    }
}

/* compile end type statement */
void
compile_struct_decl_end()
{
    TYPE_DESC stp;

    if(CTL_TYPE(ctl_top) != CTL_STRUCT) {
        error("illegal derived type declaration end");
        return;
    }

    check_type_parameters_declared(ctl_top);

    if (endlineno_flag)
      EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;

    stp = EXPV_TYPE(CTL_BLOCK(ctl_top));

    TYPE_IS_DECLARED(stp) = TRUE;

    /* check for type bound generic */
    check_type_bound_generics(stp);

    if (!TYPE_IS_ABSTRACT(stp)) {
        check_no_abstract_type_bound_procedure(stp);
    }

    if (TYPE_IS_SEQUENCE(stp) && TYPE_IS_ABSTRACT(stp)) {
        error("non-extensible derived type should not be ABSTRACT");
    }

    if (!TYPE_IS_INTERNAL_PRIVATE(stp)) {
        /*
         * To transform the fortran code.
         *
         * TYPE t
         *  INTEGER, PRIVATE :: a
         *  INTEGER :: b
         * END TYPE t
         *
         * will turn into
         *
         * TYPE t
         *  PRIVATE
         *  INTEGER :: a
         *  INTEGER, PUBLIC :: b
         * END TYPE t
         *
         */

        ID mem;
        int has_private_member = FALSE;

        FOREACH_MEMBER(mem, stp) {
            if (TYPE_IS_PRIVATE(ID_TYPE(mem))) {
                has_private_member = TRUE;
                break;
            }
        }

        if (has_private_member) {
            /*
             * If the derived-type has a PRIVATE member,
             * set is_internal_private to the derived-type.
             */
            TYPE_SET_INTERNAL_PRIVATE(stp);

            /*
             * Remove PRIVATE from PRIVATE members,
             * and add PUBLIC to other members.
             */
            FOREACH_MEMBER(mem, stp) {
                if (TYPE_IS_PRIVATE(ID_TYPE(mem))) {
                    TYPE_UNSET_PRIVATE(ID_TYPE(mem));
                } else {
                    TYPE_SET_PUBLIC(ID_TYPE(mem));
                }
            }
        }
    }

    pop_ctl();
}


/* compile sequence statement */
void
compile_SEQUENCE_statement()
{
    if(CTL_TYPE(ctl_top) != CTL_STRUCT) {
        error("unexpected sequence statement");
        return;
    }

    TYPE_SET_SEQUENCE(CTL_STRUCT_TYPEDESC(ctl_top));
}

static void
reduce_subscript(expv *pv)
{
    if(*pv && expr_is_constant_typeof(*pv, TYPE_INT)) {
        expv tmp = expr_constant_value(*pv);
        if(tmp)
            *pv = tmp;
    }
}

/* return tp if dims is NULL */
/* create dimensions block for array variable. */
TYPE_DESC
compile_dimensions(TYPE_DESC tp, expr dims)
{
    expr x;
    TYPE_DESC tq = NULL;
    int n;
    list lp;

    n = 0;
    FOR_ITEMS_IN_LIST(lp, dims) {
        expr lower = NULL, upper = NULL, step = NULL;

        x = LIST_ITEM(lp);
        if (x == NULL) {
            if (LIST_NEXT(lp) != NULL) {
                error("only last bound may be asterisk");
            }
        } else if (EXPR_CODE(x) == LIST) { /* (LIST lower upper) */
            lower = EXPR_ARG1(x);
            upper = EXPR_ARG2(x);
            /* step comes from compile_array_ref() */
            step = expr_list_get_n(x, 2);
        } else if (EXPR_CODE(x) == F_INDEX_RANGE) {
            lower = EXPR_ARG1(x);
            upper = EXPR_ARG2(x);
            step  = EXPR_ARG3(x);
        } else {
            upper = x;
        }
        n++;
        if (n > MAX_DIM) {
            error("no more than MAX_DIM(%d) dimensions",MAX_DIM);
            break;
        }

        tq = new_type_desc();
        TYPE_BASIC_TYPE(tq) = TYPE_ARRAY;

        if (tp != NULL) {
            TYPE_ATTR_FLAGS(tq) |= TYPE_IS_POINTER(tp);
            TYPE_ATTR_FLAGS(tq) |= TYPE_IS_TARGET(tp);
            TYPE_ATTR_FLAGS(tq) |= TYPE_IS_ALLOCATABLE(tp);
        }

        reduce_subscript(&lower);
        TYPE_DIM_LOWER(tq) = lower;

        reduce_subscript(&upper);
        TYPE_DIM_UPPER(tq) = upper;

        reduce_subscript(&step);
        TYPE_DIM_STEP(tq) = step;

        TYPE_N_DIM(tq) = n;
        TYPE_DIM_FIXED(tq) = 0; /* immature */

        TYPE_REF(tq) = tp;
        tp = tq;
    }

    return tp;
}


/* create codims_desc. */
codims_desc*
compile_codimensions(expr dims, int is_alloc){

  codims_desc *codims;
  expr x;
  int n = 0;
  list lp;

  assert(!dims);

  codims = XMALLOC(codims_desc*, sizeof(codims_desc));
  codims->cobound_list = list0(LIST);

  FOR_ITEMS_IN_LIST(lp, dims) {

    expr lower = NULL, upper = NULL, step = NULL;

    x = LIST_ITEM(lp);
    if (x == NULL) {
      if(LIST_NEXT(lp) != NULL)
          error("only last cobound may be \"*\"");
    }
    else if (EXPR_CODE(x) == LIST){ /* (LIST lower upper) */
      lower = compile_expression(EXPR_ARG1(x));
      upper = compile_expression(EXPR_ARG2(x));
      /* step comes from compile_array_ref() */
      step = compile_expression(expr_list_get_n(x, 2));
    }
/*     else if (EXPR_CODE(x) == F_INDEX_RANGE) { */
/*       lower = EXPR_ARG1(x); */
/*       upper = EXPR_ARG2(x); */
/*       step  = EXPR_ARG3(x); */
/*     } */
    else {
      upper = compile_expression(x);
    }

    if (is_alloc && (lower || upper || step)){
      error("An allocatable coarray must have a deferred coshape.");
      return NULL;
    }

    if (!lower && !upper && !is_alloc){
      error("deferred coshape can be specified only for ALLOCATABLE coarrays.");
      return NULL;
    }

    if (LIST_NEXT(lp) && upper && EXPV_CODE(upper) == F_ASTERISK){
      error("Only last upper-cobound can be \"*\".");
      return NULL;
    }

    if (!LIST_NEXT(lp)){
      if (!upper){
          ;
      }
      else if (EXPV_CODE(upper) == F_ASTERISK){
          //upper = NULL;
      }
      else {
          error("Last upper-cobound must be \"*\".");
          return NULL;
      }
    }

    if (!lower && upper && EXPV_CODE(upper) != F_ASTERISK)
      lower = expv_constant_1;

    reduce_subscript(&lower);
    reduce_subscript(&upper);
    reduce_subscript(&step);
    codims->cobound_list = list_put_last(codims->cobound_list,
                                         list3(F95_TRIPLET_EXPR, lower, upper, step));

    n++;

    if (n > MAX_DIM){
      error("no more than MAX_DIM(%d) dimensions", MAX_DIM);
      return NULL;
    }
  }

  codims->corank = n;

  return codims;
}

void
fix_array_dimensions(TYPE_DESC tp)
{
    ARRAY_ASSUME_KIND assumeKind = ASSUMED_NONE;
    expv size = NULL, upper = NULL, lower = NULL, step = NULL;

    if(tp == NULL) return;
    if(IS_ARRAY_TYPE(tp) == FALSE) return;
    fix_array_dimensions(TYPE_REF(tp));

    if(TYPE_DIM_FIXING(tp)){
        error("cyclic reference in adjustable array size");
        return;
    }
    if(TYPE_DIM_FIXED(tp)) return;

    TYPE_DIM_FIXING(tp) = TRUE; /* fixing */

    if(TYPE_DIM_LOWER(tp))
        lower = expv_reduce(compile_expression(TYPE_DIM_LOWER(tp)), FALSE);

    step = TYPE_DIM_STEP(tp) ?
      expv_reduce(compile_expression(TYPE_DIM_STEP(tp)), FALSE) : expv_constant_1;

    if(TYPE_DIM_UPPER(tp) == NULL) {
        /* n(lower:) */
        assumeKind = ASSUMED_SHAPE;
    } else {
        if(EXPR_CODE(TYPE_DIM_UPPER(tp)) == F_ASTERISK) {
            /* n(lower:*), n(*) */
            assumeKind = ASSUMED_SIZE;
        } else {
            upper = expv_reduce(compile_expression(TYPE_DIM_UPPER(tp)), FALSE);
            if (upper == NULL) {
              error ("internal compiler error on fix_array_dimensions(): assertion fail on upper as NULL.");
              exit (1);
            }

            if(lower == NULL)
                lower = expv_constant_1;

/* 	    if(EXPV_CODE(upper) == INT_CONSTANT && */
/* 	       EXPV_CODE(lower) == INT_CONSTANT) { */
/* 	      int s = EXPV_INT_VALUE(upper) - EXPV_INT_VALUE(lower) + 1; */
/* 	      if(s < 0) */
/* 		error_at_node(TYPE_DIM_UPPER(tp), */
/* 			      "upper bound must be larger than lower bound"); */
/* 	      size = expv_int_term(INT_CONSTANT, type_INT, s); */
/* 	    } else { */
/* 	      if(lower == expv_constant_1) size = upper; */
/* 	      else size = expv_cons(PLUS_EXPR,type_INT, */
/* 				    expv_cons(MINUS_EXPR,type_INT,upper,lower), */
/* 				    expv_constant_1); */
/* 	    } */

	    if (EXPV_CODE(upper) == INT_CONSTANT &&
		EXPV_CODE(lower) == INT_CONSTANT &&
		EXPV_CODE(step) == INT_CONSTANT){
	      int cs = EXPV_INT_VALUE(step);
	      if (cs != 0){
		int s = (EXPV_INT_VALUE(upper) - EXPV_INT_VALUE(lower) + cs) / cs;
		if (s < 0)
		  /* // Fix the error message ' "??", line ??: ' */
		  /* error_at_node(TYPE_DIM_UPPER(tp), */
		  /* 		"upper bound must be larger than lower bound when step > 0 and vice versa."); */
		  size = expv_int_term(INT_CONSTANT, type_INT, 0);
		else
		  size = expv_int_term(INT_CONSTANT, type_INT, s);
	      }
	      else
		// Fix the error message ' "??", line ??: '
		//error_at_node(TYPE_DIM_STEP(tp), "step must not be zero");
		error("step must not be zero");
	    }
	    else
	      size = expv_cons(DIV_EXPR, type_INT,
			       expv_cons(PLUS_EXPR, type_INT,
					 expv_cons(MINUS_EXPR, type_INT,
						   upper, lower),
					 step),
			       step);
        }
    }

    TYPE_ARRAY_ASSUME_KIND(tp) = assumeKind;
    TYPE_DIM_SIZE(tp) = size;
    TYPE_DIM_LOWER(tp) = lower;
    TYPE_DIM_UPPER(tp) = upper;
    TYPE_DIM_FIXED(tp) = TRUE;     /* fix it */
    TYPE_DIM_FIXING(tp) = FALSE;

    /* merge parent's and child's attributes */
    if (TYPE_REF(tp) != NULL) {
        merge_attributes(tp, TYPE_REF(tp));
    }
}


TYPE_DESC
copy_array_type(TYPE_DESC tp)
{
    return copy_dimension(tp, NULL);
}

TYPE_DESC
copy_dimension(TYPE_DESC array, TYPE_DESC base)
{
    TYPE_DESC tp1 = NULL;
    TYPE_DESC tp = tp1;

    assert(IS_ARRAY_TYPE(array));
    assert(base == NULL || !(IS_ARRAY_TYPE(base)));

    while(IS_ARRAY_TYPE(array)) {
        if(tp1) {
            TYPE_REF(tp1) = new_type_desc();
            tp1 = TYPE_REF(tp1);
        } else {
            tp1 = new_type_desc();
            tp = tp1;
        }
        TYPE_BASIC_TYPE(tp1)        = TYPE_ARRAY;
        TYPE_ARRAY_ASSUME_KIND(tp1) = TYPE_ARRAY_ASSUME_KIND(array);
        TYPE_N_DIM(tp1)             = TYPE_N_DIM(array);
        TYPE_DIM_SIZE(tp1)          = TYPE_DIM_SIZE(array);
        TYPE_DIM_LOWER(tp1)         = TYPE_DIM_LOWER(array);
        TYPE_DIM_UPPER(tp1)         = TYPE_DIM_UPPER(array);
        TYPE_DIM_STEP(tp1)          = TYPE_DIM_STEP(array);

        array = TYPE_REF(array);
    }

    if(base == NULL)
        TYPE_REF(tp1) = array;
    else
        TYPE_REF(tp1) = base;

    fix_array_dimensions(tp);

    return tp;
}

void
compile_PARAM_decl(expr const_list)
{
    expr x,ident;
    ID id;
    list lp;

    if (const_list == NULL) {
        return; /* error */
    }

    FOR_ITEMS_IN_LIST(lp, const_list) {
        x = LIST_ITEM(lp);
        if (x == NULL) {
            continue; /* error */
        }
        ident = EXPR_ARG1(x);
        if (EXPR_CODE(ident) != IDENT) {
            fatal("compile_PARAM_decl: no IDENT");
            return;
        }

        id = find_ident_local(EXPR_SYM(ident));
        if (id == NULL) {
            id = declare_ident(EXPR_SYM(ident), CL_PARAM);
            if (id == NULL) {
                continue;
            }
        } else if (ID_IS_OFMODULE(id)) {
            error_at_id(id, "\"%s\" is already declared in module \"%s\".",
                ID_NAME(id), SYM_NAME(ID_USEASSOC_INFO(id)->module_name));
        }
        if (ID_CLASS(id) == CL_VAR || ID_CLASS(id) == CL_UNKNOWN) {
            ID_CLASS(id) = CL_PARAM;
        }
        if (ID_TYPE(id) != NULL) {
            TYPE_SET_PARAMETER(ID_TYPE(id));
        } else {
            TYPE_SET_PARAMETER(id);
        }
        ID_COULD_BE_IMPLICITLY_TYPED(id) = TRUE;

        postproc_PARAM_decl(ident, EXPR_ARG2(x));

        ID_ORDER(id) = order_sequence++;
    }
}

void
postproc_PARAM_decl(expr ident, expr e)
{
    expv v;
    ID id;

    id = find_ident(EXPR_SYM(ident));

    if(e == NULL) {
        error("parameter value not specified");
        return;
    }

    v = compile_expression(e);

    if(v == NULL) return; // error

    if (expr_is_constant(e)) {
        if(v)
            v = expv_reduce(v, FALSE);
        if (v == NULL) {
            error("bad constant expression in PARAMETER statement");
            return;
        }
    } 

    if (ID_TYPE(id) != NULL) {
        if (type_is_compatible_for_assignment(ID_TYPE(id),
                                              EXPV_TYPE(v)) == FALSE) {
            error("incompatible constant expression in PARAMETER "
                  "statement");
            return;
        }
    }

    VAR_INIT_VALUE(id) = v;
}

void
compile_COMMON_decl(expr com_list)
{
    expr x, ident, dims;
    ID cid = NULL, id;
    list lp;
    TYPE_DESC tp;

    FOR_ITEMS_IN_LIST(lp, com_list) {
        x = LIST_ITEM(lp);

        if(x == NULL) {
            /* blank common name */
            if(blank_common_symbol == NULL) {
                blank_common_symbol = (SYMBOL)malloc(sizeof(*blank_common_symbol));
                bzero(blank_common_symbol, sizeof(*blank_common_symbol));
                SYM_NAME(blank_common_symbol) = BLANK_COMMON_NAME;
            }

            cid = declare_common_ident(blank_common_symbol);
            COM_IS_BLANK_NAME(cid) = TRUE;
        } else if(EXPR_CODE(x) == IDENT) {
            /* common name */
            cid = declare_common_ident(EXPR_SYM(x));
        } else {
            /* common variables */
            assert(cid);
            if (EXPR_CODE(x) != LIST) fatal("compile_COMMON_decl: not list");
            ident = EXPR_ARG1(x);
            dims = EXPR_ARG2(x);
            if (ident == NULL) continue;
            if (EXPR_CODE(ident) != IDENT) fatal("compile_COMMON_decl: not ident");
            id = declare_ident(EXPR_SYM(ident), CL_VAR);
            if(id == NULL) continue;
            ID_COULD_BE_IMPLICITLY_TYPED(id) = TRUE;
            tp = ID_TYPE(id);
            if (dims != NULL) {
                tp = compile_dimensions(tp,dims);
                fix_array_dimensions(tp);
                ID_ORDER(id) = order_sequence++;
            }
            declare_id_type(id,tp);

            switch(ID_STORAGE(id)) {
            case STG_EQUIV:
                ID_STORAGE(id) = STG_COMEQ;
                break;
            case STG_COMEQ:
            case STG_AUTO:
            case STG_SAVE:
            case STG_UNKNOWN:
                ID_STORAGE(id) = STG_COMMON;
                break;
            default:
                error("incompatible common declaration, %s '%s'",
                    ID_NAME(id), storage_class_name(ID_STORAGE(id)));
                continue;
            }

            VAR_COM_ID(id) = cid;
            if(COM_VARS(cid) == NULL)
                COM_VARS(cid) = list0(LIST);
            list_put_last(COM_VARS(cid),
                expv_sym_term(IDENT, tp, ID_SYM(id)));
        }
    }
}


/* declare external function */
void
compile_EXTERNAL_decl(expr id_list)
{
    list lp;
    expr ident;
    ID id;

    if (id_list == NULL) {
        return; /* error */
    }
    FOR_ITEMS_IN_LIST(lp, id_list) {
        ident = LIST_ITEM(lp);
        if (ident == NULL) {
            break;
        }
        if (EXPR_CODE(ident) != IDENT) {
            fatal("compile_EXTERNAL_decl:not ident");
        }
        /*
         * Little bit dirty trick.
         * EXTERAL id can have POINTER attribute, but
         * POINTER statement declares id as variable.
         * So if POINTER statement exists, declare(..., CL_RPOC) cause error.
         */
        if ((id = declare_ident(EXPR_SYM(ident), CL_VAR)) == NULL) {
            continue;
        }
        if (ID_CLASS(id) == CL_VAR) {
            ID_CLASS(id) = CL_PROC;
        }
        if (PROC_CLASS(id) == P_UNKNOWN) {
            PROC_CLASS(id) = P_EXTERNAL;
            TYPE_SET_EXTERNAL(id);
        } else if (PROC_CLASS(id) != P_EXTERNAL) {
            error_at_node(id_list,
                          "invalid external declaration, %s", ID_NAME(id));
            continue;
        }

        if(!(ID_IS_DUMMY_ARG(id)))
            ID_STORAGE(id) = STG_EXT;
    }
}

/* declare intrinsic function */
void
compile_INTRINSIC_decl(expr id_list)
{
    list lp;
    expr ident;
    ID id;
    if(id_list == NULL) return; /* error */
    FOR_ITEMS_IN_LIST(lp,id_list){
        ident = LIST_ITEM(lp);
        if (ident == NULL) {
            break;
        }

        if (EXPR_CODE(ident) != IDENT) {
            fatal("compile_INTRINSIC_decl:not ident");
        }
        if ((id = declare_ident(EXPR_SYM(ident),CL_PROC)) == NULL) {
            /* warning("cannot declare intrinsic"); */
            continue;
        }

        if (PROC_CLASS(id) == P_UNKNOWN) {
            PROC_CLASS(id) = P_INTRINSIC;
        } else if(PROC_CLASS(id) != P_INTRINSIC) {
            error("invalid intrinsic declaration, %s", ID_NAME(id));
        }

        TYPE_SET_INTRINSIC(id);

        if (ID_TYPE(id) == NULL) {
            ID_TYPE(id) = wrap_type(BASIC_TYPE_DESC(TYPE_GNUMERIC_ALL));
        }

        ID_TYPE(id) = intrinsic_function_type(ID_TYPE(id));
        FUNCTION_TYPE_SET_VISIBLE_INTRINSIC(ID_TYPE(id));
        ID_LINE(id) = EXPR_LINE(id_list);
    }
}


static int
markAsSave(id)
     ID id;
{
    if (ID_CLASS(id) == CL_PARAM) {
        return TRUE;
    }
    if ((ID_CLASS(id) != CL_VAR &&
         ID_CLASS(id) != CL_COMMON &&
         ID_CLASS(id) != CL_UNKNOWN) ||
        ID_IS_DUMMY_ARG(id)) {

        error("\"%s\" is not a variable.", SYM_NAME(ID_SYM(id)));
        return FALSE;
    }

    if (ID_CLASS(id) == CL_COMMON) {
        COM_IS_SAVE(id) = TRUE;
    } else {
        if (!TYPE_IS_PARAMETER(id)) {
            TYPE_SET_SAVE(id);
        }
    }

    return TRUE;
}


/* declare save variable */
void
compile_SAVE_decl(id_list)
    expr id_list;
{
    list lp;
    expr ident;
    ID id = NULL;

    if (id_list == NULL) {
        /*
         * special care. must save ALL variable in this scope.
         */

        current_proc_state = P_SAVE;

        /* local symbol. */
        FOREACH_ID(id, LOCAL_SYMBOLS) {
            if (!ID_IS_OFMODULE(id) &&
                ID_CLASS(id) != CL_PARAM &&
                !TYPE_IS_PARAMETER(id) &&
                (ID_CLASS(id) == CL_UNKNOWN ||
                 ID_CLASS(id) == CL_VAR) &&
                !(ID_IS_DUMMY_ARG(id))) {
                markAsSave(id);
            }
        }

        /* common */
        FOREACH_ID(id, LOCAL_COMMON_SYMBOLS) {
            if (COM_VARS(id) &&
                ID_CLASS(id) != CL_PARAM && !TYPE_IS_PARAMETER(id)) {
                markAsSave(id);
            }
        }
        return;
    }

    FOR_ITEMS_IN_LIST(lp, id_list) {
        ident = LIST_ITEM(lp);
        switch (EXPR_CODE(ident)) {
            case IDENT: {
                if ((id = find_ident_local(EXPR_SYM(ident))) == NULL) {
                    id = declare_ident(EXPR_SYM(ident), CL_UNKNOWN);
                    if (id == NULL) {
                        /* must not happen. */
                        continue;
                    }
                    ID_COULD_BE_IMPLICITLY_TYPED(id) = TRUE;
                }
                break;
            }
            case LIST: {
                /* COMMON name */
                SYMBOL comSym = EXPR_SYM(EXPR_ARG1(ident));
                id = find_common_ident(comSym);
                if (id == NULL) {
                    error("common block \"%s\" is not declared.", SYM_NAME(comSym));
                    continue;
                }
                /*
                 * After this closure is parsed, mark variables in this common as saved.
                 */
                break;
            }
            default: {
                fatal("illegal item(s) in save statement.");
                break;
            }
        }
        if(ID_IS_OFMODULE(id)) {
            error("can't change attributes of USE-associated symbol '%s'", ID_NAME(id));
            return;
        } else if (ID_IS_AMBIGUOUS(id)) {
            error("an ambiguous reference to symbol '%s'", ID_NAME(id));
            return;
        }
        (void)markAsSave(id);
    }
}


void
compile_pragma_statement(expr x)
{
  expv v = NULL;

  switch (EXPR_CODE(EXPR_ARG1(x)))
    {
    case STRING_CONSTANT:
	if (EXPR_STR(EXPR_ARG1(x)) == NULL) {
	    error("assertion fail on compile_pragma_statement, arg = NULL");
	    break;
	}
	else {
	  v = expv_str_term(STRING_CONSTANT,
			    NULL,
			    strdup(EXPR_STR(EXPR_ARG1(x))));
	  EXPV_LINE(v) = EXPR_LINE(x);
	  break;
	}
    default:
      {
        error("invalid format.");
        break;
      }
    }
  expv new_st = list1(F_PRAGMA_STATEMENT, v);
  EXPV_LINE(new_st) = EXPR_LINE(x);
  output_statement(new_st);
}


void
compile_pragma_decl(expr x)
{
  expv v = NULL;

  switch (EXPR_CODE(EXPR_ARG1(x)))
    {
    case STRING_CONSTANT:
	if (EXPR_STR(EXPR_ARG1(x)) == NULL) {
	    error("assertion fail on compile_pragma_statement, arg = NULL");
	    break;
	}
	else {
	  v = expv_str_term(STRING_CONSTANT,
			    NULL,
			    strdup(EXPR_STR(EXPR_ARG1(x))));
	  EXPV_LINE(v) = EXPR_LINE(x);
        break;
      }
    default:
      {
        error("invalid format.");
        break;
      }
    }

  SYMBOL sym = gen_temp_symbol("omni_dummy");
  ID id = declare_ident(sym, CL_DECL_PRAGMA);
  expv new_st = list1(F_PRAGMA_STATEMENT, v);
  EXPV_LINE(new_st) = EXPR_LINE(x);
  id->info.decl_pragma_info.v = new_st;
  ID_LINE(id) = EXPR_LINE(x);

}


void
compile_pragma_outside(expr x)
{
  expv v = NULL;

  switch (EXPR_CODE(EXPR_ARG1(x)))
    {
    case STRING_CONSTANT:
	if (EXPR_STR(EXPR_ARG1(x)) == NULL) {
	    error("assertion fail on compile_pragma_statement, arg = NULL");
	    break;
	}
	else {
	  v = expv_str_term(STRING_CONSTANT,
			    NULL,
			    strdup(EXPR_STR(EXPR_ARG1(x))));
	  EXPV_LINE(v) = EXPR_LINE(x);
        break;
      }
    default:
      {
        error("invalid format.");
        break;
      }
    }

  SYMBOL sym = gen_temp_symbol("omni_dummy");
  EXT_ID ep = declare_external_id(sym, STG_PRAGMA , 0);
  ep->info.pragma_info.v = list1(F_PRAGMA_STATEMENT, v);
  EXT_LINE(ep) = EXPR_LINE(x);
  
}


/*
 * declare variables those have a specified type attribute
 *   OR
 * add a specified type attribute (in block scope)
 */
void
compile_declare_or_add_attribute_statement(expr id_list, uint32_t type_attr)
{
    list lp;
    expr ident;
    ID id = NULL;

    FOR_ITEMS_IN_LIST(lp, id_list) {
        ident = LIST_ITEM(lp);
        if ((id = find_ident_local(EXPR_SYM(ident))) == NULL) {
            if ((id = find_ident(EXPR_SYM(ident))) == NULL ||
                (ID_CLASS(id) != CL_VAR && ID_CLASS(id) != CL_UNKNOWN)) {
                id = declare_ident(EXPR_SYM(ident), CL_VAR);
                if(id == NULL) {
                    fatal("cannot declare ident %s", SYM_NAME(EXPR_SYM(ident)));
                }
            } else {
                /* id is use-/host-associated or out of the current BLOCK */
                TYPE_DESC tp = ID_TYPE(id);
                if (type_attr == TYPE_ATTR_VOLATILE && TYPE_IS_COINDEXED(tp)) {
                    error("the VOLATILE attribute shall not be specified for a coarray that is not local variable");
                    return;
                }
                id = declare_ident(EXPR_SYM(ident), CL_VAR);
                tp = wrap_type(tp);
                ID_TYPE(id) = tp;
                SET_MODIFIED(tp);

                TYPE_KIND(tp) = TYPE_KIND(TYPE_REF(tp));
                TYPE_N_DIM(tp) = TYPE_N_DIM(TYPE_REF(tp));
            }
        }
        if (ID_IS_AMBIGUOUS(id)) {
            error("an ambiguous reference to symbol '%s'", ID_NAME(id));
            return;
        }
        TYPE_ATTR_FLAGS(id) |= type_attr;
    }
}


/*
 * declare volatile variable
 *   OR
 * add volatile attribute (in block scope)
 */
void
compile_VOLATILE_statement(expr id_list)
{
    compile_declare_or_add_attribute_statement(id_list, TYPE_ATTR_VOLATILE);
}


/*
 * declare asynchronous variable
 *   OR
 * add asynchronous attribute (in block scope)
 */
void
compile_ASYNCHRONOUS_statement(expr id_list)
{
    compile_declare_or_add_attribute_statement(id_list, TYPE_ATTR_ASYNCHRONOUS);
}


/*
 * Apply type parameters to expv
 */
static expv
expv_apply_type_parameter(expv v, ID type_params)
{
    ID ip;
    int expv_is_replaced = FALSE;

    if (EXPV_CODE(v) == F_VAR) {
        /*
         * If expv is the variabalbe reference and refers to the type parameter,
         * replace it with the type parameter value.
         */
        TYPE_DESC tp = EXPV_TYPE(v);
        if (TYPE_IS_KIND(tp)) {
            ip = find_ident_head(EXPR_SYM(v), type_params);
            if (ip)
                return VAR_INIT_VALUE(ip);
        } else if (TYPE_IS_LEN(tp)) {
            ip = find_ident_head(EXPR_SYM(v), type_params);
            if (ip)
                return VAR_INIT_VALUE(ip);
        } else {
            return v;
        }
    } else if (!EXPR_CODE_IS_TERMINAL_OR_CONST(EXPR_CODE(v))) {
        /*
         * Apply recursively.
         * If the child expv is replaced, replace the current expv.
         */
        list lp1, lp2;
        expv new_v = v;
        FOR_ITEMS_IN_LIST(lp1, v) {
            expv original, applied;
            original = LIST_ITEM(lp1);
            applied = expv_apply_type_parameter(original, type_params);
            if (original != applied) {
                if (!expv_is_replaced) {
                    new_v = list0(EXPV_CODE(v));
                    *new_v = *v;
                    EXPR_LIST(new_v) = NULL;
                    FOR_ITEMS_IN_LIST(lp2, v) {
                        if (lp1 == lp2) break;
                        list_put_last(new_v, LIST_ITEM(lp2));
                    }
                    expv_is_replaced = TRUE;
                }
            }
            if (expv_is_replaced) {
                list_put_last(new_v, applied);
            }
        }
        new_v = expv_reduce(new_v, TRUE) ?: new_v;
        return new_v;
    }
    return v;
}


/*
 * Update type with type parameter values
 *
 * tp -- this funciton updates this TYPE_DESC
 * type_params -- type parameters and its values, values are stored in VAR_INIT_VALUE(id)
 */
static TYPE_DESC
type_apply_type_parameter0(TYPE_DESC tp, ID type_params)
{
    int type_is_replaced = FALSE;
    expv v;
    TYPE_DESC tq;
    ID ip;

    if (TYPE_BASIC_TYPE(tp) == TYPE_STRUCT && TYPE_TYPE_PARAM_VALUES(tp) != NULL) {
        /*
         * Update type parameter values of the instance of the parameterized derived-type
         *
         * ex)
         *   1  TYPE :: t(k)
         *   2    INTEGER, KIND :: k
         *   3    TYPE(d(k, kind2=k+4)) :: u
         *   4  END TYPE t
         *   5  TYPE(t(4)) :: v
         *
         * At line 3, member `u` is instanciated with type parameter value `k + 4`.
         *
         * At line 5, variable v is instanciated with type parameter value `4`,
         * this instaciation shallw updats type parameter values of member `u`.
         */
        ID new_member = NULL;
        ID last = NULL;
        TYPE_DESC parent = NULL;
        ID new_type_params = NULL;
        ID id, new_id;
        list lp;
        expv new_type_param_values;

        if (TYPE_TYPE_ACTUAL_PARAMS(tp) == NULL) {
            /*
             * Never reached and maybe error. Ignore.
             */
            return tp;
        }

        last = NULL;
        FOREACH_ID(id, TYPE_TYPE_ACTUAL_PARAMS(tp)) {
            new_id = new_ident_desc(ID_SYM(id));
            *new_id = *id;
            ID_LINK_ADD(new_id, new_type_params, last);
        }

        /*
         * Update old type parameter values.
         */
        FOREACH_ID(id, new_type_params) {
            expv v = VAR_INIT_VALUE(id);
            v = expv_apply_type_parameter(v, type_params);
            if (v != VAR_INIT_VALUE(id)) {
                type_is_replaced = TRUE;
                VAR_INIT_VALUE(id) = v;
            }
        }

        if (!type_is_replaced) {
            /*
             * No type parameter values are updated
             */
            return tp;
        }

        tp = wrap_type(tp);
        TYPE_TYPE_ACTUAL_PARAMS(tp) = new_type_params;

        /*
         * Also update TYPE_TYPE_PARAM_VALUES. (maybe not required)
         */
        new_type_param_values = list0(LIST);
        FOR_ITEMS_IN_LIST(lp, TYPE_TYPE_PARAM_VALUES(TYPE_REF(tp))) {
            expv v = expv_apply_type_parameter(LIST_ITEM(lp), type_params);
            list_put_last(new_type_param_values, v);
        }
        TYPE_TYPE_PARAM_VALUES(tp) = new_type_param_values;

        /*
         * Update the parent type
         */
        if (TYPE_PARENT(tp)) {
            parent = type_apply_type_parameter0(TYPE_PARENT_TYPE(tp), type_params);
        }

        /*
         * Update the type of each members
         */
        last = NULL;
        FOREACH_ID(ip, TYPE_MEMBER_LIST(TYPE_REF(tp))) {
            tq = type_apply_type_parameter0(ID_TYPE(ip), type_params);
            new_id = new_ident_desc(ID_SYM(ip));
            *new_id = *ip;
            ID_TYPE(new_id) = tq;
            ID_LINK_ADD(new_id, new_member, last);
        }

        if (parent != NULL) {
            TYPE_PARENT_TYPE(tp) = parent;
        }

        if (new_member != NULL) {
            TYPE_MEMBER_LIST(tp) = new_member;
        }

        TYPE_REF(tp) = TYPE_REF(TYPE_REF(tp));

    } else if (TYPE_KIND(tp)) {
        v = expv_apply_type_parameter(TYPE_KIND(tp), type_params);
        if (v != TYPE_KIND(tp)) {
            tp = copy_type_shallow(tp);
            TYPE_KIND(tp) = v;
            while (TYPE_REF(tp) && TYPE_KIND(TYPE_REF(tp))) {
                TYPE_REF(tp) = TYPE_REF(TYPE_REF(tp));
            }
        }

    } else if (TYPE_LENG(tp)) {
        v = expv_apply_type_parameter(TYPE_LENG(tp), type_params);
        if (v != TYPE_LENG(tp)) {
            tp = copy_type_shallow(tp);
            TYPE_LENG(tp) = v;
            while (TYPE_REF(tp) && TYPE_KIND(TYPE_REF(tp))) {
                TYPE_REF(tp) = TYPE_REF(TYPE_REF(tp));
            }
        }

    } else if (TYPE_REF(tp)) {
        tq = type_apply_type_parameter0(TYPE_REF(tp), type_params);
        if (tq != TYPE_REF(tp)) {
            tp = copy_type_shallow(tp);
            TYPE_REF(tp) = tq;

        }
    }

    return tp;
}

/*
 * Applies type parameter values to the parameterised derived-type and generate a new type
 *
 * A new type is also STRUCT_TYPE and its members have a type that is applied type parameters.
 *
 * Consider the input type is the following declaration:
 *
 *    TYPE t(k)
 *      INTEGER, KIND :: k
 *      INTEGER(KIND=k) :: i
 *      TYPE(u(l=k)) :: j
 *    END TYPE t
 *
 * If the type is passed k=8 at the type instansiation, the new type works like:
 *
 *    TYPE t
 *      INTEGER(KIND=8) :: i
 *      TYPE(u(l=8)) :: j
 *    END TYPE t
 *
 */
TYPE_DESC
type_apply_type_parameter(TYPE_DESC tp, expv type_param_values)
{
    ID ip, iq;
    ID last = NULL;
    TYPE_DESC tq;
    list lp;
    ID cur;
    expv compiled_type_param_values = list0(LIST);
    ID used;


    if (!IS_STRUCT_TYPE(tp)) {
        return NULL;
    }

    if (!compile_type_param_values(tp,
                                   type_param_values, compiled_type_param_values,
                                   &used)) {
        return NULL;
    }

    type_param_values = compiled_type_param_values;

    tq = copy_type_shallow(tp);

    if (TYPE_PARENT(tq)) {
        /*
         * Apply type parameter values to its parent type.
         */
        TYPE_DESC new_parent_type;
        ID new_parent;

        expv parent_type_param_values = list0(LIST);
        ID new_type_params = get_type_params(TYPE_PARENT_TYPE(tq));

        /*
         * Fix type parameter values so that the parent type can accept.
         */
        FOREACH_ID(ip, new_type_params) {
            iq = find_ident_head(ID_SYM(ip), used);
            VAR_INIT_VALUE(ip) = VAR_INIT_VALUE(iq);
        }
        cur = new_type_params;
        FOR_ITEMS_IN_LIST(lp, type_param_values) {
            expv v = LIST_ITEM(lp);
            if (EXPV_KWOPT_NAME(v)) {
                if (find_ident_head(find_symbol(EXPV_KWOPT_NAME(v)),
                                    cur)) {
                    list_put_last(parent_type_param_values, v);
                }
            } else {
                if (cur != NULL) {
                    list_put_last(parent_type_param_values, v);
                    cur = ID_NEXT(cur);
                }
            }
        }

        new_parent = new_ident_desc(ID_SYM(TYPE_PARENT(tq)));
        *new_parent = *(TYPE_PARENT(tq));
        new_parent_type = type_apply_type_parameter(TYPE_PARENT_TYPE(tq),
                                                    parent_type_param_values);
        TYPE_PARENT(tq) = new_parent;
        TYPE_PARENT_TYPE(tq) = new_parent_type;
    }

    FOREACH_ID(ip, TYPE_MEMBER_LIST(tq)) {
        /*
         * Apply type parameter values to each members.
         */
        TYPE_DESC member_tp = type_apply_type_parameter0(ID_TYPE(ip), used);
        ID new_id = new_ident_desc(ID_SYM(ip));
        *new_id = *ip;
        ID_TYPE(new_id) = member_tp;
        ID_LINK_ADD(new_id, TYPE_MEMBER_LIST(tq), last);
    }

    TYPE_REF(tq) = tp;
    TYPE_TYPE_PARAM_VALUES(tq) = type_param_values;

    /*
     * Save the type parameter values, including type parameters with initial
     * values , as key-value pair (ID_SYM(id) and VAR_INIT_VALUE(id)).
     * These pairs are used in the type comparison for the parameterized derived-type.
     */
    TYPE_TYPE_ACTUAL_PARAMS(tq) = used;

    return tq;
}

void
compile_procedure_attributes(expr attrs,
                             uint32_t *type_attr_flags_p,
                             uint32_t *binding_attr_flags_p,
                             ID *pass_arg_p,
                             expr *bind_name)
{
    uint32_t type_attr_flags = 0;
    uint32_t binding_attr_flags = 0;
    ID pass_arg = NULL;
    list lp;

    if (type_attr_flags_p == NULL ||
        binding_attr_flags_p == NULL ||
        pass_arg_p == NULL) {
        return;
    }

    FOR_ITEMS_IN_LIST(lp, attrs) {
        expr v = LIST_ITEM(lp);
        switch (EXPR_CODE(v)) {
            /*
             * Binding attributes
             */
            case F03_PASS_SPEC:
                if (binding_attr_flags & TYPE_BOUND_PROCEDURE_PASS) {
                    error("PASS is already specified.");
                    return;
                }
                if (binding_attr_flags & TYPE_BOUND_PROCEDURE_NOPASS) {
                    error("PASS and NO_PASS conflicts.");
                    return;
                }
                binding_attr_flags |= TYPE_BOUND_PROCEDURE_PASS;
                if (EXPR_ARG1(v)) {
                    pass_arg = new_ident_desc(EXPR_SYM(EXPR_ARG1(v)));
                }
                break;
            case F03_NO_PASS_SPEC:
                if (binding_attr_flags & TYPE_BOUND_PROCEDURE_NOPASS) {
                    error("NOPASS is already specified.");
                    return;
                }
                binding_attr_flags |= TYPE_BOUND_PROCEDURE_NOPASS;
                break;
            case F03_NON_OVERRIDABLE_SPEC:
                if (binding_attr_flags & TYPE_BOUND_PROCEDURE_NON_OVERRIDABLE) {
                    error("NON_OVERRIDABLE is already specified.");
                    return;
                }
                warning("NOT IMPLEMENTED YET\n");
                binding_attr_flags |= TYPE_BOUND_PROCEDURE_NON_OVERRIDABLE;
                break;
            case F03_DEFERRED_SPEC:
                if (binding_attr_flags & TYPE_BOUND_PROCEDURE_DEFERRED) {
                    error("DEFERRED is already specified.");
                    return;
                }
                binding_attr_flags |= TYPE_BOUND_PROCEDURE_DEFERRED;
                break;

            /*
             * Accesss specs
             */
            case F95_PUBLIC_SPEC:
                if (type_attr_flags & TYPE_ATTR_PUBLIC) {
                    error("PUBLIC is already specified.");
                    return;
                }
                type_attr_flags |= TYPE_ATTR_PUBLIC;
                break;
            case F95_PRIVATE_SPEC:
                if (type_attr_flags & TYPE_ATTR_PRIVATE) {
                    error("PRIVATE is already specified.");
                    return;
                }
                type_attr_flags |= TYPE_ATTR_PRIVATE;
                break;

            /*
             * Type attirbutes
             */
            case F95_POINTER_SPEC:
                type_attr_flags |= TYPE_ATTR_POINTER;
                break;
            case F95_OPTIONAL_SPEC:
                type_attr_flags |= TYPE_ATTR_OPTIONAL;
                break;
            case F95_SAVE_SPEC:
                type_attr_flags |= TYPE_ATTR_SAVE;
                break;
            case F95_INTENT_SPEC:
                switch(EXPR_CODE(EXPR_ARG1(v))) {
                    case F95_IN_EXTENT:
                        type_attr_flags |= TYPE_ATTR_INTENT_IN;
                        break;
                    case F95_OUT_EXTENT:
                        type_attr_flags |= TYPE_ATTR_INTENT_OUT;
                        break;
                    case F95_INOUT_EXTENT:
                        type_attr_flags |= TYPE_ATTR_INTENT_INOUT;
                        break;
                    default:
                        error("unexpected INTENT value");
                        break;
                }
                break;
            case LIST:
                type_attr_flags |= TYPE_ATTR_BIND;
                if (bind_name != NULL)
                    *bind_name = EXPR_ARG1(v);
                break;
            default:
                error("unexpected expression");
                break;
        }
    }

    if ((binding_attr_flags & TYPE_BOUND_PROCEDURE_PASS) &&
        (binding_attr_flags & TYPE_BOUND_PROCEDURE_NOPASS)) {
        error("PASS and NO_PASS conflicts.");
        return;
    }

    if ((binding_attr_flags & TYPE_BOUND_PROCEDURE_DEFERRED) &&
        (binding_attr_flags & TYPE_BOUND_PROCEDURE_NON_OVERRIDABLE)) {
        error("DEFERRED and NON_OVERRIDABLE conflicts.");
        return;
    }

    if ((type_attr_flags & (TYPE_ATTR_PRIVATE)) &&
        (type_attr_flags & (TYPE_ATTR_PUBLIC))) {
        error("access specs are conflicted.");
        return;
    }

    *type_attr_flags_p = type_attr_flags;
    *binding_attr_flags_p = binding_attr_flags;
    *pass_arg_p = pass_arg;
}


/*
 * (F03_PROCEDURE_DECLARATION
 *     (declarations ...)
 *     (attrs ...)
 *     [interface_name])
 */
void
compile_procedure_declaration(expr x)
{
    expr decls = EXPR_ARG1(x);
    expr attrs = EXPR_ARG2(x);
    expr proc_interface = EXPR_ARG3(x);
    ID id = NULL;
    int has_pass_arg = FALSE;
    ID pass_arg = NULL;
    ID interface = NULL;
    SYMBOL sym = NULL;
    list lp;
    TYPE_DESC tp = NULL;
    TYPE_DESC stp = NULL;

    uint32_t type_attr_flags = 0;
    uint32_t binding_attr_flags = 0;
    expr bind_name = NULL;

    int is_inside_struct = CTL_TYPE(ctl_top) == CTL_STRUCT;

    if (is_inside_struct) {
        stp = CTL_STRUCT_TYPEDESC(ctl_top);
    }

    compile_procedure_attributes(attrs,
                                 &type_attr_flags,
                                 &binding_attr_flags,
                                 &pass_arg,
                                 &bind_name);

    if (is_inside_struct) {
        if (!(type_attr_flags & TYPE_ATTR_POINTER)) {
            error("PROCEDURE member should have the POINTER attribute");
        }
        if ((type_attr_flags & TYPE_ATTR_OPTIONAL)) {
            error("PROCEDURE member should NOT have the OPTIONAL attribute");
        }
    } else {
        if (binding_attr_flags & TYPE_BOUND_PROCEDURE_PASS) {
            error("PROCEDURE variable cannot have the PASS attribute");
            return;
        }
        if (binding_attr_flags & TYPE_BOUND_PROCEDURE_NOPASS) {
            error("PROCEDURE variable cannot have the NOPASS attribute");
            return;
        }
    }
    if (binding_attr_flags & TYPE_BOUND_PROCEDURE_DEFERRED) {
        error("PROCEDURE member cannot have the DEFERRED attribute");
        return;
    }
    if (binding_attr_flags & TYPE_BOUND_PROCEDURE_NON_OVERRIDABLE) {
        error("PROCEDURE member cannot have the NON_OVERRIDEABLE attribute");
        return;
    }

    has_pass_arg = binding_attr_flags & TYPE_BOUND_PROCEDURE_PASS;
    if (pass_arg != NULL) {
        ID_TYPE(pass_arg) = wrap_type(stp);
        TYPE_SET_CLASS(ID_TYPE(pass_arg));
    }

    if (proc_interface == NULL) {
        /*
         * no name and type_spec
         *
         * ex)
         *   PROCEDURE(), POINTER :: p
         */
        tp = NULL;

    } else if (EXPR_CODE(proc_interface) != IDENT) {
        /*
         * proc_interface is type spec
         *
         * ex)
         *   PROCEDURE(REAL(KIND=8)) :: p
         */

        if (EXPR_TYPE(EXPR_ARG1(proc_interface)) == TYPE_STRUCT) {
            proc_interface = EXPR_ARG2(proc_interface);
        }
        tp = compile_type(proc_interface,
                          /* allow_predecl = */ FALSE);

        if (tp == NULL) {
            error("unexpected typeq");
        }

        tp = function_type(tp);

    } else {
        /*
         * interface is name
         *
         * ex)
         *   PROCEDURE(f), POINTER :: p
         */

        sym = EXPR_SYM(proc_interface);
        if ((interface = find_ident(sym)) != NULL) {
            if (IS_PROCEDURE_TYPE(ID_TYPE(interface))) {
                tp = ID_TYPE(interface);
                if (has_pass_arg &&
                    !procedure_has_pass_arg(tp,
                                            ID_SYM(pass_arg),
                                            stp)) {
                    error("%s should have a PASS argument %s",
                          SYM_NAME(sym), SYM_NAME(ID_SYM(pass_arg)));
                    return;
                }

            } else if (ID_TYPE(interface) != NULL) {
                TYPE_UNSET_SAVE(ID_TYPE(interface));
                ID_TYPE(interface) = function_type(ID_TYPE(interface));
                ID_CLASS(interface) = CL_PROC;
                PROC_CLASS(interface) = P_UNDEFINEDPROC;

            } else {
                TYPE_UNSET_SAVE(interface);
                ID_TYPE(interface) = function_type(new_type_desc());
                TYPE_BASIC_TYPE(FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(interface))) = TYPE_GNUMERIC_ALL;
                TYPE_SET_IMPLICIT(ID_TYPE(interface));
            }

        } else {
            /*
             * proce_interface does not appear YET!.
             *
             * proc_interface may be a contains function/subroutine or
             * an interface function/subroutine
             */

            TYPE_DESC ret = new_type_desc();
            TYPE_SET_IMPLICIT(ret);
            TYPE_BASIC_TYPE(ret) = TYPE_GNUMERIC_ALL;
            TYPE_SET_NOT_FIXED(ret);

            if (CTL_TYPE(ctl_top) == CTL_STRUCT) {
                /*
                 * don't call delcare_id here,
                 * declare_id make a member of struct
                 */
                tp = function_type(ret);
                interface = new_ident_desc(sym);
                ID_LINE(interface) = EXPR_LINE(x);
                ID_TYPE(interface) = tp;

            } else {
                interface = declare_ident(sym, CL_PROC);
                PROC_CLASS(interface) = P_UNDEFINEDPROC;
                declare_id_type(interface, function_type(ret));
                ID_CLASS(interface) = CL_PROC;
                declare_function(interface);

            }
        }

        tp = ID_TYPE(interface);
    }

    if (bind_name) {
        if (EXPR_HAS_ARG2(decls)) {
            error("multiple identifiers are declared with NAME=");
            return;
        }
    }

    FOR_ITEMS_IN_LIST(lp, decls) {
        if (EXPR_CODE(LIST_ITEM(lp)) == F03_BIND_PROCEDURE) {
            /*
             * The procedure variable declarations with an initial value.
             *
             * ex)
             *
             *  PROCEDURE(f), POINTER :: p1 => f, p2 => null()
             *
             */
            expr init_expr;
            expv v;

            if (!(type_attr_flags & TYPE_ATTR_POINTER)) {
                error("Initialization without a POINTER attribute");
                return;
            }

            id = declare_ident(EXPR_SYM(EXPR_ARG1(LIST_ITEM(lp))), CL_VAR);
            ID_LINE(id) = EXPR_LINE(LIST_ITEM(lp));

            init_expr = EXPR_ARG2(LIST_ITEM(lp));

            if (EXPR_CODE(init_expr) == IDENT) {
                /*
                 * type check is in check_procedure_variables()
                 */

                ID fid = find_ident(EXPR_SYM(init_expr));

                if (fid != NULL) {
                    v = expv_sym_term(F_VAR, ID_TYPE(fid), ID_SYM(fid));

                } else {
                    /* The type will be fixed in other function */
                    v = expv_sym_term(F_VAR, type_GNUMERIC_ALL, EXPR_SYM(init_expr));
                }
                EXPV_NEED_TYPE_FIXUP(v) = TRUE;

            } else if (EXPR_CODE(init_expr) == F_ARRAY_REF) {
                /*
                 * initialize expression should be NULL()
                 */
                ID fid = NULL;
                expr fname = EXPR_ARG1(init_expr);
                expr args = EXPR_ARG2(init_expr);
                expv a;
                if (EXPR_CODE(fname) != IDENT ||
                    EXPR_SYM(fname) != find_symbol("null")) {
                    error("not null() initialization");
                    continue;
                }
                fid = new_ident_desc(EXPR_SYM(fname));
                a = compile_args(args);
                if ((v = compile_intrinsic_call(fid, a)) == NULL) {
                    return;
                }

            } else {
                error("unexepected initialization expression for PROCEDRUE");
                continue;

            }

            VAR_INIT_VALUE(id) = v;

        } else if (EXPR_CODE(LIST_ITEM(lp)) == IDENT) {
            /*
             * without initialization
             */

            id = declare_ident(EXPR_SYM(LIST_ITEM(lp)), CL_VAR);

        } else {
            error("unexpected expression");
            continue;

        }

        if (tp) {
            declare_id_type(id, procedure_type(tp));

        } else {
            /*
             * `If proc-interface does not appear, the procedure declaration
             *  statement does not specify whether the declared
             *  procedures or procedure pointers are subroutines or functions.`
             *
             * If proc interface is null,
             * the function return type cannot be determined.
             *
             */
            TYPE_DESC tp = function_type(wrap_type(type_GNUMERIC_ALL));
            tp = procedure_type(tp);
            declare_id_type(id, tp);
        }

        /*
         * Now setup lines and types
         */
        ID_LINE(id) = EXPR_LINE(x);
        ID_ORDER(id) = order_sequence++;

        TYPE_ATTR_FLAGS(id) |= type_attr_flags;
        TYPE_ATTR_FLAGS(ID_TYPE(id)) |= type_attr_flags;
        TYPE_BIND_NAME(ID_TYPE(id)) = bind_name;
        VAR_REF_PROC(id) = interface;
        if (CTL_TYPE(ctl_top) == CTL_STRUCT) {
            FUNCTION_TYPE_HAS_BINDING_ARG(ID_TYPE(id)) = TRUE;
            FUNCTION_TYPE_HAS_PASS_ARG(ID_TYPE(id)) = binding_attr_flags & TYPE_BOUND_PROCEDURE_PASS;
            FUNCTION_TYPE_PASS_ARG(ID_TYPE(id)) = pass_arg;
            FUNCTION_TYPE_PASS_ARG_TYPE(ID_TYPE(id)) = stp;
        }
    }
}

/*
 * (F03_TYPE_BOUND_PROCEDURE_STATEMENT
 *     (binding_attrs ...)
 *     (bindings ...)
 *     [bind_to])
 */
void
compile_type_bound_procedure(expr x)
{
    expr bindings = EXPR_ARG1(x);
    expr binding_attrs = EXPR_ARG2(x);
    expr interface_name = EXPR_ARG3(x);
    ID id;
    ID interface;
    ID procedure;
    ID pass_arg = NULL;
    list lp;

    uint32_t access_attr_flags = 0;
    uint32_t binding_attr_flags = 0;


    compile_procedure_attributes(binding_attrs,
                                 &access_attr_flags,
                                 &binding_attr_flags,
                                 &pass_arg,
                                 NULL);

    if ((access_attr_flags & !(TYPE_ATTR_PUBLIC | TYPE_ATTR_PRIVATE)) != 0) {
        error("type-bound procedure cannot have type attributes");
        return;
    }

    if (binding_attr_flags & TYPE_ATTR_BIND) {
        error("type-bound procedure cannot have a BIND attribute");
        return;
    }

    if (!(binding_attr_flags & TYPE_BOUND_PROCEDURE_NOPASS)) {
        /* set a default attribute */
        binding_attr_flags |= TYPE_BOUND_PROCEDURE_PASS;
    }

    if ((binding_attr_flags & TYPE_BOUND_PROCEDURE_DEFERRED) != 0) {
        TYPE_DESC stp;
        if (interface_name == NULL) {
            error("DEFERRED shall appears if and only if interface_name appears.");
        }
        assert(CTL_TYPE(ctl_top) == CTL_STRUCT);

        stp = CTL_STRUCT_TYPEDESC(ctl_top);

        if (!TYPE_IS_ABSTRACT(stp)) {
            error("DEFERRED shall appears in ABSTRACT TYPE.");
        }
    }

    if (interface_name) {
        if (EXPR_CODE(interface_name) == IDENT) {
            interface = new_ident_desc(EXPR_SYM(interface_name));
        } else {
            error_at_node(x, "unexpected expression");
            return;
        }

        if (!(binding_attr_flags & TYPE_BOUND_PROCEDURE_DEFERRED)) {
            error_at_node(x, "require DEFERRED attribute");
        }
        FOR_ITEMS_IN_LIST(lp, bindings) {
            if (EXPR_CODE(LIST_ITEM(lp)) != IDENT) {
                error_at_node(x, "unexpected expression");
                continue;
            }
            id = declare_ident(EXPR_SYM(LIST_ITEM(lp)), CL_TYPE_BOUND_PROC);
            ID_LINE(id) = EXPR_LINE(x);
            TBP_BINDING(id) = interface;
            TBP_BINDING_ATTRS(id) = binding_attr_flags;
            TBP_PASS_ARG(id) = pass_arg;
            TYPE_ATTR_FLAGS(id) = access_attr_flags;
            ID_TYPE(id) = type_bound_procedure_type();
            TYPE_BOUND_PROCEDURE_TYPE_HAS_PASS_ARG(ID_TYPE(id)) = binding_attr_flags & TYPE_BOUND_PROCEDURE_PASS;
            TYPE_BOUND_PROCEDURE_TYPE_PASS_ARG(ID_TYPE(id)) = pass_arg;

        }
    } else {
        FOR_ITEMS_IN_LIST(lp, bindings) {
            if (EXPR_CODE(LIST_ITEM(lp)) == IDENT) {
                id = declare_ident(EXPR_SYM(LIST_ITEM(lp)), CL_TYPE_BOUND_PROC);
                procedure = new_ident_desc(EXPR_SYM(LIST_ITEM(lp)));
            } else if (EXPR_CODE(LIST_ITEM(lp)) == F03_BIND_PROCEDURE) {
                id = declare_ident(EXPR_SYM(EXPR_ARG1(LIST_ITEM(lp))), CL_TYPE_BOUND_PROC);
                procedure = new_ident_desc(EXPR_SYM(EXPR_ARG2(LIST_ITEM(lp))));
            } else {
                error_at_node(x, "unexpected expression");
                continue;
            }
            ID_LINE(id) = EXPR_LINE(x);

            TBP_BINDING(id) = procedure;
            TBP_BINDING_ATTRS(id) = binding_attr_flags;
            TBP_PASS_ARG(id) = pass_arg;
            TYPE_ATTR_FLAGS(id) = access_attr_flags;
            ID_TYPE(id) = type_bound_procedure_type();
            TYPE_BOUND_PROCEDURE_TYPE_HAS_PASS_ARG(ID_TYPE(id)) = binding_attr_flags & TYPE_BOUND_PROCEDURE_PASS;
            TYPE_BOUND_PROCEDURE_TYPE_PASS_ARG(ID_TYPE(id)) = pass_arg;

        }
    }
}

void
compile_type_bound_generic_procedure(expr x)
{
    expr generics_spec = EXPR_ARG1(x);
    expr id_list = EXPR_ARG2(x);
    expr access_attr = EXPR_ARG3(x);

    uint32_t access_attr_flags = 0;
    uint32_t binding_attr_flags = 0;
    int is_internal_private = FALSE;

    ID id = NULL;
    ID last_ip = NULL;

    list lp;

    TYPE_DESC struct_tp;

    assert(CTL_TYPE(ctl_top) == CTL_STRUCT);
    struct_tp = CTL_STRUCT_TYPEDESC(ctl_top);
    is_internal_private = TYPE_IS_INTERNAL_PRIVATE(struct_tp);

    switch (EXPR_CODE(generics_spec)) {
        case IDENT:
            id = find_ident_head(EXPR_SYM(generics_spec), TYPE_MEMBER_LIST(struct_tp));
            if (id != NULL) {
                // Multiple GENERIC statement for the same id. Have to add to 
                // the binding list. 
                last_ip = TBP_BINDING(id);
                while(ID_NEXT(last_ip) != NULL) {
                    last_ip = ID_NEXT(last_ip);
                }
            } else {
                // First GENERIC statement for and id. Binding list is empty.
                id = declare_ident(EXPR_SYM(generics_spec), CL_TYPE_BOUND_PROC);
            }
            break;
        case F95_GENERIC_SPEC:{
            expr arg;
            arg = EXPR_ARG1(generics_spec);
            SYMBOL sym = find_symbol(EXPR_CODE_SYMBOL(EXPR_CODE(arg)));
            if ((id = find_ident_head(sym, TYPE_MEMBER_LIST(struct_tp))) == NULL) {
                id = declare_ident(sym, CL_TYPE_BOUND_PROC);
            }
            switch (EXPR_CODE(arg)) {
                case F95_ASSIGNOP:
                    binding_attr_flags |= TYPE_BOUND_PROCEDURE_IS_ASSIGNMENT;
                    break;
                case F95_PLUSOP:
                case F95_MINUSOP:
                    binding_attr_flags |= TYPE_BOUND_PROCEDURE_IS_OPERATOR;
                    break;
                case F95_NOTOP:
                    binding_attr_flags |= TYPE_BOUND_PROCEDURE_IS_UNARY_OPERATOR;
                case F95_DOTOP:
                case F95_POWEOP:
                case F95_MULOP:
                case F95_DIVOP:
                case F95_EQOP:
                case F95_NEOP:
                case F95_LTOP:
                case F95_LEOP:
                case F95_GEOP:
                case F95_GTOP:
                case F95_ANDOP:
                case F95_OROP:
                case F95_EQVOP:
                case F95_NEQVOP:
                case F95_CONCATOP:
                    binding_attr_flags |= TYPE_BOUND_PROCEDURE_IS_BINARY_OPERATOR;
                    break;
                default:
                    binding_attr_flags |= TYPE_BOUND_PROCEDURE_IS_OPERATOR;
                    break;
            }
        } break;
        case F95_USER_DEFINED:{
            expr arg;
            arg = EXPR_ARG1(generics_spec);
            if ((id = find_ident_head(EXPR_SYM(arg), TYPE_MEMBER_LIST(struct_tp))) == NULL) {
                id = declare_ident(EXPR_SYM(arg), CL_TYPE_BOUND_PROC);
                binding_attr_flags |= TYPE_BOUND_PROCEDURE_IS_OPERATOR;
            }
        } break;
        case F03_GENERIC_WRITE: {
            expr formatted = EXPR_ARG1(generics_spec);
            SYMBOL sym;
            binding_attr_flags |= TYPE_BOUND_PROCEDURE_WRITE;
            switch (EXPR_CODE(formatted)) {
                case F03_FORMATTED:
                    sym = find_symbol("_write_formatted");
                    binding_attr_flags |= TYPE_BOUND_PROCEDURE_FORMATTED;
                    break;
                case F03_UNFORMATTED:
                    sym = find_symbol("_write_unformatted");
                    binding_attr_flags |= TYPE_BOUND_PROCEDURE_UNFORMATTED;
                    break;
                default:
                    error("expect FORMATTED or UNFORMATTED");
                    return;
            }
            id = declare_ident(sym, CL_TYPE_BOUND_PROC);
        } break;
        case F03_GENERIC_READ: {
            expr formatted = EXPR_ARG1(generics_spec);
            SYMBOL sym;
            binding_attr_flags |= TYPE_BOUND_PROCEDURE_READ;
            switch (EXPR_CODE(formatted)) {
                case F03_FORMATTED:
                    sym = find_symbol("_read_formatted");
                    binding_attr_flags |= TYPE_BOUND_PROCEDURE_FORMATTED;
                    break;
                case F03_UNFORMATTED:
                    sym = find_symbol("_read_unformatted");
                    binding_attr_flags |= TYPE_BOUND_PROCEDURE_UNFORMATTED;
                    break;
                default:
                    error("expect FORMATTED or UNFORMATTED");
                    return;
            }
            id = declare_ident(sym, CL_TYPE_BOUND_PROC);
        } break;
        default:
            error_at_node(x, "unexpected expression");
            break;
    }

    if (access_attr) {
        switch (EXPR_CODE(access_attr)) {
            /*
             * Accesss specs
             */
            case F95_PUBLIC_SPEC:
                if (access_attr_flags & TYPE_ATTR_PUBLIC) {
                    error_at_node(x, "PUBLIC is already specified.");
                    return;
                }
                if (access_attr_flags & (TYPE_ATTR_PRIVATE)) {
                    error_at_node(x, "access specs are conflicted.");
                    return;
                }
                access_attr_flags |= TYPE_ATTR_PUBLIC;
                break;
            case F95_PRIVATE_SPEC:
                if (access_attr_flags & TYPE_ATTR_PRIVATE) {
                    error_at_node(x, "PRIVATE is already specified.");
                    return;
                }
                if (access_attr_flags & (TYPE_ATTR_PUBLIC)) {
                    error_at_node(x, "access specs are conflicted.");
                    return;
                }
                access_attr_flags |= TYPE_ATTR_PRIVATE;
                break;
            default:
                error_at_node(x, "unexpected expression");
            break;
        }
    } else if (is_internal_private) {
        access_attr_flags |= TYPE_ATTR_PRIVATE;
    }

    FOR_ITEMS_IN_LIST(lp, id_list) {
        ID binding;
        if (EXPR_CODE(LIST_ITEM(lp)) != IDENT) {
            error_at_node(x, "unexpected expression");
        }
        binding = new_ident_desc(EXPR_SYM(LIST_ITEM(lp)));
        ID_LINK_ADD(binding, TBP_BINDING(id), last_ip);
    }

    TBP_BINDING_ATTRS(id) = binding_attr_flags | TYPE_BOUND_PROCEDURE_IS_GENERIC;
    TYPE_ATTR_FLAGS(id) = access_attr_flags;
    ID_TYPE(id) = type_bound_procedure_type();
    ID_LINE(id) = EXPR_LINE(x);
    TYPE_BOUND_GENERIC_TYPE_GENERICS(ID_TYPE(id)) = TBP_BINDING(id);
}

void
compile_FINAL_statement(expr x)
{
    expr id_list = EXPR_ARG1(x);

    ID id = NULL;
    ID last_ip = NULL;
    list lp;

    assert(CTL_TYPE(ctl_top) == CTL_STRUCT);
    id = declare_ident(find_symbol(FINALIZER_PROCEDURE), CL_TYPE_BOUND_PROC);

    FOR_ITEMS_IN_LIST(lp, id_list) {
        ID binding;
        if (EXPR_CODE(LIST_ITEM(lp)) != IDENT) {
            error_at_node(x, "unexpected expression");
        }
        if (find_ident_head(EXPR_SYM(LIST_ITEM(lp)), TBP_BINDING(id)) != NULL) {
            error_at_node(x, "%s is already used as FINAL procedure",
                          SYM_NAME(EXPR_SYM(LIST_ITEM(lp))));
            continue;
        }
        binding = new_ident_desc(EXPR_SYM(LIST_ITEM(lp)));
        ID_LINK_ADD(binding, TBP_BINDING(id), last_ip);
    }

    TBP_BINDING_ATTRS(id) = TYPE_BOUND_PROCEDURE_IS_FINAL;

    ID_TYPE(id) = type_bound_procedure_type();
    ID_LINE(id) = EXPR_LINE(x);
}


/*
 * declare asynchronous variable
 *   OR
 * add asynchronous attribute (in block scope)
 */
void
compile_BIND_statement(expr bind_opt, expr id_list)
{
    list lp;
    ID id;
    expr bind_name = EXPR_ARG1(bind_opt);

    FOR_ITEMS_IN_LIST(lp, id_list) {
        expr elm = LIST_ITEM(lp);
        if (EXPR_CODE(elm) == IDENT) {
            if ((id = declare_ident(EXPR_SYM(elm), CL_VAR)) == NULL) {
                fatal("cannot declare ident %s", SYM_NAME(EXPR_SYM(elm)));
            }
        } else {
            /* for 'BIND(C) :: /COMMON/' */
            elm = EXPR_ARG1(elm);
            if ((id = find_ident_head(EXPR_SYM(elm), LOCAL_COMMON_SYMBOLS)) == NULL) {
                if ((id = declare_common_ident(EXPR_SYM(elm))) == NULL) {
                    fatal("cannot declare common block %s", SYM_NAME(EXPR_SYM(elm)));
                }
            }
        }
        TYPE_SET_BIND(id);
        if (bind_name){
            ID_BIND(id) = bind_name;
        }
    }
}
