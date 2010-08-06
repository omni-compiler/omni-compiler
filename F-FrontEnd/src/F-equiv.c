/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file F-equiv.c
 */

#include "F-front.h"

void
compile_EQUIVALENCE_decl(expr x)
{
    list lp;
    list lp1;
    expr spec;
    expr vX;
    expv v;
    ID id, refID = NULL;
    SYMBOL vS;

    expv refVarV = NULL;
    expv listVarV = NULL;

    if (EXPR_CODE(x) != LIST) {
        fatal("paser error in equivalence??");
    }

    FOR_ITEMS_IN_LIST(lp, x) {
        spec = LIST_ITEM(lp);
        if (EXPR_CODE(spec) != LIST) {
            fatal("paser error in equivalence??");
        }

        refVarV = NULL;
        listVarV = list0(LIST);
        FOR_ITEMS_IN_LIST(lp1, spec) {
            vX = LIST_ITEM(lp1);
            vS = NULL;

            switch (EXPR_CODE(vX)) {
                case IDENT: {
                    vS = EXPR_SYM(vX);
                    break;
                }
                case F_ARRAY_REF: {
                    vS = EXPR_SYM(EXPR_ARG1(vX));
                    break;
                }
                default: {
                    error_at_node(vX, "invalid expression.");
                    return;
                }
            }

            id = find_ident(vS);
            if (id == NULL) {
	      // vS should be treated as being implicitly declared. (Hitoshi Murai)
	      id = declare_ident(vS, CL_VAR);
	      implicit_declaration(id);
	      //                error_at_node(vX, "'%s' is not declared.", SYM_NAME(vS));
	      //                return;
            }

            v = compile_lhs_expression(vX);

            if(v == NULL)
                return;

            switch (EXPV_CODE(v)) {
                case F_VAR:
                case ARRAY_REF: {
                    if (ID_CLASS(id) == CL_PARAM ||
                        TYPE_IS_PARAMETER(id)) {
                        error_at_node(vX, "'%s' is a parameter.",
                                      SYM_NAME(vS));
                        return;
                    }
                    if (TYPE_IS_ALLOCATABLE(id)) {
                        error_at_node(vX, "'%s' is an allocatable type.",
                                      SYM_NAME(vS));
                        return;
                    }
                    if (TYPE_IS_POINTER(id)) {
                        error_at_node(vX, "'%s' is an pointer.",
                                      SYM_NAME(vS));
                        return;
                    }
                    /*
                     * FIXME:
                     *	Add more restriction check.
                     */

                    break;
                }
                default: {
                    error_at_node(vX, "'%s' is not a variable.", SYM_NAME(vS));
                    return;
                }
            }

            if (refVarV == NULL) {
                refVarV = v;
                refID = id;
            } else {
                listVarV = list_put_last(listVarV, v);
                id->equivID = refID;
            }
        }
        output_statement(list2(F_EQUIV_DECL, refVarV, listVarV));
    }
}
