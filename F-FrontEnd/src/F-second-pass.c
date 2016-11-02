#include "F-front.h"
/* #define FE_DEBUG */
#ifdef FE_DEBUG
#include "test.h"
#endif

#define TYPE_ID 1
#define TYPE_EXPR 2

struct sp_list {
  struct sp_list *prev;
  struct sp_list *next;
  int type;
  int nest_level;
  int err_no;
  lineno_info *line;
  EXT_ID nest_ext_id[MAX_UNIT_CTL_CONTAINS];
  union {
    ID id;
    expr ep;
  } info;
};

struct sp_list_l{
  struct sp_list *head;
  struct sp_list *tail;
};

typedef struct sp_list SP_LIST;
typedef struct sp_list_l SP_LIST_L;

static SP_LIST_L top_sp_list; /* undefined ID and expr(expv) list */

void second_pass_init()
{
  top_sp_list.head = XMALLOC(SP_LIST*, sizeof(SP_LIST));
  top_sp_list.tail = XMALLOC(SP_LIST*, sizeof(SP_LIST));
  top_sp_list.head->prev = NULL;
  top_sp_list.head->next = top_sp_list.tail;
  top_sp_list.tail->prev = top_sp_list.head;
  top_sp_list.tail->next = NULL;
  top_sp_list.head->err_no = 0;
  top_sp_list.tail->err_no = 0;
}

static void link_sp_list(SP_LIST *list)
{
  top_sp_list.tail->prev->next = list;
  list->next = top_sp_list.tail;
  list->prev = top_sp_list.tail->prev;
  top_sp_list.tail->prev = list;
}

static SP_LIST *unlink_sp_list(SP_LIST *list)
{
  SP_LIST *ret;

  ret = list->prev;
  list->prev->next = list->next;
  list->next->prev = list->prev;
  free(list);

  return ret;
}

#define FOREACH_SP_LIST(x) for((x)=top_sp_list.head->next;(x)!=top_sp_list.tail;(x)=(x)->next)

void sp_link_id(ID id, int err_no, lineno_info *line)
{
  SP_LIST *list;
  int i;

  FOREACH_SP_LIST(list){
    if(list->info.id == id) return;
  }

  /* printf("!!! debug sp_link_id(%s)\n", ID_NAME(id)); */
  list = XMALLOC(SP_LIST*, sizeof(SP_LIST));
  list->next = NULL;
  list->info.id = id;
  list->type = TYPE_ID;
  list->nest_level = unit_ctl_level;
  list->err_no = err_no;
  list->line = line;
  for(i=0; i<=unit_ctl_level; i++){
    list->nest_ext_id[i] = UNIT_CTL_CURRENT_EXT_ID(unit_ctls[i]);
  }
  link_sp_list(list);
}

void sp_link_expr(expr ep, int err_no, lineno_info *line)
{
  SP_LIST *list;
  int i;

  FOREACH_SP_LIST(list){
    if(list->info.ep == ep) return;
  }
  /* printf("!!! debug sp_link_expr(%s)\n", _expr_code[EXPR_CODE(ep)]); */
  list = XMALLOC(SP_LIST*, sizeof(SP_LIST));
  list->next = NULL;
  list->info.ep = ep;
  list->type = TYPE_EXPR;
  list->nest_level = unit_ctl_level;
  list->err_no = err_no;
  list->line = line;
  for(i=0; i<=unit_ctl_level; i++){
    list->nest_ext_id[i] = UNIT_CTL_CURRENT_EXT_ID(unit_ctls[i]);
  }
  link_sp_list(list);
}

static int second_pass_clean()
{
  int err_num=0;
  SP_LIST *list = top_sp_list.head;
  while(list){
    SP_LIST *next = list->next;
    /* error */
    switch(list->err_no){
    case 1:
      current_line = list->line;
      error("attempt to use undefined type variable, %s", ID_NAME(list->info.id));
      err_num++;
      break;
    case 2:
      error_at_node(list->info.ep,
                    "character string length must be integer.");
      err_num++;
      break;
    case 3:
      current_line = list->line;
      error("%s: invalid code", SYM_NAME(EXPR_SYM(list->info.ep)));
      err_num++;
      break;
    case 4:
      current_line = list->line;
      TYPE_DESC tp = list->info.id->type;
      if (tp && !TYPE_IS_NOT_FIXED(tp)) break;
      error("attempt to use undefined type function, %s", ID_NAME(list->info.id));
      err_num++;
      break;
    default:
      break;
    }
    free((void*)list);
    list = next;
  }

  return err_num;
}


#ifdef FE_DEBUG
static int slen=0;

static void second_pass_expv_scan(expv v)
{
  enum expr_code code;
  int i;

  if(v == NULL)
    return;
  code = EXPV_CODE(v);
  printf(" body: ");
  for(i=0; i<slen; i++) printf("  ");
  printf("%s\n", _expr_code[code]);
  switch(code) {
  /*
   * child elements
   */
  case LIST:
    {
      list lp;
      slen++;
      FOR_ITEMS_IN_LIST(lp, v)
        second_pass_expv_scan(LIST_ITEM(lp));
      slen--;
    }
    break;

  /*
   * identifiers
   */
  case F_FUNC:
  case IDENT:
  case F_VAR:
  case F_PARAM:

  /*
   * constants
   */
  case INT_CONSTANT:
  case STRING_CONSTANT:
  case FLOAT_CONSTANT:
  case COMPLEX_CONSTANT:

  /*
   * declarations
   */
  case F_FORMAT_DECL:
    break;

  /*
   * general statements
   */
  case EXPR_STATEMENT:
    {
      expv v1;
      v1 = EXPR_ARG1(v);        /* expression */
    }
    break;
  case F_DO_STATEMENT:
    {
      expv vl, vr, v1, v2, v3, v4, v5;
      vl = EXPR_ARG1(v);        /* ConstructName */
      vr = EXPR_ARG2(v);        /* condition */
      v1 = EXPR_ARG1(vr);       /* do var */
      v2 = EXPR_ARG2(vr);       /* init variable */
      v3 = EXPR_ARG3(vr);       /* end variable */
      v4 = EXPR_ARG4(vr);       /* step variable */
      v5 = EXPR_ARG5(vr);       /* body */
      second_pass_expv_scan(v5);
    }
    break;
  case F_DOWHILE_STATEMENT:
    {
      expv v1, v2, v3;
      v1 = EXPR_ARG1(v);        /* condition */
      v2 = EXPR_ARG2(v);        /* body */
      v3 = EXPR_ARG3(v);        /* ConstructName */
      second_pass_expv_scan(v2);
    }
    break;
  case F03_SELECTTYPE_STATEMENT:
    {
      expv v3, v4;
      list lp = EXPR_LIST(v);   /* condition & body */
      v3 = EXPR_ARG3(v);        /* ConstructName */
      v4 = EXPR_ARG4(v);        /* associate name */

      /* LIST_ITEM(lp) : select(var) ?*/
      if(LIST_NEXT(lp) && LIST_ITEM(LIST_NEXT(lp))) {
        FOR_ITEMS_IN_LIST(lp, LIST_ITEM(LIST_NEXT(lp)))
          second_pass_expv_scan(LIST_ITEM(lp));
      }
    }
    break;
  case F_SELECTCASE_STATEMENT:
    {
      expv v3, v4;
      list lp = EXPR_LIST(v);   /* condition & body */
      v3 = EXPR_ARG3(v);        /* ConstructName */

      /* LIST_ITEM(lp) : select(var) ?*/
      if(LIST_NEXT(lp) && LIST_ITEM(LIST_NEXT(lp))) {
        FOR_ITEMS_IN_LIST(lp, LIST_ITEM(LIST_NEXT(lp)))
          second_pass_expv_scan(LIST_ITEM(lp));
      }
    }
    break;
  case IF_STATEMENT:
  case F_WHERE_STATEMENT:
  case F_RETURN_STATEMENT:
  case F_CONTINUE_STATEMENT:
  case GOTO_STATEMENT:
  case F_COMPGOTO_STATEMENT:
  case STATEMENT_LABEL:
    break;
  case F03_TYPEIS_STATEMENT:
  case F03_CLASSIS_STATEMENT:
  case F_CASELABEL_STATEMENT:
    {
      expv v1, v2, v3;
      v1 = EXPR_ARG1(v);        /* condition */
      v2 = EXPR_ARG2(v);        /* body */
      v3 = EXPR_ARG3(v);        /* ConstructName */
      slen++;
      second_pass_expv_scan(v2);
      slen--;
    }
    break;
  case F_STOP_STATEMENT:
  case F_PAUSE_STATEMENT:
  case F_LET_STATEMENT:
  case F_PRAGMA_STATEMENT:
  case F95_CYCLE_STATEMENT:
  case F95_EXIT_STATEMENT:
  case F_ENTRY_STATEMENT:

  /*
   * IO statements
   */
  case F_PRINT_STATEMENT:
  case F_READ_STATEMENT:
  case F_WRITE_STATEMENT:
  case F_INQUIRE_STATEMENT:
  case F_READ1_STATEMENT:
  case F_OPEN_STATEMENT:
  case F_CLOSE_STATEMENT:
  case F_BACKSPACE_STATEMENT:
  case F_ENDFILE_STATEMENT:
  case F_REWIND_STATEMENT:

  /*
   * F90/95 Pointer related.
   */
  case F95_POINTER_SET_STATEMENT:
  case F95_ALLOCATE_STATEMENT:
  case F95_DEALLOCATE_STATEMENT:
  case F95_NULLIFY_STATEMENT:

  /*
   * expressions
   */
  case FUNCTION_CALL:
  case F95_MEMBER_REF:
  case ARRAY_REF:
  case F_SUBSTR_REF:
  case F95_ARRAY_CONSTRUCTOR:
  case F95_STRUCT_CONSTRUCTOR:

  case XMP_COARRAY_REF:
  /*
   * operators
   */
  case PLUS_EXPR:
  case MINUS_EXPR:
  case MUL_EXPR:
  case DIV_EXPR:
  case POWER_EXPR:
  case LOG_EQ_EXPR:
  case LOG_NEQ_EXPR:
  case LOG_GE_EXPR:
  case LOG_GT_EXPR:
  case LOG_LE_EXPR:
  case LOG_LT_EXPR:
  case LOG_AND_EXPR:
  case LOG_OR_EXPR:
  case F_EQV_EXPR:
  case F_NEQV_EXPR:
  case F_CONCAT_EXPR:
  case LOG_NOT_EXPR:
  case UNARY_MINUS_EXPR:


  case F95_USER_DEFINED_BINARY_EXPR:
  case F95_USER_DEFINED_UNARY_EXPR:

  /*
   * misc.
   */
  case F_IMPLIED_DO:
  case F_INDEX_RANGE:
  case F_SCENE_RANGE_EXPR:
  case F_MODULE_INTERNAL:
  /*
   * When using other module, F_MODULE_INTERNAL is set as dummy
   * expression insted of real value defined in module.
   * We emit dummy FintConstant for F_Back.
   */

  /*
   * elements to skip
   */
  case F_DATA_DECL:
  case F_EQUIV_DECL:
  case F95_TYPEDECL_STATEMENT:
  case FIRST_EXECUTION_POINT:
  case F95_INTERFACE_STATEMENT:
  case F95_USE_STATEMENT:
  case F95_USE_ONLY_STATEMENT:

  /*
   * invalid or no corresponding tag
   */
  case ERROR_NODE:
  case BASIC_TYPE_NODE:
  case DEFAULT_LABEL:
  case ID_LIST:
  case VAR_DECL:
  case EXT_DECL:
  case F_PROGRAM_STATEMENT:
  case F_BLOCK_STATEMENT:
  case F_SUBROUTINE_STATEMENT:
  case F_FUNCTION_STATEMENT:
  case F_INCLUDE_STATEMENT:
  case F_END_STATEMENT:
  case F_TYPE_DECL:
  case F_COMMON_DECL:
  case F_EXTERNAL_DECL:
  case F_INTRINSIC_DECL:
  case F_IMPLICIT_DECL:
  case F_NAMELIST_DECL:
  case F_SAVE_DECL:
  case F_PARAM_DECL:
  case F_DUP_DECL:
  case F_UNARY_MINUS:
  case F_ENDDO_STATEMENT:
  case F_ELSEWHERE_STATEMENT:
  case F_ENDWHERE_STATEMENT:
  case F_ENDSELECT_STATEMENT:
  case F_IF_STATEMENT:
  case F_ELSEIF_STATEMENT:
  case F_ELSE_STATEMENT:
  case F_ENDIF_STATEMENT:
  case F_ASSIGN_LABEL_STATEMENT:
  case F_GOTO_STATEMENT:
  case F_ASGOTO_STATEMENT:
  case F_ARITHIF_STATEMENT:
  case F_CALL_STATEMENT:
  case F_CRAY_POINTER_DECL:
  case F_SET_EXPR:
  case F_LABEL_REF:
  case F_PLUS_EXPR:
  case F_MINUS_EXPR:
  case F_MUL_EXPR:
  case F_DIV_EXPR:
  case F_UNARY_MINUS_EXPR:
  case F_POWER_EXPR:
  case F_EQ_EXPR:
  case F_GT_EXPR:
  case F_GE_EXPR:
  case F_LT_EXPR:
  case F_LE_EXPR:
  case F_NE_EXPR:
  case F_OR_EXPR:
  case F_AND_EXPR:
  case F_NOT_EXPR:
  case F_ARRAY_REF:
  case F_STARSTAR:
  case F_TRUE_CONSTANT:
  case F_FALSE_CONSTANT:
  case F_TYPE_NODE:
  case F95_CONSTANT_WITH:
  case F95_TRUE_CONSTANT_WITH:
  case F95_FALSE_CONSTANT_WITH:
  case F95_ENDPROGRAM_STATEMENT:
  case F95_ENDSUBROUTINE_STATEMENT:
  case F95_ENDFUNCTION_STATEMENT:
  case F95_MODULE_STATEMENT:
  case F95_ENDMODULE_STATEMENT:
  case F95_ENDINTERFACE_STATEMENT:
  case F95_CONTAINS_STATEMENT:
  case F95_RECURSIVE_SPEC:
  case F95_PURE_SPEC:
  case F95_ELEMENTAL_SPEC:
  case F95_DIMENSION_DECL:
  case F95_ENDTYPEDECL_STATEMENT:
  case F95_PRIVATE_STATEMENT:
  case F03_PROTECTED_STATEMENT:
  case F95_SEQUENCE_STATEMENT:
  case F95_PARAMETER_SPEC:
  case F95_ALLOCATABLE_SPEC:
  case F95_DIMENSION_SPEC:
  case F95_EXTERNAL_SPEC:
  case F95_INTENT_SPEC:
  case F95_INTRINSIC_SPEC:
  case F95_OPTIONAL_SPEC:
  case F95_POINTER_SPEC:
  case F95_SAVE_SPEC:
  case F95_TARGET_SPEC:
  case F95_PUBLIC_SPEC:
  case F95_PRIVATE_SPEC:
  case F03_PROTECTED_SPEC:
  case F95_IN_EXTENT:
  case F95_OUT_EXTENT:
  case F95_INOUT_EXTENT:
  case F95_KIND_SELECTOR_SPEC:
  case F95_LEN_SELECTOR_SPEC:
  case F95_STAT_SPEC:
  case F95_TRIPLET_EXPR:
  case F95_PUBLIC_STATEMENT:
  case F95_OPTIONAL_STATEMENT:
  case F95_POINTER_STATEMENT:
  case F95_INTENT_STATEMENT:
  case F95_TARGET_STATEMENT:
  case F_ASTERISK:
  case F_EXTFUNC:
  case F_DOUBLE_CONSTANT:
  case F95_ASSIGNOP:
  case F95_DOTOP:
  case F95_POWEOP:
  case F95_MULOP:
  case F95_DIVOP:
  case F95_PLUSOP:
  case F95_MINUSOP:
  case F95_EQOP:
  case F95_NEOP:
  case F95_LTOP:
  case F95_LEOP:
  case F95_GEOP:
  case F95_GTOP:
  case F95_NOTOP:
  case F95_ANDOP:
  case F95_OROP:
  case F95_EQVOP:
  case F95_NEQVOP:
  case F95_CONCATOP:
  case F95_USER_DEFINED:
  case F95_MODULEPROCEDURE_STATEMENT:
  case F95_ARRAY_ALLOCATION:
  case F95_ALLOCATABLE_STATEMENT:
  case F95_GENERIC_SPEC:
  case XMP_CODIMENSION_SPEC:
  case EXPR_CODE_END:
  case OMP_PRAGMA:

  case XMP_PRAGMA:
  case ACC_PRAGMA:
    break;

  default:
    fatal("unkown exprcode : %d", code);
    abort();
  }
}
#endif

int second_pass()
{
#ifdef FE_DEBUG
  EXT_ID ep;
  ID     id;
  list   lp;
  FOREACH_EXT_ID(ep, EXTERNAL_SYMBOLS) {
    if(EXT_SYM(ep)){
      printf("ext symbol name: %s\n", SYM_NAME(EXT_SYM(ep)));
      FOR_ITEMS_IN_LIST(lp, EXT_PROC_ARGS(ep)){
        printf("  args type: %s\n", _expr_code[EXPR_CODE(LIST_ITEM(lp))]);
      }
      FOREACH_ID(id, EXT_PROC_ID_LIST(ep)){
        if(id){
          printf("  proc symbol name: %s(%p)(class=%d)", ID_NAME(id), ID_SYM(id), ID_CLASS(id));
          if(ID_TYPE(id) == NULL){
            printf(" (type null)");
          } else {
            printf(" (type = %d)", ID_TYPE(id)->basic_type);
          }
          printf("\n");
        }
      }
      if(EXT_PROC_CONT_EXT_SYMS(ep)){
        FOREACH_ID(id, EXT_PROC_ID_LIST(EXT_PROC_CONT_EXT_SYMS(ep))){
          if(id){
            printf("  contains symbol name: %s(%p)(class=%d)", ID_NAME(id), ID_SYM(id), ID_CLASS(id));
            if(ID_TYPE(id) == NULL){
              printf(" (type null)");
            } else {
              printf(" (type = %d)", ID_TYPE(id)->basic_type);
            }
            printf("\n");
          }
        }
      }
/* EXT_PROC_CONT_EXT_SYMS */
      FOREACH_ID(id, EXT_PROC_COMMON_ID_LIST(ep)) {
        if(id){
          printf("  common block symbol name: %s", ID_NAME(id));
          printf("\n");
        }
      }
    }

    if(EXT_PROC_BODY(ep)){
      EXT_ID contains_1, contains_1_ep;
      EXT_ID contains_2, contains_2_ep;
      second_pass_expv_scan(EXT_PROC_BODY(ep));
      contains_1 = EXT_PROC_CONT_EXT_SYMS(ep);
      if(contains_1){
        slen++;
        FOREACH_EXT_ID(contains_1_ep, contains_1){
          second_pass_expv_scan(EXT_PROC_BODY(contains_1_ep));
          contains_2 = EXT_PROC_CONT_EXT_SYMS(contains_1_ep);
          if(contains_2){
            slen++;
            FOREACH_EXT_ID(contains_2_ep, contains_2){
              second_pass_expv_scan(EXT_PROC_BODY(contains_2_ep));

            }
            slen--;
          }
        }
        slen--;
      }
    }
  }
#endif

  SP_LIST *sp_list;
  FOREACH_SP_LIST(sp_list){
    int i;

#ifdef FE_DEBUG
    /* debug */
    if(sp_list->type == TYPE_ID){ /* ID */
      printf(" undefined symbol name: %s(%p) class=%d: nest=(",
             ID_NAME(sp_list->info.id), ID_SYM(sp_list->info.id), ID_CLASS(sp_list->info.id));
    } else {                    /* expr */
      printf(" undefined expr: %s: nest=(", _expr_code[EXPR_CODE(sp_list->info.ep)]);
    }
    for(i=0; i<=sp_list->nest_level; i++){
      printf("%s", SYM_NAME(EXT_SYM(sp_list->nest_ext_id[i])));
      if(i!=sp_list->nest_level){
        printf(",");
      }
    }
    printf(")\n");
#endif

    /* fix ID & expr */
    if(sp_list->type == TYPE_ID){ /* ID */
      int is_exist = 0;
      if(EXT_PROC_CONT_EXT_SYMS(sp_list->nest_ext_id[0])){
        ID id;
        FOREACH_ID(id, EXT_PROC_ID_LIST(EXT_PROC_CONT_EXT_SYMS(sp_list->nest_ext_id[0]))){
          if(ID_CLASS(id) == CL_PROC){
            if(ID_SYM(id) == ID_SYM(sp_list->info.id)){
              is_exist = 1;
              break;
            }
          }
        }
      }
      if(!is_exist){
        for(i=sp_list->nest_level-1; i>=0 && !is_exist; i--){
          ID id;
          FOREACH_ID(id, EXT_PROC_ID_LIST(sp_list->nest_ext_id[i])){
            if(ID_SYM(id) == ID_SYM(sp_list->info.id)){
              is_exist = 1;
              break;
            }
          }
        }
      }
      if(is_exist){
        ID id, prev=NULL;
        FOREACH_ID(id, EXT_PROC_ID_LIST(sp_list->nest_ext_id[sp_list->nest_level])){
          if(id == sp_list->info.id){
            if(prev){
              prev->next = id->next;
            } else {
              EXT_PROC_ID_LIST(sp_list->nest_ext_id[sp_list->nest_level]) = id->next;
            }
            free(id);
            sp_list = unlink_sp_list(sp_list);
            break;
          }
          prev = id;
        }
      }

    } else {                    /* expr */
      if(EXPR_CODE(sp_list->info.ep) == IDENT || EXPR_CODE(sp_list->info.ep) == F_VAR){
        int is_exist = 0;
        for(i=sp_list->nest_level; i>=0 && !is_exist; i--){
          ID id;
          FOREACH_ID(id, EXT_PROC_ID_LIST(sp_list->nest_ext_id[i])){
            if(ID_SYM(id) == EXPR_SYM(sp_list->info.ep)){
              EXPV_TYPE(sp_list->info.ep) = ID_TYPE(id);
              sp_list = unlink_sp_list(sp_list);
              is_exist = 1;
              break;
            }
          }
        }
      }
    }
  }

  return second_pass_clean();
}
