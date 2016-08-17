/**
 * \file F-dump.c
 */

#include "F-front.h"

static void     print_string_constant _ANSI_ARGS_((FILE *fp, char *str));

char *basic_type_names[] = BASIC_TYPE_NAMES;
char *name_class_names[] = NAME_CLASS_NAMES;
char *proc_class_names[] = PROC_CLASS_NAMES;
char *storage_class_names[] = STORAGE_CLASS_NAMES;
char *control_type_names[] = CONTROL_TYPE_NAMES;


static void
print_string_constant(fp, str)
    FILE *fp;
    char *str;
{
    if (str == NULL || str[0] == '\0') {
        fprintf(fp, " \"\")");
        return;
    }
    fprintf(fp, " \"");
    while (*str != '\0') {
        if (*str < 0x20 || *str == '\\' || *str == '"' || *str >= 0x7F) {
            fprintf(fp, "\\%03o", *str);
        } else {
            fprintf(fp, "%c", *str);
        }
        str++;
    }
    fprintf(fp, "\")");
    return;
}


/* tree print routine */
static void
expv_output_rec(v,l,fp)
    expv v;
    int l;      /* indent level */
    FILE *fp;
{
    int i;
    struct list_node *lp;
    TYPE_DESC tp;
    char *expvKWName = NULL;

    /* indent */
    for(i = 0; i < l; i++) fprintf(fp,"  ");

    if(v == NULL){
        /* special case */
        fprintf(fp,"()");
        return;
    }

    /* XCODE NAME */
    fprintf(fp,"(%s",EXPR_CODE_NAME(EXPV_CODE(v)));
    tp = EXPV_TYPE(v);
    if(tp != NULL) {
        fprintf(fp,":");
        print_type(tp,fp,TRUE);
    }

    /* EXPV_KWOPT_NAME */
    expvKWName = (char *)EXPV_KWOPT_NAME(v);
    if (expvKWName != NULL && *expvKWName != '\0') {
        fprintf(fp, ":{keyword '%s'}", expvKWName);
    }

    if(EXPR_CODE_IS_TERMINAL(EXPV_CODE(v))){
        switch(EXPV_CODE(v)){
        case IDENT:
        case F_VAR:
        case F_PARAM:
        case F_FUNC:
            fprintf(fp," %s)",SYM_NAME(EXPV_NAME(v)));
            break;

        case INT_CONSTANT:
            fprintf(fp," 0x"OMLL_XFMT")",EXPV_INT_VALUE(v));
            break;

        case ID_LIST:
            if (EXPV_ANY(void *, v) != NULL) {
                ID id;
                int i = 0;
                fprintf(fp, ":{");
                FOREACH_ID(id, EXPV_ANY(ID, v)) {
                    if(i++ > 0)
                        fprintf(fp, ",");
                    fprintf(fp, "%s", ID_NAME(id));
                }
                fprintf(fp,"})");
            } else {
                fprintf(fp,")");
            }
            break;

        case STRING_CONSTANT:
            print_string_constant(fp,EXPV_STR(v));
            break;

        case FLOAT_CONSTANT:
            fprintf(fp," %Lf)", EXPV_FLOAT_VALUE(v));
            break;
        case ERROR_NODE:
            fprintf(fp,")");
            break;
        default:
            fprintf(fp," UNKNOWN)");
            break;
        }
        return;
    }

    if(EXPV_LIST(v) != NULL){
        if(l < 0) fprintf(fp," ");
        else {
            fprintf(fp,"\n");
            l++;
        }
        for(lp = EXPV_LIST(v); lp != NULL; lp=lp->l_next){
            expv_output_rec(lp->l_item,l, fp);
            if(lp->l_next != NULL){
                if(l < 0) fprintf(fp," ");
                else fprintf(fp,"\n");
            }
        }
    }
    fprintf(fp,")");
}


/* for debug */
static void
print_ID(id,fp,rec)
    ID id;
    FILE *fp;
    int rec;
{
    fprintf(fp,"'%s',class=%s,",ID_NAME(id),name_class_name(ID_CLASS(id)));
    fprintf(fp,"type=");
    print_type(ID_TYPE(id),fp,rec);
    switch(ID_CLASS(id)){
    case CL_PROC:
        fprintf(fp,",proc_class=%s",proc_class_name(PROC_CLASS(id)));
    default: {}
    }
    fprintf(fp,"\n");
}

void
print_IDs(ip,fp,rec)
    ID ip;
    FILE *fp;
    int rec;
{
    fprintf(fp, "# ID dump by %s\n", __func__);
    for( ; ip != NULL; ip = ID_NEXT(ip)) print_ID(ip,fp,rec);
    fflush(fp);
}

static void
print_EXT_ID(EXT_ID ep, FILE *fp)
{
    fprintf(fp, "'%s',tag=%s",SYM_NAME(EXT_SYM(ep)),
            storage_class_name(EXT_TAG(ep)));
    print_type(EXT_PROC_TYPE(ep), fp, FALSE);
    fprintf(fp,"\n");
}

void
print_EXT_IDs(EXT_ID extids, FILE *fp)
{
    EXT_ID ep;
    fprintf(fp, "# EXT_ID dump by %s\n", __func__);
    FOREACH_EXT_ID(ep, extids) print_EXT_ID(ep, fp);
    fflush(fp);
}

void
print_interface_IDs(ID id, FILE *fd) {
    EXT_ID eId;
    for (; id != NULL; id = ID_NEXT(id)) {
        if (ID_CLASS(id) == CL_PROC && PROC_CLASS(id) == P_EXTERNAL) {
            eId = PROC_EXT_ID(id);
            if (eId != NULL && EXT_PROC_CLASS(eId) == EP_INTERFACE &&
                EXT_PROC_INTR_DEF_EXT_IDS(eId) != NULL) {
                fprintf(fd, "# interface '%s' consists of:\n",
                        ID_NAME(id));
                print_EXT_IDs(EXT_PROC_INTR_DEF_EXT_IDS(eId), fd);
            }
        }
    }
}

void
print_types(tp,fp)
    TYPE_DESC tp;
    FILE *fp;
{
    fprintf(fp, "# TYPE dump by %s\n", __func__);
    for( ; tp != NULL; tp = TYPE_LINK(tp)) print_type(tp,fp,TRUE);
    fflush(fp);
}

void
type_output(tp,fp)
    TYPE_DESC tp;
    FILE *fp;
{
    fprintf(fp, "# TYPE dump by %s\n", __func__);
    print_types(tp, fp);
    fputs("\n", fp);
}

void
type_dump(TYPE_DESC tp)
{
    type_output(tp, stderr);
}


void
print_type(TYPE_DESC tp, FILE *fp, int recursive)
{
    if(tp == NULL){
        fprintf(fp,"{<NULL>}");
        return;
    }
    if(IS_STRUCT_TYPE(tp) && TYPE_REF(tp) == NULL){
        if(TYPE_TAGNAME(tp))
            fprintf(fp,"{type(%s):",SYM_NAME(ID_SYM(TYPE_TAGNAME(tp))));
        else
            fprintf(fp,"{type(<NULL>):"); // tagname may be private
        fprintf(fp,"\n");
        if(recursive == TRUE) {
            print_IDs(TYPE_MEMBER_LIST(tp),fp,FALSE);
            fprintf(fp,"}\n");
        }
    } else if(TYPE_N_DIM(tp) != 0){
        fprintf(fp,"{array(dim=%d):",TYPE_N_DIM(tp));
        print_type(TYPE_REF(tp),fp,recursive);
        fprintf(fp,"}");
    } else if(TYPE_REF(tp)){
        fprintf(fp,"{basic:");
        print_type(TYPE_REF(tp),fp,recursive);
        fprintf(fp,"}");
    } else {
        if(TYPE_BASIC_TYPE(tp) == TYPE_CHAR)
            fprintf(fp,"{character(%d)",TYPE_CHAR_LEN(tp));
        else
            fprintf(fp,"{%s",basic_type_name(TYPE_BASIC_TYPE(tp)));
        if(TYPE_KIND(tp))
            fprintf(fp,":kind="OMLL_DFMT, EXPV_INT_VALUE(TYPE_KIND(tp)));
        fprintf(fp,"}");
    }
}


/**
 * @brief
 * dump expv tree for debug
 */
void
expv_output(x,fp)
    expv x;
    FILE *fp;
{
    fprintf(fp, "# EXPV dump by %s\n", __func__);
    expv_output_rec(x, 0, fp);
    fprintf(fp, "\n");
}


/* tree print routine */
static void
expr_print_rec(x,l,fp)
     expr x;
     int l;
     FILE *fp;
{
    int i;
    struct list_node *lp;
    
    /* indent */
    for(i = 0; i < l; i++) fprintf(fp,"    ");
    
    if(x == NULL)
      {
          /* special case */
          fprintf(fp,"<NULL>");
          return;
      }
    
    fprintf(fp,"(%s",EXPR_CODE_NAME(EXPR_CODE(x)));

    fprintf(fp,":%d",
            (EXPR_LINE(x) != NULL) ? EXPR_LINE_NO(x) : 0);

    if(EXPR_CODE_IS_TERMINAL(EXPR_CODE(x))){
        switch(EXPR_CODE(x)){
        case IDENT:
            fprintf(fp, " \"%s\")", SYM_NAME(EXPR_SYM(x)));
            return;
        case STRING_CONSTANT:
            fprintf(fp," \"%s\")",EXPR_STR(x));
            return;
        case INT_CONSTANT:
            fprintf(fp," "OMLL_DFMT")",EXPR_INT(x));
            return;
        case FLOAT_CONSTANT:
            fprintf(fp," %Lf)",EXPR_FLOAT(x));
            return;
        case BASIC_TYPE_NODE:
            fprintf(fp," <%s>)",basic_type_name(EXPR_TYPE(x)));
            return;
        case ID_LIST:
            fprintf(fp, " \"%s\" '%s')",
                    ID_NAME(EXPV_ANY(ID, x)),
                    storage_class_names[(int)ID_STORAGE(EXPV_ANY(ID, x))]);
            return;
        default:
            fprintf(fp," "OMLL_DFMT")",EXPR_INT(x));
        }
        return;
    }

    /* list */
    if((lp = EXPR_LIST(x)) == NULL){
        fprintf(fp,")");
        return;
    }
    for(/* */; lp != NULL; lp = LIST_NEXT(lp)){
        fprintf(fp,"\n");
        expr_print_rec(LIST_ITEM(lp),l+1,fp);
    }
    fprintf(fp,")");
}


/**
 * @brief
 * dump expr tree for debug
 */
void
expr_print(x, fp)
     expr x;
     FILE *fp;
{
    fprintf(fp, "# EXPR dump by %s\n", __func__);
    expr_print_rec(x, 0, fp);
    fprintf(fp,"\n");
}


void
expr_print_indent(expr x, int l, FILE *fp) {
    expr_print_rec(x, l, fp);
}


void
print_controls(fp)
    FILE *fp;
{
    int i;
    fprintf(fp, "#ctls: levels=%d\n", CURRENT_BLK_LEVEL);

    for(i = 0; i < CURRENT_BLK_LEVEL; ++i) {
        CTL *ctl = &ctls[i];
        fprintf(fp, "  [%d] : %s\n",
            i,
            control_type_name(ctl->ctltype));
    }
}


/*
 * for debug
 */
void
expr_dump(expr x)
{
    expr_print_rec(x, 0, stderr);
    fprintf(stderr, "\n");
}

void
expv_dump(expv x)
{
    expv_output_rec(x, 0, stderr);
    fprintf(stderr, "\n");
}


char *
basic_type_name(t)
    BASIC_DATA_TYPE t;
{
    return basic_type_names[(int)t];
}


char *
name_class_name(c)
    enum name_class c;
{
    return name_class_names[(int)c];
}


char *
proc_class_name(c)
    enum proc_class c;
{
    return proc_class_names[(int)c];
}


char *
storage_class_name(c)
    enum storage_class c;
{
    return storage_class_names[(int)c];
}


char *
control_type_name(c)
    enum control_type c;
{
    return control_type_names[(int)c];
}

