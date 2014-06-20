/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-expr.h
 * - defines AST element types.
 * - declare functions for AST operation.
 */

#ifndef _C_EXPR_H_
#define _C_EXPR_H_

#include <stdio.h>
#include <sys/types.h>

#include "ccol.h"
#include "c-cmn.h"
#include "c-error.h"

/*
 * Enums
 */

/**
 * \brief
 * CExpr node types
 */
typedef enum {
    ET_TERMINAL,
    ET_LIST,
    ET_BINARYOPE,
    ET_UNARYOPE
} CExprTypeEnum;

#define CExprTypeEnumNamesDef {\
    "TERMINAL",\
    "LIST",\
    "BINNARYOPE",\
    "UNARYOPE"\
}

#include "c-exprcode.h"

/**
 * \brief
 * strage specifiers
 */
typedef enum {
    SS_UNDEF,
    SS_STATIC,
    SS_AUTO,
    SS_EXTERN,
    SS_REGISTER,
    SS_TYPEDEF,
    SS_THREAD,
    SS_END
} CSCSpecEnum;

#define CSCSpecEnumNamesDef {\
    "UNDEF",\
    "static",\
    "auto",\
    "extern",\
    "register",\
    "typedef",\
    "thread"\
}

/**
 * \brief
 * type specifiers (yylval)
 */
typedef enum {
    TS_UNDEF,
    TS_VOID,
    TS_UNSIGNED,
    TS_SIGNED,
    TS_CHAR,
    TS_WCHAR,
    TS_SHORT,
    TS_INT,
    TS_LONG,
    TS_FLOAT,
    TS_DOUBLE,
    TS_BOOL,
    TS_COMPLEX,
    TS_IMAGINARY,
    TS_END
} CTypeSpecEnum;

#define CTypeSpecEnumNamesDef {\
    "UNDEF",\
    "VOID",\
    "UNSIGNED",\
    "SIGNED",\
    "CHAR",\
    "WCHAR",\
    "SHORT",\
    "INT",\
    "LONG",\
    "FLOAT",\
    "DOUBLE",\
    "BOOL",\
    "COMPLEX",\
    "IMAGINARY"\
}


/**
 * \brief
 * type qualifiers
 */
typedef enum {
    TQ_UNDEF,
    TQ_CONST,
    TQ_VOLATILE,
    TQ_RESTRICT,
    TQ_INLINE,
    TQ_END
} CTypeQualEnum;


#define CTypeQualEnumNamesDef {\
    "UNDEF",\
    "const",\
    "volatile",\
    "restrict",\
    "inline"\
}


/**
 * \brief
 * cardinal numbers for constant number
 */
typedef enum {
    CD_UNDEF,
    CD_BIN,
    CD_OCT,
    CD_DEC,
    CD_HEX,
    CD_END
} CCardinalEnum;

#define CCardinalEnumNamesDef {\
    "UNDEF",\
    "BIN",\
    "OCT",\
    "DEC",\
    "HEX",\
}


/**
 * \brief
 * character types for constan character
 */
typedef enum {
    CT_UNDEF,
    CT_MB,
    CT_WIDE,
    CT_END
} CCharTypeEnum;

#define CCharTypeEnumNamesDef {\
    "UNDEF",\
    "MB",\
    "WIDE"\
}


/**
 * \brief
 * assign operations (yylval)
 */
typedef enum {
    AS_UNDEF,
    AS_EQ,
    AS_RSHIFT_EQ,
    AS_LSHIFT_EQ,
    AS_PLUS_EQ,
    AS_MINUS_EQ,
    AS_MUL_EQ,
    AS_DIV_EQ,
    AS_MOD_EQ,
    AS_AND_EQ,
    AS_XOR_EQ,
    AS_OR_EQ,
    AS_END
} CAssignEnum;

#define CAssignEnumNamesDef {\
    "UNDEF",\
    "EQ",\
    "RSHIFT_EQ",\
    "LSHIFT_EQ",\
    "PLUS_EQ",\
    "MINUS_EQ",\
    "MUL_EQ",\
    "DIV_EQ",\
    "MOD_EQ",\
    "AND_EQ",\
    "XOR_EQ",\
    "OR_EQ"\
}

#define CAssignEnumToExprCodeEnumDef {\
    EC_ERROR_NODE,\
    EC_ASSIGN,\
    EC_ASSIGN_RSHIFT,\
    EC_ASSIGN_LSHIFT,\
    EC_ASSIGN_PLUS,\
    EC_ASSIGN_MINUS,\
    EC_ASSIGN_MUL,\
    EC_ASSIGN_DIV,\
    EC_ASSIGN_MOD,\
    EC_ASSIGN_BIT_AND,\
    EC_ASSIGN_BIT_XOR,\
    EC_ASSIGN_BIT_OR,\
}


/**
 * \brief
 * unary operators (yylval)
 */
typedef enum {
    UO_UNDEF,
    UO_ADDR_OF,
    UO_PLUS,
    UO_MINUS,
    UO_PLUSPLUS,
    UO_MINUSMINUS,
    UO_BIT_NOT,
    UO_LOG_NOT
} CUnaryOpeEnum;

#define CUnaryOpeEnumNamesDef {\
    "UNDEF",\
    "ADDR_OF",\
    "PLUS",\
    "MINUS",\
    "PLUSPLUS",\
    "MINUSMINUS",\
    "BIT_NOT",\
    "LOG_NOT"\
}

#define CUnaryOpeEnumToExprCodeEnumDef {\
    EC_ERROR_NODE,\
    EC_ADDR_OF,\
    EC_NULL_NODE,\
    EC_UNARY_MINUS,\
    EC_PRE_INCR,\
    EC_PRE_DECR,\
    EC_BIT_NOT,\
    EC_LOG_NOT\
}


/**
 * \brief
 * basic types
 */
typedef enum {
    // order of enumerators must not be changed
    BT_UNDEF,
    BT_VOID,
    BT_BOOL,
    BT_CHAR,
    BT_UNSIGNED_CHAR,
    BT_SHORT,
    BT_UNSIGNED_SHORT,
    BT_INT,
    BT_UNSIGNED_INT,
    BT_LONG,
    BT_UNSIGNED_LONG,
    BT_WCHAR,
    BT_LONGLONG,
    BT_UNSIGNED_LONGLONG,
    BT_FLOAT,
    BT_DOUBLE,
    BT_LONGDOUBLE,
    // following constants must be located in tail
    BT_FLOAT_IMAGINARY,
    BT_DOUBLE_IMAGINARY,
    BT_LONGDOUBLE_IMAGINARY,
    BT_FLOAT_COMPLEX,
    BT_DOUBLE_COMPLEX,
    BT_LONGDOUBLE_COMPLEX,
    BT_END
} CBasicTypeEnum;

#define CBasicTypeEnumNamesDef {\
    "UNDEF",\
    "VOID",\
    "BOOL",\
    "CHAR",\
    "UNSIGNED_CHAR",\
    "SHORT",\
    "UNSIGNED_SHORT",\
    "INT",\
    "UNSIGNED_INT",\
    "LONG",\
    "UNSIGNED_LONG",\
    "WCHAR",\
    "LONGLONG",\
    "UNSIGNED_LONGLONG",\
    "FLOAT",\
    "DOUBLE",\
    "LONGDOUBLE",\
    "FLOAT_IMAGINARY",\
    "DOUBLE_IMAGINARY",\
    "LONGDOUBLE_IMAGINARY",\
    "FLOAT_COMPLEX",\
    "DOUBLE_COMPLEX",\
    "LONGDOUBLE_COMPLEX",\
}

#define CBasicTypeEnumXcodeDef {\
    "UNDEF",\
    "void",\
    "bool",\
    "char",\
    "unsigned_char",\
    "short",\
    "unsigned_short",\
    "int",\
    "unsigned",\
    "long",\
    "unsigned_long",\
    "wchar_t",\
    "long_long",\
    "unsigned_long_long",\
    "float",\
    "double",\
    "long_double",\
    "float_imaginary",\
    "double_imaginary",\
    "long_double_imaginary",\
    "float_complex",\
    "double_complex",\
    "long_double_complex",\
}


#define CBasicTypeEnumCcodeDef {\
    "UNDEF",\
    "void",\
    "bool",\
    "char",\
    "unsigned char",\
    "short",\
    "unsigned short",\
    "int",\
    "unsigned",\
    "long",\
    "unsigned long",\
    "wchar_t",\
    "long long",\
    "unsigned long long",\
    "float",\
    "double",\
    "long double",\
    "float imaginary",\
    "double imaginary",\
    "long double imaginary",\
    "float complex",\
    "double complex",\
    "long double complex",\
}

/**
 * \brief
 * number value kinds
 */
typedef enum {
    NK_LL,  //! long long (all signed intger)
    NK_ULL, //! unsigned long long (all unsigned integer)
    NK_LD   //! long double (all floating point number)
} CNumValueKind;


/**
 * \brief
 * symbol types
 */
typedef enum {
    ST_UNDEF,
    ST_TYPE,
    ST_FUNC,
    ST_VAR,
    ST_PARAM,
    ST_ENUM,
    ST_TAG,
    ST_LABEL,
    ST_GCC_LABEL,
    ST_FUNCID,
    ST_MEMBER,
    ST_GCC_BUILTIN,
    ST_GCC_ASM_IDENT,
    ST_END
} CSymbolTypeEnum;

#define CSymbolTypeEnumNamesDef {\
    "UNDEF",\
    "TYPE",\
    "FUNC",\
    "VAR",\
    "PARAM",\
    "ENUM",\
    "TAG",\
    "LABEL",\
    "GCC_LABEL",\
    "FUNCID", \
    "MEMBER", \
    "GCC_BUILTIN", \
    "GCC_ASM_IDENT", \
}

/**
 * \brief
 * symbol table groups
 */
typedef enum {
    STB_IDENT,
    STB_TAG,
    STB_LABEL
} CSymbolTableGroupEnum;

/**
 * \brief
 * directive types
 */
typedef enum {
    DT_UNDEF,
    DT_PRAGMA,
    DT_IDENT,
    DT_WARNING,
    DT_ERROR,
    DT_END
} CDirectiveTypeEnum;

#define CDirectiveTypeEnumNamesDef {\
    "UNDEF",\
    "PRAGMA",\
    "IDENT",\
    "WARNING",\
    "ERROR",\
}

/**
 * \brief
 * type descriptor types
 */
typedef enum {
    TD_UNDEF,
    TD_BASICTYPE,
    TD_STRUCT,
    TD_UNION,
    TD_ENUM,
    TD_POINTER,
    TD_ARRAY,
    TD_FUNC,
    TD_FUNC_OLDSTYLE, // temporary use at compiling
    TD_TYPEREF,
    TD_DERIVED,
    TD_GCC_TYPEOF,
    TD_GCC_BUILTIN, // type specified id
    TD_GCC_BUILTIN_ANY, // unknown type id
    TD_COARRAY,
    TD_END
} CTypeDescKindEnum;

#define CTypeDescKindEnumNamesDef {\
    "UNDEF",\
    "BASICTYPE",\
    "STRUCT",\
    "UNION",\
    "ENUM",\
    "POINTER",\
    "ARRAY",\
    "FUNC",\
    "FUNC_OLDSTYLE",\
    "TYPEREF",\
    "DERIVED",\
    "GCC_TYPEOF",\
    "GCC_BUILTIN",\
    "GCC_BUILTIN_ANY",\
    "COARRAY",\
}

/**
 * \brief
 * contexts of declarator
 */
typedef enum CDeclaratorContext
{
    DC_IN_ANY,
    DC_IN_FUNC_DEF,
    DC_IN_MEMBER_DECL,
    DC_IN_PARAMS,
    DC_IN_TYPENAME,
    DC_IN_NO_MEMBER_DECL,
} CDeclaratorContext;


/**
 * \brief
 * line number information
 */
typedef struct CLineNumInfo {

    //! line number calculated with line directive
    int    ln_lineNum;
    //! column number (not acculate)
    int    ln_column;
    //! raw line number in input file
    int    ln_rawLineNum;
    //! file ID in file ID table
    int    ln_fileId;

} CLineNumInfo;

/*
 * Expression Node Struct Kind
 */

#include "c-gcctype.h"

typedef struct CExpr CExpr;

/**
 * \brief
 * CExpr struct types
 */
typedef enum {
    STRUCT_CExpr_UNDEF,
    STRUCT_CExprOfArrayDecl,
    STRUCT_CExprOfBinaryNode,
    STRUCT_CExprOfCharConst,
    STRUCT_CExprOfDirective,
    STRUCT_CExprOfErrorNode,
    STRUCT_CExprOfGeneralCode,
    STRUCT_CExprOfList,
    STRUCT_CExprOfNull,
    STRUCT_CExprOfNumberConst,
    STRUCT_CExprOfStringConst,
    STRUCT_CExprOfSymbol,
    STRUCT_CExprOfTypeDesc,
    STRUCT_CExprOfUnaryNode,
    STRUCT_CExpr_END
} CExprStructEnum;

#define CExprStructEnumNamesDef {\
    "CExpr_UNDEF",\
    "CExprOfArrayDecl",\
    "CExprOfBinaryNode",\
    "CExprOfCharConst",\
    "CExprOfDirective",\
    "CExprOfErrorNode",\
    "CExprOfGeneralCode",\
    "CExprOfList",\
    "CExprOfNull",\
    "CExprOfNumberConst",\
    "CExprOfStringConst",\
    "CExprOfSymbol",\
    "CExprOfTypeDesc",\
    "CExprOfUnaryNode",\
}

/**
 * Expression Node
 */

struct CSymbolTable;

/**
 * \brief
 * CExpr's header struct.
 * structs exteded fron CExpr have this struct at a first member.
 */
typedef struct CExprCommon {

    //! reference count
    int                     e_refCount;
    //! struct type 
    CExprStructEnum         e_struct;
    //! node type
    CExprCodeEnum           e_exprCode;
    //! line number info.
    CLineNumInfo            e_lineNumInfo;
    //! reference to symbol table
    struct CSymbolTable     *e_symTab;
    //! reference to expression's type description
    struct CExprOfTypeDesc  *e_exprsType;
    //! reference to parent node
    CExpr                   *e_parentExpr;
    //! gcc attributes at prefix position
    CExpr                   *e_gccAttrPre;
    //! gcc attributes at postfix position
    CExpr                   *e_gccAttrPost;
    //! is error node
    unsigned int            e_isError       : 1;
    //! is constant value
    unsigned int            e_isConstValue  : 1;
    //! is checked if constant value
    unsigned int            e_isConstValueChcked  : 1;
    //! is specified __extension__
    unsigned int            e_gccExtension  : 1;
    //! use in initializer for check duplicated initialization
    unsigned int            e_isCompleted   : 1;
    //! has initializer (for EC_LDECLARATOR, EC_DECLARATOR, EC_IDENT)
    unsigned int            e_hasInit       : 1;
    //! for compile()
    unsigned int            e_isCompiled    : 1;
    //! is converted from original syntax
    unsigned int            e_isConverted   : 1;
    //! is generated in convertSyntax()
    unsigned int            e_isGenerated   : 1;
    //! is gcc extended syntax
    unsigned int            e_isGccSyntax   : 1;
    //! if or not to delete in convertSyntax()
    unsigned int            e_isDeleting    : 1;

} CExprCommon;


typedef CExprCommon *cec_t;

#define EXPR_ISERROR(x)             EXPR_C(x)->e_isError
#define EXPR_GCCEXTENSION(x)        EXPR_C(x)->e_gccExtension
#define EXPR_ISCOMPILED(x)          EXPR_C(x)->e_isCompiled
#define EXPR_ISCONSTVALUE(x)        EXPR_C(x)->e_isConstValue
#define EXPR_ISCONSTVALUECHECKED(x) EXPR_C(x)->e_isConstValueChcked
#define EXPR_ISCONVERTED(x)         EXPR_C(x)->e_isConverted
#define EXPR_ISGCCSYNTAX(x)         EXPR_C(x)->e_isGccSyntax
#define EXPR_ISCOMPLETED(x)         EXPR_C(x)->e_isCompleted
#define EXPR_ISDELETING(x)          EXPR_C(x)->e_isDeleting

/**
 * \brief
 * CExpr for general list
 */
typedef struct CExprOfList {

    //! common header
    CExprCommon         e_common;
    //! member nodes
    CCOL_DList          e_dlist;
    //! EC_INITIALIZERS : member designator's symbol
    struct CExprOfSymbol *e_symbol;

    //! annoted aux info (extended for pragma, ...)
    int e_aux;
    struct CExpr *e_aux_info;

} CExprOfList;


/**
 * \brief
 * CExpr for symbol
 */
typedef struct CExprOfSymbol {

    //! common header
    CExprCommon             e_common;
    //! symbol string
    char                    *e_symName;
    //! symbol type
    CSymbolTypeEnum         e_symType;
    //! for enumerator initial value
    CExpr                   *e_valueExpr;
    /**
     * - e_symType = ST_FUNC : temporary use for funcdef old style
     * - e_symType = ST_MEMBER : pointer type of member ref
     */
    struct CExprOfTypeDesc  *e_headType;
    //! list of coarray dimensions (ID=284)
    //CExpr                   *e_codimensions;
    //! reference to declarator
    CExpr                   *e_declrExpr;
    //! order in SymbolTable
    int                     e_putOrder;
    //! is global symbol 
    unsigned int            e_isGlobal : 1;
    //! if specified enumerator initial value
    unsigned int            e_isEnumInited : 1;
    //! is symbol of gcc label declaration
    unsigned int            e_isGccLabelDecl : 1;
    //! enumerator value is constant but unreducable
    unsigned int            e_isConstButUnreducable : 1;

} CExprOfSymbol;

#define EXPR_SYMBOL_VALUEEXPR(x) ((x)->e_valueExpr ?\
    (x)->e_valueExpr : (x)->e_valueExprRef)

/**
 * \brief
 * number value
 */
typedef union CNumValue {
    //! floatin point number
    long double ld;
    //! signed integer
    long long ll;
    //! unsigned integer
    unsigned long long ull;
} CNumValue;


/**
 * \brief
 * number value with type information
 */
typedef struct CNumValueWithType
{
    //! basic type
    CBasicTypeEnum  nvt_basicType;
    //! kind of number value
    CNumValueKind   nvt_numKind;
    //! value of number
    CNumValue       nvt_numValue;
    //! is constant but mutable at compile time
    unsigned int    nvt_isConstButMutable : 1;
    //! is constant but unreducable value
    unsigned int    nvt_isConstButUnreducable : 1;
} CNumValueWithType;


/**
 * \brief
 * CExpr for number constant
 */
typedef struct CExprOfNumberConst {

    //! common header
    CExprCommon         e_common;
    //! basic type
    CBasicTypeEnum      e_basicType;
    //! cardinal number
    CCardinalEnum       e_cardinal;
    //! original token
    char                *e_orgToken;
    //! value of number
    CNumValue           e_numValue;

} CExprOfNumberConst;


/**
 * \brief
 * CExpr for character constant
 */
typedef struct CExprOfCharConst {

    //! common header
    CExprCommon         e_common;
    //! original token
    char                *e_orgToken;
    //! formatted token
    char                *e_token;
    //! character type
    CCharTypeEnum       e_charType;

} CExprOfCharConst;


/**
 * \brief
 * CExpr for string constant
 */
typedef struct CExprOfStringConst {

    //! common header
    CExprCommon         e_common;
    //! original token
    char                *e_orgToken;
    //! character type
    CCharTypeEnum       e_charType;
    //! number of characters
    unsigned int        e_numChars;

} CExprOfStringConst;


/**
 * \brief
 * CExpr for simple code value
 */
typedef struct CExprOfGeneralCode {

    //! common header
    CExprCommon         e_common;

    /**
     * \brief code value.
     *
     * available values:
     *   - EC_TYPEQUAL:    CTypeQualEnum 
     *   - EC_TYPESPEC:    CTypeSpecEnum
     *   - EC_SCSPEC:      CSCSpecEnum
     */
    int                 e_code;

} CExprOfGeneralCode;


/**
 * \brief
 * CExpr of one-child node
 */
typedef struct CExprOfUnaryNode {

    //! common header
    CExprCommon         e_common;
    //! child node
    CExpr               *e_node;

} CExprOfUnaryNode;


/**
 * \brief
 * CExpr of two-children nodes
 */
typedef struct CExprOfBinaryNode {

    //! common header
    CExprCommon         e_common;
    //! children nodes
    CExpr               *e_nodes[2];

    //! for EC_GCC_ATTR_ARG, attribute kind
    CGccAttrKindEnum    e_gccAttrKind;
    //! for EC_GCC_ATTR_ARG, reference to attribute info
    CGccAttrInfo        *e_gccAttrInfo;
    //! for EC_GCC_ATTR_ARG, is ingnored attribute 
    unsigned int        e_gccAttrIgnored : 1;
    //! for EC_GCC_ATTR_ARG, is output attribute 
    unsigned int        e_gccAttrOutput : 1;
} CExprOfBinaryNode;


/**
 * \brief
 * CExpr for array declaration (using temporary)
 */
typedef struct CExprOfArrayDecl {

    //! common header
    CExprCommon         e_common;
    //! reference to type qualifer node
    CExpr               *e_typeQualExpr;
    //! reference to array length node
    CExpr               *e_lenExpr;
    //! is variable array length
    unsigned int        e_isVariable : 1;
    //! is qualified with 'static'
    unsigned int        e_isStatic : 1;

} CExprOfArrayDecl;


/**
 * \brief
 * CExpr for directive
 */
typedef struct CExprOfDirective {

    //! common header
    CExprCommon         e_common;
    //! directive type
    CDirectiveTypeEnum  e_direcType;
    //! directive name
    char                *e_direcName;
    //! directive arguments
    char                *e_direcArgs;

  // for OpenMP, e_direcType=DT_PRAGMA, e_direcName="omp"
  CExprOfList *e_direcInfo;

} CExprOfDirective;


/**
 * \brief
 * type qualifiers
 */
typedef struct
{
    //! is specified 'const'
    unsigned int    etq_isConst     : 1;
    //! is specified 'inline'
    unsigned int    etq_isInline    : 1;
    //! is specified 'volatile'
    unsigned int    etq_isVolatile  : 1;
    //! is specified 'restrict'
    unsigned int    etq_isRestrict  : 1;
} CTypeQual;

/**
 * \brief
 * storage classes
 */
typedef struct
{
    //! is specified 'auto'
    unsigned int    esc_isAuto      : 1;
    //! is specified 'static'
    unsigned int    esc_isStatic    : 1;
    //! is specified 'extern'
    unsigned int    esc_isExtern    : 1;
    //! is specified 'register'
    unsigned int    esc_isRegister  : 1;
    //! is specified '__thread' (gcc syntax)
    unsigned int    esc_isGccThread : 1;
} CStorageClass;

/**
 * \brief
 * type descriptor
 */
typedef struct CExprOfTypeDesc {

    //! common header
    CExprCommon         e_common;
    //! type description kind
    CTypeDescKindEnum   e_tdKind;
    //! TD_BASICTYPE: basic type
    CBasicTypeEnum      e_basicType;

    /**
     * \brief reference type descriptor or expression node
     *
     *  - TD_STRUCT:      EC_STRUCT_TYPE
     *  - TD_UNION:       EC_UNION_TYPE
     *  - TD_ENUM:        EC_ENUM_TYPE
     *  - TD_POINTER:     EC_TYPE_DESC
     *  - TD_ARRAY:       EC_TYPE_DESC
     *  - TD_FUNC:        EC_TYPE_DESC (for returning type)
     *  - TD_TYPEREF:     EC_IDENT
     *  - TD_GCC_TYPEOF:  EC_EXPRS
     *  - TD_COARRAY:     EC_TYPE_DESC
     */
    CExpr               *e_typeExpr;

    /**
     * \brief function parameter node
     *
     *  - TD_FUNC:        EC_PARAMS
     *  - TD_DERIVED:     EC_EXPRS, EC_IDENT, EC_TYPE_DESC
     *                    (original TD_TYPEREF/TD_GCC_TYPEOF's expression)
     *  - TD_ARRAY:       EC_TYPE_DESC
     *                    (pointer type of e_typeExpr)
    */
    CExpr               *e_paramExpr;

    /**
     * \brief bit field node
     *  - TD_STRUCT, TD_UNION: member's bit field
     */
    CExpr               *e_bitLenExpr;
    //! constant-folded bit field
    int                 e_bitLen;

    /**
     * \brief original type
     * 
     *  - TD_STRUCT:  type which has body definition
     *  - TD_UNION:   type which body definition
     *  - TD_DERIVED: reference type
     */
    struct CExprOfTypeDesc *e_refType;

    //! pre-declarared symbol type
    struct CExprOfTypeDesc *e_preDeclType;

    //! type ID and cotype ID for XcodeML (ID=284)
    char                *e_typeId;

    //! alignment of type
    int                 e_align;
    //! size of type
    int                 e_size;

    /**
     * \brief function name
     *
     *  EC_FUNC : symbol of function name
     */
    CExprOfSymbol       *e_symbol;

    //! array length, coarray dimension
    struct {
        //! length node (constant-folded)
        CExpr           *eln_lenExpr;
        //! original length node
        CExpr           *eln_orgLenExpr;
        //! constant-folded length
        unsigned int    eln_len;
        //! is length of '*'
        unsigned int    eln_isVariable  : 1;
        //! is flexible array in struct/union
        unsigned int    eln_isFlexible  : 1;
        //! is specified 'static'
        unsigned int    eln_isStatic    : 1;
        //! is specified 'const'
        unsigned int    eln_isConst     : 1;
        //! is specified 'volatie'
        unsigned int    eln_isVolatile  : 1;
        //! is specified 'restrict'
        unsigned int    eln_isRestrict  : 1;
    } e_len;

    //! type qualifiers
    CTypeQual           e_tq;

    //! storage classes
    CStorageClass       e_sc;

    //! is typedef declaration
    unsigned int        e_isTypeDef     : 1;
    //! is already defined
    unsigned int        e_isExist       : 1;
    //! TD_STRUCT, TD_UNION, TD_ENUM: has no member decl
    unsigned int        e_isNoMemDecl   : 1;
    //! TD_STRUCT, TD_UNION, TD_ENUM: is anonoymous tag
    unsigned int        e_isAnonTag     : 1;
    //! is anonoymous member
    unsigned int        e_isAnonMember  : 1;
    //! for ST_VAR
    unsigned int        e_isDefined     : 1;
    //! is type in s_constantTypeDescs or s_gccBuiltinTypes is used,
    // or is gcc builtin function.
    unsigned int        e_isUsed        : 1;
    //! added to s_exprsTypeDescList
    unsigned int        e_isExprsType   : 1;
    //! added to s_typeDescList
    unsigned int        e_isCollected   : 1;
    //! for compareType()
    unsigned int        e_isTemporary   : 1;
    //! for fixTypeDesc()
    unsigned int        e_isFixed       : 1;
    //! for fixTypeDesc()
    unsigned int        e_isCompiling   : 1;
    //! TD_FUNC : for __attribute__((const)) or const builtin function
    unsigned int        e_isGccConst    : 1;
    //! gcc attribute is duplicated to other type
    unsigned int        e_isGccAttrDuplicated : 1;
    //! size/align is constant but included unreducable expression
    unsigned int        e_isSizeUnreducable : 1;
    //! complete type but size is zero
    unsigned int        e_isSizeZero    : 1;
    //! for Output XcodeML : disable to output typeId
    unsigned int        e_isNoTypeId    : 1;
    //! for Output XcodeML : is marked for any purpose
    unsigned int        e_isMarked      : 1;
    //! for Output XcodeML : same type exists in typeTable
    unsigned int        e_isDuplicated  : 1;
    //! for Output XcodeML : has different qualifer/gcc attrs from refType
    unsigned int        e_isDifferentQaulifierFromRefType : 1;

} CExprOfTypeDesc;


/**
 * \brief
 * CExpr of error node
 */
typedef struct CExprOfErrorNode {

    //! common header
    CExprCommon         e_common;
    //! error node
    CExpr               *e_nearExpr;

} CExprOfErrorNode;


/**
 * \brief
 * CExpr of null node
 */
typedef struct CExprOfNull {

    //! common header
    CExprCommon         e_common;

} CExprOfNull;


/**
 * \brief
 * node of Abstract Syntax Tree
 */
struct CExpr {

    union {
        CExprOfSymbol       e_symbol;
        CExprOfList         e_list;
        CExprOfNumberConst  e_numberConst;
        CExprOfCharConst    e_charConst;
        CExprOfStringConst  e_stringConst;
        CExprOfGeneralCode  e_generalCode;
        CExprOfUnaryNode    e_unaryNode;
        CExprOfBinaryNode   e_binaryNode;
        CExprOfArrayDecl    e_arrayDecl;
        CExprOfDirective    e_directive;
        CExprOfTypeDesc     e_typeDesc;
        CExprOfErrorNode    e_errorNode;
        CExprOfNull         e_null;
    } u;
};


/**
 * \brief
 * iterator for iterating CExprOfUnaryNode, CExprOfBinaryNode, CExprOfList
 */
typedef struct CExprIterator {
    //! child node
    CExpr *node;
    union {
        //! child node index for CExprOfUnaryNode, CExprOfBinaryNode
        int             index;
        //! child node for CExprOfList
        CCOL_DListNode  *listNode;
    } i;
} CExprIterator;


extern CExpr*                 exprSet(CExpr **plexpr, CExpr *rexpr);
extern CExpr*                 exprSet0(CExpr **plexpr, CExpr *rexpr);
extern void                   freeExpr(CExpr *expr);
extern CExpr*                 copyExpr(CExpr *expr);
extern CExprOfList*           allocExprOfList(CExprCodeEnum exprCode);
extern CExprOfList*           duplicateExprOfList(CExprOfList *src);
extern CExprOfList*           allocExprOfList1(CExprCodeEnum exprCode, CExpr *e1);
extern CExprOfList*           allocExprOfList2(CExprCodeEnum exprCode, CExpr *e1, CExpr *e2);
extern CExprOfList*           allocExprOfList3(CExprCodeEnum exprCode,
                                    CExpr *e1, CExpr *e2, CExpr *e3);
extern CExprOfList*           allocExprOfList4(CExprCodeEnum exprCode,
                                    CExpr *e1, CExpr *e2, CExpr *e3, CExpr *e4);
extern CExprOfNumberConst*    allocExprOfNumberConst(CExprCodeEnum exprCode,
                                    CBasicTypeEnum btype, CCardinalEnum c, char *orgToken);
extern CExprOfNumberConst*    allocExprOfNumberConst1(CNumValueWithType *nvt);
extern CExprOfNumberConst*    allocExprOfNumberConst2(long long n, CBasicTypeEnum bt);
extern CExprOfCharConst*      allocExprOfCharConst(CExprCodeEnum exprCode, char *orgToken,
                                  CCharTypeEnum charType);
extern CExprOfStringConst*    allocExprOfStringConst(CExprCodeEnum exprCode, char *orgToken,
                                  CCharTypeEnum charType);
extern CExprOfStringConst*    allocExprOfStringConst2(const char *orgToken);
extern CExprOfSymbol*         allocExprOfSymbol(CExprCodeEnum exprCode, char *token);
extern CExprOfSymbol*         allocExprOfSymbol2(const char *token);
extern CExprOfSymbol*         duplicateExprOfSymbol(CExprOfSymbol *sym);
extern CExprOfGeneralCode*    allocExprOfGeneralCode(CExprCodeEnum exprCode, int code);
extern CExprOfUnaryNode*      allocExprOfUnaryNode(CExprCodeEnum exprCode, CExpr *node);
extern CExprOfBinaryNode*     allocExprOfBinaryNode1(CExprCodeEnum exprCode, CExpr *node1, CExpr *node2);
extern CExprOfBinaryNode*     allocExprOfBinaryNode2(CExprCodeEnum exprCode, CExpr *node1, CExpr *node2);
extern CExprOfArrayDecl*      allocExprOfArrayDecl(CExpr *typeQualExpr, CExpr *sizeExpr);
extern CExprOfDirective*      allocExprOfDirective(CDirectiveTypeEnum type, char *name, char *args);
extern CExprOfTypeDesc*       allocExprOfTypeDesc(void);
extern CExprOfTypeDesc*       allocPointerTypeDesc(CExprOfTypeDesc *refType);
extern CExprOfTypeDesc*       allocDerivedTypeDesc(CExprOfTypeDesc *td);
extern void                   innerFreeExprOfTypeDesc(CExprOfTypeDesc *expr);
extern CExprOfTypeDesc*       duplicateExprOfTypeDesc(CExprOfTypeDesc *src);
extern CExprOfErrorNode*      allocExprOfErrorNode(CExpr *nearExpr);
extern CExprOfNull*           allocExprOfNull(void);
extern int                    exprRef(CExpr *expr);
extern int                    exprUnref(CExpr *expr);
extern void                   exprSetExprsType(CExpr *expr, CExprOfTypeDesc *td);
extern CExpr*                 exprSubArrayDimension(CExpr *el, CExpr *eu, CExpr *es);
extern CExpr*                 exprCoarrayRef(CExpr *prim, CExpr *dims);
extern int                    isSubArrayRef(CExpr *expr);
extern int                    isSubArrayRef2(CExpr *expr);

//! alloc CExprOfList with no child
#define exprList(c)                         ((CExpr*)allocExprOfList(c))
//! alloc CExprOfList with 1 child
#define exprList1(c, e1)                    ((CExpr*)allocExprOfList1(c, e1))
//! alloc CExprOfList with 2 children
#define exprList2(c, e1, e2)                ((CExpr*)allocExprOfList2(c, e1, e2))
//! alloc CExprOfList with 3 children
#define exprList3(c, e1, e2, e3)            ((CExpr*)allocExprOfList3(c, e1, e2, e3))
//! alloc CExprOfList with 4 children
#define exprList4(c, e1, e2, e3, e4)        ((CExpr*)allocExprOfList4(c, e1, e2, e3, e4))
//! alloc CExprOfUnary
#define exprUnary(c, e)                     ((CExpr*)allocExprOfUnaryNode(c, e))
//! 
//! alloc CExprOfBinary
#define exprBinary(c, e1, e2)               ((CExpr*)allocExprOfBinaryNode2(c, e1, e2))
//! alloc CExprOfArrayDecl
#define exprArrayDecl(typeExpr, sizeExpr)   ((CExpr*)allocExprOfArrayDecl(typeExpr, sizeExpr))
//! alloc CExprOfNull
#define exprNull()                          ((CExpr*)allocExprOfNull())

#define EXPR_ALLOC(type, p, exprCode) \
    type *p = XALLOC(type);\
    p->e_common.e_struct = STRUCT_##type;\
    p->e_common.e_exprCode = exprCode;\
    p->e_common.e_lineNumInfo = s_lineNumInfo;

#define EXPR_ALLOC_COPY(type, des, src) \
    type *des = XALLOC(type);\
    memcpy(des, src, sizeof(type));\
    innerCopyExprCommon((CExpr*)(des), (CExpr*)(src));

//! cast to CExprCommon
#define EXPR_C(x)                   ((CExprCommon*)(x))
//! increment reference count
#define EXPR_REF(x)                 exprRef((CExpr*)x)
//! decrement reference count
#define EXPR_UNREF(x)               exprUnref((CExpr*)x)
//! set child node
#define EXPR_SET(l, r)              exprSet((CExpr**)&(l), (CExpr*)(r))
//! set child node
#define EXPR_SET0(l, r)             exprSet0((CExpr**)&(l), (CExpr*)(r))
//! cast to CExprOfSymbol
#define EXPR_SYMBOL(x)              (&((CExpr*)(x))->u.e_symbol)
//! cast to CExprOfList
#define EXPR_L(x)                   (&((CExpr*)(x))->u.e_list)
//! get child list in CExprOfList
#define EXPR_DLIST(x)               (&((CExpr*)(x))->u.e_list.e_dlist)
//! cast to CExprOfNumberConst
#define EXPR_NUMBERCONST(x)         (&((CExpr*)(x))->u.e_numberConst)
//! cast to CExprOfCharConst
#define EXPR_CHARCONST(x)           (&((CExpr*)(x))->u.e_charConst)
//! cast to CExprOfStringConst
#define EXPR_STRINGCONST(x)         (&((CExpr*)(x))->u.e_stringConst)
//! cast to CExprOfGeneralCode
#define EXPR_GENERALCODE(x)         (&((CExpr*)(x))->u.e_generalCode)
//! cast to CExprOfUnaryNode
#define EXPR_U(x)                   (&((CExpr*)(x))->u.e_unaryNode)
//! cast to CExprOfBinaryNode
#define EXPR_B(x)                   (&((CExpr*)(x))->u.e_binaryNode)
//! cast to CExprOfArrayDecl
#define EXPR_ARRAYDECL(x)           (&((CExpr*)(x))->u.e_arrayDecl)
//! cast to CExprOfDirective
#define EXPR_DIRECTIVE(x)           (&((CExpr*)(x))->u.e_directive)
//! cast to CExprOfTypeDesc
#define EXPR_T(x)                   (&((CExpr*)(x))->u.e_typeDesc)
//! cast to CExprOfErrorNode
#define EXPR_ERRORNODE(x)           (&((CExpr*)(x))->u.e_errorNode)
//! get expression code
#define EXPR_CODE(x)                (EXPR_C(x)->e_exprCode)
//! get struct kind
#define EXPR_STRUCT(x)              (EXPR_C(x)->e_struct)
//! cast to CExprOfNull
#define EXPR_NULL(x)                (&((CExpr*)(x))->u.e_null)
//! is NULL or CExprOfNull
#define EXPR_ISNULL(x)              ((x) == NULL || EXPR_C(x)->e_struct == STRUCT_CExprOfNull)
//! cast to CExpr from CCOL_DListNode's data
#define EXPR_L_DATA(x)              ((CExpr*)CCOL_DL_DATA(x))
//! get list size of CExprOfList
#define EXPR_L_SIZE(l)              CCOL_DL_SIZE(EXPR_DLIST(l))
//! get nth child  in CExprOfList
#define EXPR_L_AT(l,n)             CCOL_DL_AT(EXPR_DLIST(l),n)
//! is NULL or CExprOfNull or list size zero
#define EXPR_L_ISNULL(l)            (EXPR_ISNULL(l) || EXPR_L_SIZE(l) == 0)
//! get expression's type descriptor
#define EXPRS_TYPE(x)               (EXPR_C(x)->e_exprsType)
//! get type ID for XcodeML
#define EXPR_TYPEID(x)              (EXPRS_TYPE(x)->e_typeId)
//! get parent node
#define EXPR_PARENT(x)              (EXPR_C(x)->e_parentExpr)
//! has gcc attributes
#define EXPR_HAS_GCCATTR(x)         ((EXPR_L_ISNULL(EXPR_C(x)->e_gccAttrPre) == 0) \
                                    || (EXPR_L_ISNULL(EXPR_C(x)->e_gccAttrPost) == 0))

//! is node of EC_MEMBER_DECL
#define EXPR_IS_MEMDECL(x)          (EXPR_ISNULL(x) == 0 && EXPR_CODE(x) == EC_MEMBER_DECL)

//! iterate children of CExprOfList
#define EXPR_FOREACH(ite, expr) \
    CCOL_DL_FOREACH(ite, EXPR_DLIST(expr))

//! iterate children of CExprOfList (can replace/delete node of ite)
#define EXPR_FOREACH_SAFE(ite, iten, expr) \
    CCOL_DL_FOREACH_SAFE(ite, iten, EXPR_DLIST(expr))

//! iterate children of CExprOfList from ite
#define EXPR_FOREACH_FROM(ite, expr) \
    CCOL_DL_FOREACH_FROM(ite, EXPR_DLIST(expr))

//! iterate children of CExprOfList reversely
#define EXPR_FOREACH_REVERSE(ite, expr) \
    CCOL_DL_FOREACH_REVERSE(ite, EXPR_DLIST(expr))

//! iterate children of CExprOfList reversely (can replace/delete node of ite)
#define EXPR_FOREACH_REVERSE_SAFE(ite, iten, expr) \
    CCOL_DL_FOREACH_REVERSE_SAFE(ite, iten, EXPR_DLIST(expr))

//! iterate children of CExprOfList reversely from ite
#define EXPR_FOREACH_REVERSE_FROM(ite, expr) \
    CCOL_DL_FOREACH_REVERSE_FROM(ite, EXPR_DLIST(expr))

//! iterate children of CExprOfUnaryNode, CExprOfBinaryNode, CExprOfList
#define EXPR_FOREACH_MULTI(ite, expr) \
    for(memset(&(ite), 0, sizeof(ite)), \
        (EXPR_STRUCT(expr) == STRUCT_CExprOfList ? \
        ((ite).i.listNode = exprListHead(expr)) : 0); \
        (EXPR_STRUCT(expr) == STRUCT_CExprOfUnaryNode ? \
        ((ite).i.index == 0 && ((ite).node = EXPR_U(expr)->e_node, 1)) : \
        (EXPR_STRUCT(expr) == STRUCT_CExprOfBinaryNode ? \
        ((ite).i.index < 2 && ((ite).node = EXPR_B(expr)->e_nodes[(ite).i.index], 1)) : \
        (EXPR_STRUCT(expr) == STRUCT_CExprOfList ? \
        ((ite).i.listNode && ((ite).node = EXPR_L_DATA((ite).i.listNode), 1)) : \
        (0)))); \
        (EXPR_STRUCT(expr) == STRUCT_CExprOfUnaryNode || \
        EXPR_STRUCT(expr) == STRUCT_CExprOfBinaryNode) ? (++(ite).i.index, 0) : \
        (EXPR_STRUCT(expr) == STRUCT_CExprOfList ? \
        ((ite).i.listNode = CCOL_DL_NEXT((ite).i.listNode), 0) : (0)))


//! child index of tag symbol in EC_STRUCT/EC_UNION/EC_ENUM
#define EXPR_L_TAG_INDEX 1
//! child index of member decls in EC_STRUCT/EC_UNION/EC_ENUM
#define EXPR_L_MEMBER_DECLS_INDEX 2
//! child index of function body in EC_FUNC_DEF
#define EXPR_L_FUNC_BODY_INDEX 3


//! is type descriptor kind tk
#define ETYP_IS_KINDOF(td, tk)      (EXPR_T(td)->e_tdKind == (tk))
//! is type descriptor kind TD_BASICTYPE
#define ETYP_IS_BASICTYPE(td)       ETYP_IS_KINDOF(td, TD_BASICTYPE)
//! is type descriptor kind TD_STRUCT
#define ETYP_IS_STRUCT(td)          ETYP_IS_KINDOF(td, TD_STRUCT)
//! is type descriptor kind TD_UNION
#define ETYP_IS_UNION(td)           ETYP_IS_KINDOF(td, TD_UNION)
//! is type descriptor kind TD_ENUM
#define ETYP_IS_ENUM(td)            ETYP_IS_KINDOF(td, TD_ENUM)
//! is type descriptor kind TD_POINTER
#define ETYP_IS_POINTER(td)         ETYP_IS_KINDOF(td, TD_POINTER)
//! is type descriptor kind TD_ARRAY
#define ETYP_IS_ARRAY(td)           ETYP_IS_KINDOF(td, TD_ARRAY)
//! is type descriptor kind TD_FUNC
#define ETYP_IS_FUNC(td)            ETYP_IS_KINDOF(td, TD_FUNC)
//! is type descriptor kind TD_FUNC_OLDSTYLE
#define ETYP_IS_FUNC_OLDSTYLE(td)   ETYP_IS_KINDOF(td, TD_FUNC_OLDSTYLE)
//! is type descriptor kind TD_TYPEREF
#define ETYP_IS_TYPEREF(td)         ETYP_IS_KINDOF(td, TD_TYPEREF)
//! is type descriptor kind TD_DERIVED
#define ETYP_IS_DERIVED(td)         ETYP_IS_KINDOF(td, TD_DERIVED)
//! is type descriptor kind TD_GCC_TYPEOF
#define ETYP_IS_GCC_TYPEOF(td)      ETYP_IS_KINDOF(td, TD_GCC_TYPEOF)
//! is type descriptor kind TD_GCC_BUILTIN
#define ETYP_IS_GCC_BUILTIN(td)     ETYP_IS_KINDOF(td, TD_GCC_BUILTIN)
//! is type descriptor kind TD_COARRAY
#define ETYP_IS_COARRAY(td)         ETYP_IS_KINDOF(td, TD_COARRAY)
//! is type descriptor kind TD_STRUCT/TD_UNION
#define ETYP_IS_COMPOSITE(td)       (ETYP_IS_STRUCT(td) || ETYP_IS_UNION(td))
//! is type descriptor kind TD_STRUCT/TD_UNION/TD_ENUM
#define ETYP_IS_TAGGED(td)          (ETYP_IS_COMPOSITE(td) || ETYP_IS_ENUM(td))
//! is type descriptor kind TD_POINTER/TD_ARRAY
#define ETYP_IS_PTR_OR_ARRAY(td)    (ETYP_IS_POINTER(td) || ETYP_IS_ARRAY(td))
//! is type descriptor kind TD_BASICTYPE and is basic type BT_VOID
#define ETYP_IS_VOID(td)            (ETYP_IS_BASICTYPE(td) && (td)->e_basicType == BT_VOID)
//! is type descriptor kind TD_BASICTYPE and is basic type BT_INT
#define ETYP_IS_INT(td)             (ETYP_IS_BASICTYPE(td) && \
                                    ((td)->e_basicType == BT_INT || (td)->e_basicType == BT_UNSIGNED_INT))
//! is basic type char/u char/short/u short/_Bool
#define BTYP_IS_SMALLER_INT(bt)     ((bt) == BT_CHAR || (bt) == BT_UNSIGNED_CHAR ||\
                                    (bt) == BT_SHORT || (bt) == BT_UNSIGNED_SHORT ||\
                                    (bt) == BT_BOOL)
//! is type descriptor _Bool
#define ETYP_IS_BOOL(td)            (ETYP_IS_BASICTYPE(td) && (td)->e_basicType == BT_BOOL)

//! is type completely unknown size
#define ETYP_IS_UNKNOWN_SIZE(td)    ((td)->e_isSizeUnreducable == 0 && \
                                    (td)->e_isSizeZero == 0 && \
                                    (td)->e_len.eln_isFlexible == 0 && \
                                    getTypeSize(td) == 0 && \
                                    EXPR_ISNULL((td)->e_len.eln_lenExpr))

//! copy std's size information to dtd
#define ETYP_COPY_SIZE(dtd, std)    (void)(\
                                        (dtd)->e_size = (std)->e_size,\
                                        (dtd)->e_align = (std)->e_align,\
                                        (dtd)->e_isSizeUnreducable = (std)->e_isSizeUnreducable,\
                                        (dtd)->e_isSizeZero = (std)->e_isSizeZero)


/**
 * \brief
 * error kinds
 */
typedef enum {
    EK_WARN,
    EK_ERROR,
    EK_FATAL
} CErrorKind;

/**
 * \brief
 * error info
 */
typedef struct {
    //! error node
    CExpr           *pe_expr;
    //! error message
    char            *pe_msg;
    //! line number info
    CLineNumInfo    pe_lineNumInfo;
    //! error kind
    CErrorKind      pe_errorKind;
} CError;


extern void             freeError(CError *err);
extern void             freeErrors(CCOL_SList *errList);

extern void             exprCopyLineNum(CExpr *to, CExpr *from);
extern CExpr*           exprListJoin(CExpr *exprHead, CExpr *exprTail);
extern CExpr*           exprListCons(CExpr *exprHead, CExpr *exprTail);
extern CExpr*           exprListAdd(CExpr *exprHead, CExpr *exprTail);
extern CExpr*           exprListRemoveHead(CExpr *exprList);
extern CExpr*           exprListRemoveTail(CExpr *exprList);
extern CExpr*           exprListRemove(CExpr *exprList, CCOL_DListNode *ite);
extern CExpr*           exprListClear(CExpr *exprList);
extern CCOL_DListNode*  exprListHead(CExpr *listExpr);
extern CExpr*           exprListHeadData(CExpr *listExpr);
extern CCOL_DListNode*  exprListTail(CExpr *listExpr);
extern CExpr*           exprListTailData(CExpr *listExpr);
extern CCOL_DListNode*  exprListNext(CExpr *listExpr);
extern CExpr*           exprListNextData(CExpr *listExpr);
extern CCOL_DListNode*  exprListNextN(CExpr *listExpr, unsigned int n);
extern CExpr*           exprListNextNData(CExpr *listExpr, unsigned int n);
extern void             printErrors(FILE *fp);
extern void             dumpExpr(FILE *fp, CExpr *expr);
extern void             dumpExprSingle(FILE *fp, CExpr *expr);
extern CExpr*           exprSetExtension(CExpr *expr);
extern void             exprReplace(CExpr *parent, CExpr *oldChild, CExpr *newChild);
extern void             exprRemove(CExpr *parent, CExpr *child);
extern CExpr*           exprError(void);
extern CExpr*           exprError1(CExpr *cause);

extern CExpr*           allocPointerDecl(CExpr *typeQuals);
extern CExprOfTypeDesc* setPointerTypeOfArrayElem(CExprOfTypeDesc *aryTd);
extern CCOL_DListNode*  skipPointerDecl(CCOL_DListNode *declrChildNode,
                            CExpr *declarator, int *numPtrDecl);
extern void             procTypeDefInParser(CExpr *dataDefExpr);
extern void             addFuncDefDeclaratorSymbolInParser(CExpr *declarator);
extern void             addInitDeclSymbolInParser(CExpr *initDecl);
extern CError*          addWarn(CExpr *expr, const char *fmt, ...);
extern CError*          addError(CExpr *expr, const char *fmt, ...);
extern void             addFatal(CExpr *expr, const char *fmt, ...);
extern int              isSkipErrorOutput(CExpr *expr);
extern void             dumpError(FILE *fp);
extern int              hasChildren(CExpr *expr);;

#define exprListNth(listExpr,n) exprListNextNData(listExpr,n)

/**
 * \brief
 * symbol table
 */
typedef struct CSymbolTable {

    //! next symbol table
    struct CSymbolTable *stb_parentTab, *stb_childTab;
    //! identifier group (var, func, type, enumerator)
    CCOL_HashTable      stb_identGroup;
    //! tag group (struct/union/enum tag)
    CCOL_HashTable      stb_tagGroup;
    //! label group
    CCOL_HashTable      stb_labelGroup;
    //! line number info
    CLineNumInfo        stb_lineNumInfo;
    // count of put symbol
    int                 stb_putCount;
    // is global scope symbol table
    unsigned int        stb_isGlobal        : 1;
    // is symbol table of function definition's STMTS_AND_DECLS
    unsigned int        stb_isFuncDefBody   : 1;

} CSymbolTable;


extern CSymbolTable*        allocSymbolTable(void);
extern CSymbolTable*        allocSymbolTable1(CSymbolTable *parentSymTab);
extern void                 freeSymbolTable(CSymbolTable *symTab);
extern void                 freeSymbolTableList(void);
extern void                 addSymbolInParser(CExpr *sym, CSymbolTypeEnum symType);
extern void                 addSymbolAt(CExprOfSymbol *sym, CExpr *parent, CSymbolTable *symTab,
                                CExprOfTypeDesc *td, CSymbolTypeEnum symType, int forwardNum,
                                int redeclCheck);
extern void                 addSymbolDirect(CSymbolTable *symTab, CExprOfSymbol *sym,
                                CSymbolTableGroupEnum group);
extern CExprOfSymbol*       findSymbol(const char *symbol);
extern CExprOfSymbol*       findSymbolByGroup(const char *symbol, CSymbolTableGroupEnum group);
extern CExprOfSymbol*       findSymbolByGroup1(CSymbolTable *symTab, const char *symbol,
                                CSymbolTableGroupEnum group);
extern CExprOfSymbol*       removeSymbolByGroup(CSymbolTable *symTab, const char *symbol,
                                CSymbolTableGroupEnum group);
extern CCOL_HashTable*      getSymbolHashTable(CSymbolTable *symTab, CSymbolTableGroupEnum group);
extern void                 pushSymbolTable(void);
extern void                 pushSymbolTableToExpr(CExpr *expr);
extern CLineNumInfo         popSymbolTable(void);
extern CSymbolTable*        getCurrentSymbolTable(void);
extern CSymbolTable*        getGlobalSymbolTable(void);
extern void                 reorderSymbol(CExprOfSymbol *sym);
extern void                 dumpSymbolTable(FILE *fp);
extern void                 dumpFileIdTable(FILE *fp);
extern void                 initStaticData(void);
extern void                 freeStaticData(void);
extern CExprOfSymbol*       getTagSymbol(CExprOfTypeDesc *td);


/*
 * TypeDesc Operations
 */

/**
 * \brief
 * type descriptor comparison results
 */
typedef enum {
    CMT_EQUAL,
    CMT_DIFF_TYPE,
    CMT_DIFF_TYPEQUAL,
    CMT_DIFF_ARRAYLEN,
    CMT_DIFF_FUNCPARAM,
    CMT_DIFF_FUNCRETURN,
} CCompareTypeEnum;

/**
 * \brief
 * type descriptor comparison levels
 */
typedef enum {
    TRL_ASSIGN,
    TRL_PARAM,
    TRL_DATADEF,
    TRL_EXTERN,
    TRL_MAX,
} CTypeRestrictLevel;

extern void                 initStaticTypeDescData(void);
extern void                 freeStaticTypeDescData(void);
extern CExprOfTypeDesc*     getFuncType(CExprOfTypeDesc *td, CExprOfTypeDesc **parentTd);
extern CExprOfTypeDesc*     resolveType(CExpr *expr);
extern CExprOfTypeDesc*     resolveType_initVal(CExprOfTypeDesc *td, CExpr *initVal);
extern CExprOfTypeDesc*     resolveType_ident(CExpr *expr,
                                int occurErrorIfNotFound, int *ignore);
extern CExprOfTypeDesc*     getRefType(CExprOfTypeDesc *typeDesc);
extern CExprOfTypeDesc*     getRefTypeWithoutFix(CExprOfTypeDesc *typeDesc);
extern CExprOfTypeDesc*     getMemberType(CExprOfTypeDesc *parentTd,
                                const char *memberName, CExprOfTypeDesc **outParentTd,
                                CExprOfSymbol **outParentSym);
extern CCompareTypeEnum     compareType(CExprOfTypeDesc *td1, CExprOfTypeDesc *td2);
extern CCompareTypeEnum     compareTypeForAssign(
                                CExprOfTypeDesc *td1, CExprOfTypeDesc *td2);
extern CCompareTypeEnum     compareTypeExcludeGccAttr(
                                CExprOfTypeDesc *td1, CExprOfTypeDesc *td2);
extern void                 setExprParent(CExpr *expr, CExpr *parent);
extern CExpr*               getFuncCall(CExpr *expr);
extern CExpr*               getFuncCallAbsolutelyCalling(CExpr *expr);
extern CExpr*               getLastExprStmt(CExpr *stmts);
extern CExprOfTypeDesc*     getSizedType(CExprOfTypeDesc *td);
extern int                  getTypeSize(CExprOfTypeDesc *td);
extern int                  getTypeAlign(CExprOfTypeDesc *td);
extern int                  isStatement(CExpr *expr);
extern int                  isStatementOrLabelOrDeclOrDef(CExpr *expr);
extern int                  isCirculatedInnerStatement(CExpr *expr);
extern int                  isScopedStmt(CExpr *expr);
extern int                  isVoidPtrType(CExprOfTypeDesc *td);
extern int                  isScalarOrPointerType(CExprOfTypeDesc *td);
extern int                  isCompatiblePointerType(CExprOfTypeDesc *td1,
                                CExprOfTypeDesc *td2, CExprOfTypeDesc **rtd, int inhibitPtrAndPtr);
extern CExprOfTypeDesc*     getPriorityType(CExprOfTypeDesc *td1, CExprOfTypeDesc *td2);
extern CExprOfTypeDesc*     getPriorityTypeForAssign(CExprOfTypeDesc *td1, CExprOfTypeDesc *td2);
extern int                  isExprCodeChildOf(CExpr *expr, CExprCodeEnum ec, CExpr *parentExpr,
                                CExpr **outParentExpr);
extern int                  isExprCodeChildStmtOf(CExpr *expr, CExprCodeEnum ec,
                                CExpr *parentExpr, CExpr **outParentExpr);
extern int                  isLogicalExpr(CExpr *e);
extern int                  isIntegerType(CExprOfTypeDesc *td);
extern int                  isBasicTypeOrEnum(CExprOfTypeDesc *td);
extern int                  isScspecSet(CExprOfTypeDesc *typeDesc);
extern int                  isTypeQualSet(CExprOfTypeDesc *typeDesc);
extern int                  isTypeQualOrExtensionSet(CExprOfTypeDesc *typeDesc);
extern int                  isTypeQualEquals(CExprOfTypeDesc *td1, CExprOfTypeDesc *td2);
extern int                  isCoArrayAssign(CExpr *expr);

extern void             startCheckGccAttr(CExpr *expr);
extern void             endCheckGccAttr(CExpr *expr);
extern int              hasGccAttr(CExprOfTypeDesc *td, CGccAttrKindEnum gak);
extern int              hasGccAttrId(CExpr *e, CGccAttrIdEnum id);
extern int              hasGccAttrDerived(CExprOfTypeDesc *td, CGccAttrKindEnum gak);
extern void             getGccAttrRecurse(CExpr *attrs, CExprOfTypeDesc *td, CGccAttrKindEnum gak);
extern CExpr*           exprSetAttrPre(CExpr *expr, CExpr *attr);
extern CExpr*           exprSetAttrPost(CExpr *expr, CExpr *attr);
extern CExpr*           exprSetAttrHeadNode(CExpr *expr, CExpr *attr);
extern CExpr*           exprSetAttrTailNode(CExpr *expr, CExpr *attr);
extern void             exprJoinAttr(CExpr *dst, CExpr *src);
extern void             exprJoinAttrToPre(CExpr *dst, CExpr *src);
extern void             exprJoinDuplicatedAttr(CExprOfTypeDesc *dst,
                            CExprOfTypeDesc *src);
extern void             exprAddAttrToPre(CExprOfTypeDesc *td, CExprOfBinaryNode *arg);
extern void             exprCopyAttr(CExpr *dst, CExpr *src);
extern void             exprFixAttr(CExprOfTypeDesc *td, CExprOfBinaryNode *declr,
                            CDeclaratorContext declrContext, int isTypeDef);
extern void             exprChoiceAttr(CExpr *attrs);
extern CGccAttrInfo*    getGccAttrInfo(const char *symbol);
extern int              exprHasGccAttr(CExpr *expr);
extern int              isGccBuiltinType(const char *typeName);
extern CExprOfTypeDesc* getGccBuiltinFuncReturnType(const char *funcName,
                            CExprOfTypeDesc *argTd, CExpr *errExpr, int *isConstFunc);
extern void             checkGccAttrOutput();


/*
 * Enumerator Info
 */

extern const char   *s_CSCSpecEnumNames[SS_END];
extern const char   *s_CTypeSpecEnumNames[TS_END];
extern const char   *s_CTypeQualEnumNames[TQ_END];
extern const char   *s_CAssignEnumNames[AS_END];
extern const char   *s_CExprStructEnumNames[STRUCT_CExpr_END];
extern const char   *s_CSymbolTypeEnumNames[ST_END];
extern const char   *s_CBasicTypeEnumNames[BT_END];
extern const char   *s_CCharTypeEnumNames[CT_END];
extern const char   *s_CCardinalEnumNames[CD_END];
extern const char   *s_CDirectiveTypeEnumNames[DT_END];
extern const char   *s_CTypeDescKindEnumNames[TD_END];

/*
 * Lexer/Parser/Compiler Functions/Macros
 */

extern CExprOfTypeDesc* allocIntTypeDesc();
extern int              unescChar(char *s, char *cstr, int *numChar, int isWChar);
extern CExpr*           getMemberDeclsExpr(CExprOfTypeDesc *td);
extern int              hasSymbols(CExpr *expr);
extern int              hasDeclarationsCurLevel(CExpr *expr);
extern void             convertFileIdToNameTab();
const char*             getFileNameByFileId(int fileId);

/*
 * Lexer/Parser/Compiler Variable
 */

/**
 * \brief
 * 
 */
typedef struct {
    int fie_id;
} CFileIdEntry;

//! root node of AST
extern CExpr            *s_exprStart;
//! current line number info at parsing
extern CLineNumInfo     s_lineNumInfo;
//! symbol table stack
extern CCOL_DList       s_symTabStack;
//! top symbol table
extern CSymbolTable     *s_defaultSymTab;
//! errors
extern CCOL_SList       s_errorList;
//! expression's type descriptors
extern CCOL_SList       s_exprsTypeDescList;
//! collected type descriptors to output typeTable
extern CCOL_DList       s_typeDescList;
//! has errors
extern int              s_hasError;
//! has warnings
extern int              s_hasWarn;
//! file name to file ID table
extern CCOL_HashTable   s_fileIdTab;
//! file ID to file name table
extern CCOL_HashTable   s_fileIdToNameTab;
//! is s_fileIdTabl freed 
extern int              s_freedFileIdTab;
//! basic number type descriptors
extern CExprOfTypeDesc  s_numTypeDescs[BT_END];
//! enumerator type descriptor
extern CExprOfTypeDesc  s_enumeratorTypeDesc;
//! multi byte string constant type descriptor
extern CExprOfTypeDesc  s_stringTypeDesc;
//! wide string constant type descriptor
extern CExprOfTypeDesc  s_wideStringTypeDesc;
//! multi byte charactor constant type descriptor
extern CExprOfTypeDesc  s_charTypeDesc;
//! wide charactor constant type descriptor
extern CExprOfTypeDesc  s_wideCharTypeDesc;
//! type descritors which are static allocated
extern CCOL_SList       s_staticTypeDescs;
//! char pointer type descriptor
extern CExprOfTypeDesc  s_charPtrTypeDesc;
//! void type descriptor
extern CExprOfTypeDesc  s_voidTypeDesc;
//! void pointer type descriptor
extern CExprOfTypeDesc  s_voidPtrTypeDesc;
//! dummy type descriptor
extern CExprOfTypeDesc  s_undefTypeDesc;
//! int64 type descriptor
extern CBasicTypeEnum   s_int64Type;
//! wchar_t type descriptor
extern CBasicTypeEnum   s_wcharType;
//! number of conversion for pragma pack
extern int              s_numConvsPragmaPack;
//! number of conversion for anonymous member access
extern int              s_numConvsAnonMemberAccess;
//! is enabled pragma pack
extern int              s_pragmaPackEnabled;
//! alignment specified by pragma pack
extern int              s_pragmaPackAlign;
//! temporary char buffer
extern char             s_charBuf[2][MAX_NAME_SIZ];

#endif /*  _C_EXPR_H_ */

