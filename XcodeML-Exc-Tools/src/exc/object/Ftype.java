/*
 *   Class for Fortran type and kind/length parameter
 */

package exc.object;

import exc.block.*;
import exc.xmpF.*;
//import java.util.*;

public class Ftype
{
  // data type (integer number, avoiding conflict with BasicType.INT, etc.)
  public final static int UNKNOWN       = 1000;
  public final static int INTEGER       = 1001;   // builtin & numeric type
  public final static int REAL          = 1002;   // builtin & numeric type
  public final static int COMPLEX       = 1003;   // builtin & numeric type
  public final static int LOGICAL       = 1004;   // builtin & nonnumeric type
  public final static int CHARACTER     = 1005;   // builtin & nonnumeric type
  public final static int DERIVED       = 1006;   // derived type (structure)
  
  // kind and length parameter (value of kind and len)
  private static final int UNEVALUABLE  = -1;  // cannot evaluate into integer
  private static final int ASTERISK     = -2;  // len=*

  //------------------------------------
  //  internal structure
  //------------------------------------
  // Xobject representation
  // -- Xtype of the ultimate element of array/structure
  private Xtype   xtype = null;
  // -- the identification number of the basic type (for BasicType)
  private int     basicType = BasicType.UNDEF;
  // -- the tag name of the structure (for StructType)
  private Ident   structTag = null;

  // context for constant-folding
  public Block    block = null;

  //------------------------------------
  //  open members
  //------------------------------------
  // data type
  public int     type;         // INTEGER, REAL, ... or UNKNOWN
  // kind parameter
  public int     kind;         // positive value, UNEVALUABLE, or UNKNOWN
  public Xobject kindExpr;     // expression or null (for UNKNOWN)
  // length parameter
  public int     len;          // positive value,ASTERISK,UNEVALUABLE or UNKNOWN
  public Xobject lenExpr;      // expression or null (for UNKNOWN or ASTERISK)


  //------------------------------------
  //  constructor from Xtype
  //------------------------------------
  public Ftype(Xobject obj)
  {
    this(obj, null);
  }
  public Ftype(Xobject obj, Block block)
  {
    this(obj.Type(), block);

    if (obj.isConstant() && ((XobjConst)obj).getFkind() != null) {
      // Scratch kind and kindExpr to avoid having wrong value.
      // xobjConst has unreadable kind parameter in the form of String.
      kind = UNKNOWN;
      kindExpr = null;
    }
  }

  public Ftype(Xtype xtype)
  {
    this(xtype, null);
  }
  public Ftype(Xtype xtype, Block block)
  {
    this.block = block;
    _setXtypeEtc(xtype);          // set element xtype and tag

    _setType();
    _setKindAndKindExpr();
    _setLenAndLenExpr();
  }

  // test
  public static Ftype Ftype(Xtype xtype)
  {
    Ftype ftype = new Ftype(xtype);
    return ftype;
  }

  //------------------------------------
  //  constructor from basicType and kind
  //------------------------------------
  public Ftype(int basicType, int kind, int len, Block block)
  {
    this.block = block;
    this.basicType = basicType;
    xtype = null;
    structTag = null;

    _setType();
    _setKindAndKindExpr(kind);
    _setLenAndLenExpr(len);
  }

  public Ftype(int basicType, int kind, Block block)
  {
    this.block = block;
    this.basicType = basicType;
    xtype = null;
    structTag = null;

    _setType();
    _setKindAndKindExpr(kind);
    _setLenAndLenExpr();
  }


  //------------------------------------
  //  sub
  //------------------------------------
  private void _setXtypeEtc(Xtype xtype)
  {
    switch (xtype.getKind()) {
    case Xtype.BASIC:
      // should be one of BasicType.{BOOL,INT,FLOAT,FLOAT_COMPLEX,CHARACTER}
      this.xtype = xtype;
      basicType = xtype.getBasicType();
      structTag = null;
      break;

    case Xtype.STRUCT:
      this.xtype = xtype;
      basicType = BasicType.UNDEF;
      structTag = xtype.getTagIdent();
      break;

    case Xtype.F_ARRAY:
    case Xtype.F_COARRAY:
      _setXtypeEtc(xtype.getRef());
      break;
    }
  }


  private void _setType()
  {
    type = _getType();
  }

  private int _getType()
  {
    switch (basicType) {
    case BasicType.INT:
    case BasicType.SHORT:
    case BasicType.LONG:
    case BasicType.LONGLONG:
      return INTEGER;

    case BasicType.FLOAT:
    case BasicType.DOUBLE:
    case BasicType.LONG_DOUBLE:
      return REAL;

    case BasicType.FLOAT_COMPLEX:
    case BasicType.DOUBLE_COMPLEX:
    case BasicType.LONG_DOUBLE_COMPLEX:
      return COMPLEX;

    case BasicType.F_CHARACTER:
      return CHARACTER;

    case BasicType.BOOL:
      return LOGICAL;

    case BasicType.UNDEF:     // for derived type
      return DERIVED;

    default:
      break;
    }

    return UNKNOWN;
  }


  /* priority of the kind parameter interpretation
   *  1. fkind member of xtype. 
   *    Eg, if xtype.fkind != null, kindExpr=xtype.fkind
   *  2. BasicType.basic_type.
   *    Eg, for BasicType.DOUBLE, kind=8 and kindExpr=Xcons.IntConstant(8)
   *  3. default kind parameter.
   *    Eg, kind=4 for INTEGER, kind=1 for CHARACTER.
   */
  private void _setKindAndKindExpr()
  {
    if (xtype != null && xtype.getFkind() != null) {
      // get kind parameter from xtype.fkind
      kindExpr = xtype.getFkind().cfold(block);
      if (kindExpr.isIntConstant())
        kind = kindExpr.getInt();
      else
        kind = UNEVALUABLE;
    } else {
      // get kind parameter from BasicType.basic_type
      kind = _getKind();
      if (kind > 0)
        kindExpr = Xcons.IntConstant(kind);
      else
        kindExpr = null;
    }
  }

  private void _setKindAndKindExpr(int kind)
  {
    this.kind = kind;
    if (kind > 0)
      kindExpr = Xcons.IntConstant(kind);
    else
      kindExpr = null;
  }

  private int _getKind()
  {
    switch (basicType) {
    case BasicType.INT:
    case BasicType.FLOAT:
    case BasicType.FLOAT_COMPLEX:
    case BasicType.BOOL:
    case BasicType.F_CHARACTER:
      return _getDefaultKind(type);

    case BasicType.SHORT:
      return 2;

    case BasicType.LONG:
      return 4;

    case BasicType.LONGLONG:
    case BasicType.DOUBLE:
    case BasicType.DOUBLE_COMPLEX:
      return 8;

    case BasicType.LONG_DOUBLE:
    case BasicType.LONG_DOUBLE_COMPLEX:
      return 16;

    default:                         // for derived type
      break;
    }

    return UNKNOWN;
  }

    
  /* priority of the kind parameter interpretation
   *  1. flen member of xtype. 
   *    Eg, if xtype.flen != null, lenExpr=xtype.len
   *        if xtype.isFlenVariable(), len=ASTERISK
   *  2. default length parameter.
   *    Eg, len=1 for CHARACTER, len=UNKNOWN otherwise.
   */
  private void _setLenAndLenExpr()
  {
    // if not character
    if (type != CHARACTER) {
      len = UNKNOWN;
      lenExpr = null;
      return;
    }

    // get length parameter from xtype
    if (xtype != null) {
      if (xtype.isFlenVariable()) {
        len = ASTERISK;
        lenExpr = null;
        return;
      } else if (xtype.getFlen() != null) {
        lenExpr = xtype.getFlen().cfold(block);
        if (lenExpr.isIntConstant())
          len = lenExpr.getInt();
        else
          len = UNEVALUABLE;
        return;
      }
    }

    // get default length parameter
    len = _getDefaultLen(type);
    lenExpr = Xcons.IntConstant(len);
  }


  private void _setLenAndLenExpr(int len)
  {
    this.len = len;
    if (len > 0)
      lenExpr = Xcons.IntConstant(len);
    else
      lenExpr = null;
  }



  /**********************************************
  private void _setKindAndLen()
  {
    int kind1, kind2;
    int len1, len2;
    Xobject kindExpr2;
    Xobject lenExpr2;

    // Internal Representation #1: type and type parameter
    kind1 = UNKNOWN;
    len1 = UNKNOWN;
    switch (basicType) {
    case BasicType.BOOL:
      kind1 = _getDefaultKind(basicType);
      break;

    case BasicType.INT:
      kind1 = _getDefaultKind(basicType);
      break;
    case BasicType.SHORT:
      kind1 = 2;
      break;
    case BasicType.LONG:
      kind1 = 4;
      break;
    case BasicType.LONGLONG:
      kind1 = 8;
      break;

    case BasicType.FLOAT:
      kind1 = _getDefaultKind(basicType);
      break;
    case BasicType.DOUBLE:
      kind1 = 8;
      break;
    case BasicType.LONG_DOUBLE:
      kind1 = 16;
      break;

    case BasicType.FLOAT_COMPLEX:
      kind1 = _getDefaultKind(basicType);
      break;
    case BasicType.DOUBLE_COMPLEX:
      kind1 = 8;
      break;
    case BasicType.LONG_DOUBLE_COMPLEX:
      kind1 = 16;
      break;

    case BasicType.F_CHARACTER:
      kind1 = _getDefaultKind(basicType);
      len1 = 1;
      break;

    default:
      XMP.fatal("found illegal BasicType No: " + basicType);
    }

    // Internal Representation #2: type parameter and length parameter
    kind2 = UNKNOWN;
    kindExpr2 = null;
    if (xtype != null)
      kindExpr2 = xtype.getFkind().cfold(block);
    if (kindExpr2 != null) {           // kind parameter specified 
      if (kindExpr2.isIntConstant())
        kind2 = kindExpr2.getInt();       /// evaluated
    }

    len2 = UNKNOWN;
    lenExpr2 = null;
    if (xtype != null)
      lenExpr2 = xtype.getFlen().cfold(block);
    if (lenExpr2 != null) {           // len parameter specified 
      if (lenExpr2.isIntConstant())
        len2 = lenExpr2.getInt();       /// evaluated
    }

    // Overall type, kind parameter and length parameter
    if (kindExpr2 == null) {
      kind = kind1;
      kindExpr = (kind1 > 0) ? Xcons.IntConstant(kind1) : null;
    } else {
      kind = kind2;
      kindExpr = kindExpr2;
    }

    if (lenExpr2 == null) {
      len = len1;
      lenExpr = (len1 > 0) ? Xcons.IntConstant(len1) : null;
    } else {
      len = len2;
      lenExpr = lenExpr2;
    }
  }
******************************************************/


  //------------------------------------
  //  get default
  //------------------------------------
  private int _getDefaultKind(int type)
  {
    switch (type) {
    case CHARACTER:
      return 1;         // Default kind parameter is 1.
    case INTEGER:
    case REAL:
    case COMPLEX:
    case LOGICAL:
      return 4;         // Default kind parameter is 4.
    }
    return UNKNOWN;
  }

  private int _getDefaultLen(int type)
  {
    switch (type) {
    case CHARACTER:
      return 1;           // Default character length is 1.
    }

    return UNKNOWN;
  }


  //------------------------------------
  //  inquires
  //------------------------------------
  public Boolean sameTypeAndKind(Xobject xobj)
  {
    Ftype ftype = new Ftype(xobj);
    return sameTypeAndKind(ftype);
  }
  public Boolean sameTypeAndKind(Xobject xobj, Block block)
  {
    Ftype ftype = new Ftype(xobj, block);
    return sameTypeAndKind(ftype);
  }

  public Boolean sameTypeAndKind(Xtype xtype)
  {
    Ftype ftype = new Ftype(xtype);
    return sameTypeAndKind(ftype);
  }
  public Boolean sameTypeAndKind(Xtype xtype, Block block)
  {
    Ftype ftype = new Ftype(xtype, block);
    return sameTypeAndKind(ftype);
  }

  public Boolean sameTypeAndKind(Ftype ftype)
  {
    if (type == ftype.getType() && kind == ftype.getKind())
      return true;
    return false;
  }


  // name as intrinsic function e.g., "int", "cmplx", "char"
  public String getNameOfConvFunction()
  {
    switch (type) {
    case INTEGER:
      return "int";
    case REAL:
      return "real";
    case COMPLEX:
      return "cmplx";
    case LOGICAL:
      return "logical";
    case CHARACTER:
      return "char";
    default:
      break;
    }
    return null;
  }

  public int getType()     // type in terms of Fortran
  {
    return type;
  }

  public int getKind()     // kind parameter
  {
    return kind;
  }

  public Xobject getKindExpr()     // kind parameter
  {
    return kindExpr;
  }

  public int getLen()      // length parameter
  {
    return len;
  }

  public Xobject getLenExpr()      // length parameter
  {
    return lenExpr;
  }

  public Xtype getXtype()
  {
    return xtype;
  }

  public String toString()
  {
    switch (type) {
    case INTEGER:
      return "INTEGER(kind=" + kind + ")";
    case REAL:
      return "REAL(kind=" + kind + ")";
    case COMPLEX:
      return "COMPLEX(kind=" + kind + ")";
    case LOGICAL:
      return "LOGICAL(kind=" + kind + ")";
    case CHARACTER:
      return "REAL(len=" + len + ")";
    case DERIVED:
      return "type(" + structTag.getName() + ")";
    case UNKNOWN:
      return "UNKNOWN";
    default:
      break;
    }
    return "illegal type number=" + type;
  }


  /****************************
   * UNDER CONSTRUCTION
   * USE this in XMPcoarray
   *******************************/
  /***************************************
  public int getElementLength()
  {
    switch (type) {
    case INTEGER:
    case REAL:
    case LOGICAL:
      return kind;
    case COMPLEX:
      return kind * 2;
    case CHARACTER:
      return len;
    case DERIVED:
      return 
  **************************************/

}
