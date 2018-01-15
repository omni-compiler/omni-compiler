package exc.xmpF;

import exc.object.*;
import exc.block.*;
import java.util.*;

/*
 * Madiator for coindexed object such as a(i,j)[k]
 */
public class XMPcoindexObj {

  // GET/PUT Interface types
  // (see also 1libxmpf/include/xmpf_internal_coarray.h)
  //final static int GetInterfaceType = 6;           // valid last implementation
  final static int GetInterfaceType = 8;
  //final static int PutInterfaceType = 7;           // valid last implementation
  final static int PutInterfaceType = 8;

  // constants
  final static String COARRAYPUT_PREFIX = "xmpf_coarray_put";                // for Type 7
  final static String COARRAYGET_PREFIX = "xmpf_coarray_get";                // for Type 6
  final static String COARRAYGET_GENERIC_NAME = "xmpf_coarray_get_generic";  // for Type 8
  final static String COARRAYPUT_GENERIC_NAME = "xmpf_coarray_put_generic";  // for Type 8
  // optimization
  final static String COARRAYGETSUB_GENERIC_NAME = "xmpf_coarray_getsub_generic";

  // attributes
  String name;
  Xobject obj;               // Xcode.CO_ARRAY_REF or MEMBER_REF (derived-type scalar)
                             // or F_ARRAY_REF (derived-type array)
  Xobject subscripts;
  Xobject cosubscripts;
  int exprRank;                  // rank of the reference

  // mediator
  XMPcoarray coarray;

  //------------------------------
  //  CONSTRUCTOR
  //------------------------------
  public XMPcoindexObj(Xobject obj, XMPcoarray coarray) {
    this.obj = obj;
    name = _getName(obj);
    this.coarray = coarray;
    _initOthers();
  }

  public XMPcoindexObj(Xobject obj, ArrayList<XMPcoarray> coarrays) {
    this.obj = obj;
    name = _getName(obj);
    coarray = XMPcoarray.findCoarrayInCoarrays(name, coarrays);
    if (coarray == null)
      XMP.fatal("INTERNAL: cannot find the coarray by name in ArrayList");
    _initOthers();
  }

  private void _initOthers() {
    subscripts = _getSubscripts(obj);
    cosubscripts = _getCosubscripts(obj);
    exprRank = _getExprRank();
    if (subscripts == null && exprRank > 0)
      subscripts = _wholeArraySubscripts(exprRank);
  }

  private String _getName(Xobject obj) {
    switch(obj.Opcode()) {
    case CO_ARRAY_REF:      // v[k]
      return _getName_coarray(obj);
    case MEMBER_REF:        // v[k]%b..%c
      return _getName(obj.getArg(0).getArg(0));
    case F_ARRAY_REF:       // v[k]%b..%c(i,..,j)
      return _getName(obj.getArg(0).getArg(0));
    default:
      break;
    }
    XMP.fatal("INTERNAL: unexpected form of coindexed object: " +
              obj);
    return null;
  }

  private String _getName_coarray(Xobject obj) {
    Xobject varRef = obj.getArg(0).getArg(0);
    switch (varRef.Opcode()) {
    case F_ARRAY_REF:                     // subarray
      return varRef.getArg(0).getArg(0).getName();
    case VAR:                      // scalar or whole array
      return varRef.getName();
    default:
      break;
    }
    XMP.fatal("broken Xcode to describe a coindexed object");
    return null;
  }

  private Xobject _getSubscripts(Xobject xobj) {
    return xobj.getSubscripts();    // see XobjList
  }

  private Xobject _getCosubscripts(Xobject xobj) {
    Xobject xobj1, xobj2;
    switch (xobj.Opcode()) {
    case CO_ARRAY_REF:           // v[k]
      return xobj.getArg(1);

    case MEMBER_REF:             // guess (v[k]%b..)%c
      xobj1 = xobj.getArg(0);
      if (xobj1.Opcode() != Xcode.F_VAR_REF)
        break;
      xobj2 = xobj1.getArg(0);
      return _getCosubscripts(xobj2);

    case F_ARRAY_REF:            // guess v[k]%b..%c(i,..,j)
      xobj1 = xobj.getArg(0);
      if (xobj1.Opcode() != Xcode.F_VAR_REF)
        break;
      xobj2 = xobj1.getArg(0);
      return _getCosubscripts(xobj2);

      /*************************
      if (xobj2.Opcode() != Xcode.MEMBER_REF)
        break;
      xobj3 = xobj2.getArg(0);
      if (xobj3.Opcode() != Xcode.F_VAR_REF)
        break;
      xobj4 = xobj3.getArg(0);
      if (xobj4.Opcode() != Xcode.CO_ARRAY_REF)
        break;
      return xobj4.getArg(1);
      *********************************/

    default:
      break;
    }
    return null;
  }


  // make whole-array shape (:,:,...,:)
  //
  private Xobject _wholeArraySubscripts(int exprRank) {
    Xobject list = Xcons.List();
    for (int i = 0; i < exprRank; i++)
      list.add(Xcons.FindexRangeOfAssumedShape());
    if (list.hasNullArg())
      XMP.fatal("INTERNAL: generated null argument (_wholeArraySubscripts)");

    return list;
  }


  /*  Implicit type conversion if necessary
   *  ex.  RHS -> int(RHS)
   *  ex.  RHS -> real(RHS,8)
   */
  private Xobject _convRhsType(Xobject rhs)
  {
    return _typeConv(rhs, obj);
  }


  /*  convert xobj to int(xobj,4), real(xobj,8), etc. if necessary
   */
  private Xobject _typeConv(Xobject xobj, Xobject mold)
  {
    Ftype newType = new Ftype(mold, getBlock());
    return _typeConv(xobj, newType);
  }

  private Xobject _typeConv(Xobject xobj, int type, int kind)
  {
    Ftype newType = new Ftype(type, kind, getBlock());
    return _typeConv(xobj, newType);
  }

  private Xobject _typeConv(Xobject xobj, Ftype newType)
  {
    //Block fblock = getEnv().getCurrentDef().getBlock();
    Block fblock = getBlock();

    if (newType.sameTypeAndKind(xobj))
      // no need to convert
      return xobj;

    // build cast function
    FunctionType funcType =
      new FunctionType(newType.getXtype(), Xtype.TQ_FINTRINSIC);
    String castName = newType.getNameOfConvFunction();
    Xobject kind = newType.getKindExpr();
    Ident castFunc = getEnv().declIntrinsicIdent(castName, funcType);
    Xobject callExpr = castFunc.Call(Xcons.List(xobj, kind));
    return callExpr;
  }


  /***********************************************************************
  private Xobject _convRhsType___OLD___(Xobject rhs)
  {
    String lhsTypeStr = coarray.getFtypeString();
    Xobject lhsKind = coarray.getFkind();

    if (lhsTypeStr == null) {
      XMP.warning("No cast function for RHS was generated because of " +
                  "unknown type of coarray LHS: " + coarray);
      return rhs;
    }

    Xtype rhsType = rhs.Type();
    if (rhsType.getKind() == Xtype.F_ARRAY)
      rhsType = rhsType.getRef();
    String rhsTypeStr = coarray.getFtypeString(rhsType);
    Xobject rhsKind = rhsType.getFkind();

    if (lhsKind == null) {      // case default kind
      if (rhsKind == null && lhsTypeStr.equals(rhsTypeStr)) {
        // same type and same default kind
        return rhs;
      }

      //XMP.warning("Invoked automatic type conversion ("+lhsTypeStr+")", block);
      return _convTypeByIntrinsicCall(lhsTypeStr, rhs);
    }

    if (lhsTypeStr.equals(rhsTypeStr) && rhsKind != null) {
      if (lhsKind.canGetInt() && rhsKind.canGetInt() &&
          lhsKind.getInt() == rhsKind.getInt())
        // same type and same kind
        return rhs;
    }

    //XMP.warning("Invoked automatic type conversion ("+lhsTypeStr+" with kind)", block);
    return _convTypeByIntrinsicCall(lhsTypeStr, rhs, lhsKind);
  }

  private Xobject _convTypeByIntrinsicCall(String fname, Xobject expr)
  {
    //FunctionType ftype = new FunctionType(Xtype.FunspecifiedType, Xtype.TQ_FINTRINSIC);
    FunctionType ftype = new FunctionType(Xtype.FintType, Xtype.TQ_FINTRINSIC);
    Ident fident = getEnv().declIntrinsicIdent(fname, ftype);
    return fident.Call(Xcons.List(expr));
  }

  private Xobject _convTypeByIntrinsicCall(String fname, Xobject expr, Xobject kind)
  {
    //FunctionType ftype = new FunctionType(Xtype.FunspecifiedType, Xtype.TQ_FINTRINSIC);
    FunctionType ftype = new FunctionType(Xtype.FintType, Xtype.TQ_FINTRINSIC);
    Ident fident = getEnv().declIntrinsicIdent(fname, ftype);
    return fident.Call(Xcons.List(expr, kind));
  }
  **********************************************************************/

  private int _getExprRank() {
    int hostRank = coarray.getRank();

    if (subscripts == null)
      return hostRank;

    int count = 0;
    for (int i = 0; i < hostRank; i++)
      if (isTripletIndex(i))
        ++count;
    return count;
  }


  //------------------------------
  //  run: GET communication
  //------------------------------
  public Xobject toFuncRef() {
    Xtype type = getType();

    switch (getType().getKind()) {
    case Xtype.BASIC:
    case Xtype.STRUCT:                      // scalar struct
    case Xtype.F_ARRAY:                     // array struct
      Xobject mold = removeCoindex();
      return toFuncRef_core(mold);

    default:
      XMP.fatal("INTERNAL: unexpected type kind (XMPcoindexObj:toFuncRef)");
    }

    return obj;
  }


  private Xobject toFuncRef_struct___radical____() {
    // "character(len=1) :: xmpf_moldchar(0)"
    // coindexed obj a[k] -->
    //    transfer(
    //      xmpf_coarray_get0d_any(
    //        DP_a, k, transfer(a, xmpf_moldchar))
    //      

    // transfer(get_as_string, obj)

    // call runtime as character(len=1), dimension(sizeof(obj))
    Xobject mold = Xcons.FcharacterConstant(Xtype.FcharacterType, " ", null);
    Xobject funcRef = toFuncRef_core(mold);  

    // cast operation
    Ident transferId = _declIntIntrinsicIdent("transfer");
    Xobject castExpr = transferId.Call(Xcons.List(funcRef, getIdent()));

    return castExpr;
  }


  Xobject toFuncRef_core(Xobject mold) {
    switch (GetInterfaceType) {
    case 6:
      return toFuncRef_core_type6(mold, COARRAYGET_PREFIX);
    case 8:
      return toFuncRef_core_type8(mold, COARRAYGET_GENERIC_NAME);
    default:
      XMP.fatal("INTERNAL: obsoleted Get Interface Type: " +
                GetInterfaceType);
      break;
    }
    return null;
  }

  private Xobject toFuncRef_core_type8(Xobject mold, String funcName) {
    // type8 used
    Xobject actualArgs = _makeActualArgs_type8(mold);

    Ident funcIdent = getEnv().findVarIdent(funcName, null);
    if (funcIdent == null) {
      Xtype baseType = new BasicType(BasicType.F_NUMERIC_ALL);
                                            // ^^^^^^^^^^^^^ 
                                            // should be F_ALL or F_GENERIC if possible
      Xtype funcType = Xtype.Function(baseType);
      funcIdent = getEnv().declExternIdent(funcName, funcType);
    }                                           
    Xobject funcRef = funcIdent.Call(actualArgs);
    return funcRef;
  }

  private Xobject toFuncRef_core_type6(Xobject mold, String funcPrefix) {
    // type6 used
    Xobject actualArgs = _makeActualArgs_type6(mold);

    Xtype xtype = getType().copy();
    xtype.removeCodimensions();

    String funcName = funcPrefix + exprRank + "d";
    Ident funcIdent = getEnv().findVarIdent(funcName, null);
    if (funcIdent == null) {
      // bug460: funcIdent could not find because the module declaring the
      // name is not defined in the same file.
      Xtype baseType = _getBasicType(xtype);   // regard type as its basic type
                                               //// RETHINK! It might be neutral type?
      Xtype funcType = Xtype.Function(baseType);
      funcIdent = getEnv().declExternIdent(funcName, funcType);
      // This workaround for bug460 does not work well.
      //funcIdent = Ident.Fident(funcName, null);
    }                                           

    Xobject funcRef = funcIdent.Call(actualArgs);
    return funcRef;
  }


  //------------------------------
  //  run: PUT communication
  //------------------------------
  public Xobject toCallStmt(Xobject rhs, Xobject condition) {
    Xtype type = getType();
    Xobject subrStmt;
    Xobject mold;

    /*************************
    if (type.isStruct()) {
      XMP.fatal("Not supported type of coarray: " + getName());
      return null;
    } else {
    *************************/
    mold = removeCoindex();
    /**************************
    }
    *************************/

    switch (PutInterfaceType) {
    case 8:
      return toCallStmt_type8(mold, rhs, COARRAYPUT_GENERIC_NAME);
    case 7:
      return toCallStmt_type7(rhs, condition, COARRAYPUT_PREFIX);
    default:
      XMP.fatal("INTERNAL: obsoleted Get Interface Type: " +
                PutInterfaceType);
      break;
    }
    return null;
  }

  public Xobject toCallStmt_type8(Xobject mold, Xobject rhs, String subrName) {
    // type8 used
    Xobject actualArgs = _makeActualArgs_type8(mold, _convRhsType(rhs));

    Ident subrIdent = getEnv().findVarIdent(subrName, null);
    if (subrIdent == null) {
      subrIdent = getEnv().declExternIdent(subrName,
                                           Xtype.FexternalSubroutineType);
    }
    Xobject subrCall = subrIdent.callSubroutine(actualArgs);
    return subrCall;
  }

  public Xobject toCallStmt_type7(Xobject rhs, Xobject condition, String subrPrefix) {
    // type7 used
    Xobject actualArgs =
      _makeActualArgs_type7(_convRhsType(rhs), condition);

    // "scalar" or "array" or "spread" will be selected.
    String pattern = _selectCoarrayPutPattern(rhs);

    String subrName = subrPrefix + "_" + pattern;
    //// I'm not clear why this is OK and the case xmpf_coarray_proc_init is 
    //// not OK with the similar interface blocks.
    Ident subrIdent = getEnv().findVarIdent(subrName, null);
    if (subrIdent == null)
      subrIdent = getEnv().declExternIdent(subrName,
                                           BasicType.FexternalSubroutineType);
    Xobject subrCall = subrIdent.callSubroutine(actualArgs);

    return subrCall;
  }


  /***
      restriction: pattern spread is not supported yet.
   ***/
  private String _selectCoarrayPutPattern(Xobject rhs) {
    String pattern;

    if (exprRank == 0)                      // scalar = scalar
      pattern = "scalar";
    /*******************
    else if (rhs.getFrank() == 0)       // array = scalar
      pattern = "spread";
    *******************/
    else                                // array = array
      pattern = "array";

    return pattern;
  }


  /* generate actual arguments
   * cf. libxmpf/src/xmpf_coarray_put.c
   *
   * Type-8:
   *   utilize F90 generic-name interface
   *     COARRAYGET_GENERIC_NAME(descPtr, coindex, mold) result(dst)
   *     COARRAYPUT_GENERIC_NAME(descPtr, coindex, mold, src)
   *       integer(8), intent(in)            :: descPtr
   *       integer, intent(in)               :: coindex
   *       anytype&kind_anydim, intent(in)   :: mold
   *         local data corresponding to the coindexed remote object/variable
   *       same type&kind, same rank as mold :: src
   *       same type&kind, same rank as mold :: dst
   * Type-7:
   *   add baseAddr to Type-6 as an extra argument 
   *   in order to tell the optimization compiler the data will be referred.
   * Type-6:
   *       (void *descPtr, void* baseAddr, int element, int image,
   *        [void* rhs, int scheme,] int exprRank,
   *        void* nextAddr1, int count1,
   *        ...
   *        void* nextAddrN, int countN )
   *   where N is rank of the reference (0<=N<=15 in Fortran 2008).
   */

  // for subroutine atrimc_define
  public Xobject makeActualArgs(Xobject src) {
    return _makeActualArgs_type8(removeCoindex(), src);
  }

  private Xobject _makeActualArgs_type8(Xobject mold, Xobject src) {
    Xobject actualArgs = _makeActualArgs_type8();
    actualArgs.add(mold);
    actualArgs.add(src);
    return actualArgs;
  }
  private Xobject _makeActualArgs_type8(Xobject mold) {
    Xobject actualArgs = _makeActualArgs_type8();
    actualArgs.add(mold);
    return actualArgs;
  }
  private Xobject _makeActualArgs_type8() {
    XMPcoarray coarray = getCoarray();
    Xobject baseAddr = getBaseAddr();
    Xobject descPtr = coarray.getDescPointerIdExpr(baseAddr);
    Xobject coindex = coarray.getImageIndex(baseAddr, cosubscripts);
    Xobject actualArgs = Xcons.List(descPtr, coindex);

    if (actualArgs.hasNullArg())
      XMP.fatal("INTERNAL: generated null argument (_makeActualArgs_type8)");

    return actualArgs;
  }

  private Xobject _makeActualArgs_type7(Xobject addArg1) {
    return _makeActualArgs_type7(addArg1, null);
  }
  private Xobject _makeActualArgs_type7(Xobject addArg1, Xobject addArg2) {
    Xobject actualArgs = _makeActualArgs_type6(addArg1, addArg2);
    Xobject coarrayName =  Xcons.FvarRef(coarray.getIdent());
    actualArgs.add(coarrayName);
    return actualArgs;
  }

  private Xobject _makeActualArgs_type6(Xobject addArg1) {
    return _makeActualArgs_type6(addArg1, null);
  }
  private Xobject _makeActualArgs_type6(Xobject addArg1, Xobject addArg2) {
    XMPcoarray coarray = getCoarray();

    Xobject baseAddr = getBaseAddr();
    Xobject descPtr = coarray.getDescPointerIdExpr(baseAddr);
    Xobject locBaseAddr = getBaseAddr_type6();
    Xobject element = coarray.getElementLengthExpr_runtime();
    Xobject image = coarray.getImageIndex(baseAddr, cosubscripts);
    Xobject actualArgs =
      Xcons.List(descPtr, locBaseAddr, element, image);

    if (addArg1 != null)
      actualArgs.add(addArg1);
    if (addArg2 != null)
      actualArgs.add(addArg2);

    actualArgs.add(Xcons.IntConstant(exprRank));

    int hostRank = coarray.getRank();
    for (int i = 0; i < hostRank; i++) {
      if (isTripletIndex(i)) {
        actualArgs.add(getNeighboringAddr_type6(i));
        actualArgs.add(getSizeFromTriplet(i));
      }
    }

    if (actualArgs.hasNullArg())
      XMP.fatal("INTERNAL: generated null argument (_makeActualArgs_type6)");

    return actualArgs;
  }


  // TEMPORARY VERSION (for Type 6)
  private Xtype _getBasicType(Xtype xtype) {

    int baseTypeCode = 0;
    switch (xtype.getKind()) {
    case Xtype.F_ARRAY:
      baseTypeCode = xtype.getRef().getBasicType();
      break;
    case Xtype.BASIC:
      baseTypeCode = xtype.getBasicType();
      break;
    case Xtype.STRUCT:
      XMP.fatal("internal error: STRUCT unsupported in _getTypeSuffix()");
      break;
    default:
      XMP.fatal("internal error: unexpected kind in _getTypeSuffix(): xtype.getKind()");
      break;
    }

    return new BasicType(baseTypeCode);
  }


  /* generate actual arguments
   * cf. libxmpf/src/xmpf_coarray_put.c
   *
   * Type-5:
   *       (void *descptr, void* baseAddr, int element, int image,
   *        [void* rhs, int scheme,] int exprRank,
   *        void* nextAddr1, int count1,
   *        ...
   *        void* nextAddrN, int countN )
   *   where N is rank of the reference (0<=N<=15 in Fortran 2008).
   */
  private Xobject _makeActualArgs_type5() {
    return _makeActualArgs_type5(null, null);
  }

  private Xobject _makeActualArgs_type5(Xobject rhs, Xobject condition) {
    XMPcoarray coarray = getCoarray();

    Xobject baseAddr = getBaseAddr();
    Xobject descPtr = coarray.getDescPointerIdExpr(baseAddr);
    Xobject element = coarray.getElementLengthExpr_runtime();
    Xobject image = coarray.getImageIndex(baseAddr, cosubscripts);
    Xobject actualArgs = Xcons.List(descPtr, baseAddr, element, image);

    if (rhs != null)
      actualArgs.add(_convRhsType(rhs));
    if (condition != null)
      actualArgs.add(condition);

    actualArgs.add(Xcons.IntConstant(exprRank));

    int hostRank = coarray.getRank();
    for (int i = 0; i < hostRank; i++) {
      if (isTripletIndex(i)) {
        actualArgs.add(getNeighboringAddr(i));
        actualArgs.add(getSizeFromTriplet(i));
      }
    }

    if (actualArgs.hasNullArg())
      XMP.fatal("INTERNAL: generated null argument (_makeActualArgs_type5)");

    return actualArgs;
  }


  //------------------------------
  //  inquirement and evaluation
  //------------------------------
  public Xobject getImageIndex() {
    return coarray.getImageIndex(getBaseAddr(), cosubscripts);
  }

  public Boolean isScalarIndex(int i) {
    Xobject subscr = subscripts.getArg(i);
    if (subscr.Opcode() != Xcode.F_ARRAY_INDEX)
      return (subscr.getFrank(getBlock()) == 0);
    return false;
  }

  public Boolean isVectorIndex(int i) {
    Xobject subscr = subscripts.getArg(i);
    if (subscr.Opcode() != Xcode.F_ARRAY_INDEX) {
      switch (subscr.getFrank(getBlock())) {
      case 0: return false;
      case 1: return true;
      default:
        XMP.fatal("unexpected subscript expression: "+subscr);
      }
    }
    return false;
  }

  public Boolean isTripletIndex(int i) {
    Xobject subscr = subscripts.getArg(i);
    return (subscr.Opcode() == Xcode.F_INDEX_RANGE);
  }

  public Xobject getBaseAddr() {
    return getNeighboringAddr(-1);
  }

  public Xobject getBaseAddr_type6() {
    return getNeighboringAddr_type6(-1);
  }

  /* get address of
      a(i[0], ..., i[hostRank-1])  for rank=-1
      a(i[0], ..., i[rank]+stride[axis], ..., i[hostRank-1]) for rank>=0
  */
  public Xobject getNeighboringAddr(int axis) {
    int hostRank = coarray.getRank();

    if (hostRank == 0) {
      // if host variable is a scalar
      return Xcons.FvarRef(coarray.getIdent());
    }

    Xobject arrElemRef = Xcons.FarrayRef(coarray.getIdent().Ref());

    for (int i = 0; i < hostRank; i++) {
      Xobject start = getStart(i);
      if (i == axis) {
        start = Xcons.binaryOp(Xcode.PLUS_EXPR,
                               start, 
                               getStride(i));
      }
      Xobject subscr = Xcons.FarrayIndex(start);
      arrElemRef.getArg(1).setArg(i, subscr);
    }

    return arrElemRef;
  }

  public Xobject getNeighboringAddr_type6(int axis) {
    Xobject arrElem = getNeighboringAddr(axis);

    Ident locId = _declInt8IntrinsicIdent("loc");
    return locId.Call(Xcons.List(arrElem));
  }


  public Xobject getStart(int i) {
    Xobject subscr = subscripts.getArg(i);
    Xobject start;
    switch(subscr.Opcode()) {
    case F_ARRAY_INDEX:         // scalar
      start = subscr.getArg(0);
      break;
    case F_INDEX_RANGE:         // triplet
      start = subscr.getArg(0);
      if (start == null) 
        start = coarray.getLbound(i);
      break;
    default:        // vector subscript is not supported
      XMP.fatal("internal error: unexpected Xcode: "+subscr.Opcode());
      start = null;
      break;
    }
    return start;
  }


  public Xobject getEnd(int i) {
    Xobject subscr = subscripts.getArg(i);
    Xobject end;
    switch(subscr.Opcode()) {
    case F_ARRAY_INDEX:         // scalar
      end = subscr.getArg(0);
      break;
    case F_INDEX_RANGE:         // triplet
      end = subscr.getArg(1);
      if (end == null) 
        end = coarray.getUbound(i);
      break;
    default:        // vector subscript is not supported
      XMP.fatal("internal error: unexpected Xcode: "+subscr.Opcode());
      end = null;
      break;
    }
    return end;
  }

  public Xobject getStride(int i) {
    Xobject subscr = subscripts.getArg(i);
    Xobject stride;
    switch(subscr.Opcode()) {
    case F_ARRAY_INDEX:         // scalar
      stride = Xcons.IntConstant(1);
      break;
    case F_INDEX_RANGE:         // triplet
      stride = subscr.getArg(2);
      if (stride == null)
        stride = Xcons.IntConstant(1);
      break;
    default:        // vector subscript is not supported
      XMP.fatal("internal error: unexpected Xcode: "+subscr.Opcode());
      stride = null;
      break;
    }
    return stride;
  }


  public Xobject getSizeFromTriplet(int i) {
    Xobject subscr = subscripts.getArg(i);
    Xobject size;
    switch(subscr.Opcode()) {
    case F_ARRAY_INDEX:         // scalar
      size = Xcons.IntConstant(1);
      break;

    case F_INDEX_RANGE:         // triplet
      Xobject i1 = subscr.getArg(0);  // can be null
      Xobject i2 = subscr.getArg(1);  // can be null
      Xobject i3 = subscr.getArg(2);  // can be null
      size = coarray.getSizeFromTriplet(i, i1, i2, i3);

      if (size != null) {            // success
        // call type conversion function for safe
        Ftype int4 = new Ftype(BasicType.INT, 4, getBlock());
        size = _typeConv(size, int4);
        break;
      }

      // The following codes until the break statement are not needed
      // if you could guarantee the value of size is always non-null
      // even for allocatable arrays.

      // for case a(:), 
      // use a Fortran intrinsic function size.
      if (i1 == null && i2 == null && i3 == null) {
        Xobject arg1 = Xcons.Symbol(Xcode.VAR, name);
        Xobject arg2 = Xcons.IntConstant(i + 1);
        Ident sizeId = _declIntIntrinsicIdent("size");
        size = sizeId.Call(Xcons.List(arg1, arg2));
        break;
      }

      // for case a(i1:), a(:i2), a(::i3), a(i1::i3) and a(:i2:i3),
      // retry coarray.getSizeFromTriplet(i, i1, i2, i3) with
      // intrinsic functions lbound and ubound.
      if (i1 == null) i1 = coarray.getLbound(i);
      if (i2 == null) i2 = coarray.getUbound(i);
      size = coarray.getSizeFromTriplet(i, i1, i2, i3);

      if (size != null) {
        // cast function for safe
        if (!size.Type().isBasic() &&
            size.Type().getBasicType() != BasicType.INT) {
          Ident intId = _declIntIntrinsicIdent("int");
          size = intId.Call(Xcons.List(size));
        }
      }
      break;

    default:        // vector subscript is not supported
      XMP.fatal("internal error: maybe vector subscript. Xcode: "
                + subscr.Opcode());
      size = null;
      break;
    }
    return size;
  }    

  public int getTotalArraySize() {
    return getTotalArraySize(getBlock());
  }
  public int getTotalArraySize(Block block) {
    Xobject size = getTotalArraySizeExpr(block);
    if (size != null || ! size.isIntConstant())
      return -1;
    return size.getInt();
  }

  public Xobject getTotalArraySizeExpr() {
    return getTotalArraySizeExpr(getBlock());
  }
  public Xobject getTotalArraySizeExpr(Block block) {
    return obj.Type().getTotalArraySizeExpr(block);
  }


  //------------------------------
  //  tool
  //------------------------------
  private Ident _declIntIntrinsicIdent(String name) { 
    FunctionType ftype = new FunctionType(Xtype.FintType, Xtype.TQ_FINTRINSIC);
    Ident ident = getEnv().declIntrinsicIdent(name, ftype);
    return ident;
  }

  private Ident _declInt8IntrinsicIdent(String name) { 
    FunctionType ftype = new FunctionType(Xtype.Fint8Type, Xtype.TQ_FINTRINSIC);
    Ident ident = getEnv().declIntrinsicIdent(name, ftype);
    return ident;
  }

  //------------------------------
  //  inquire
  //------------------------------
  public static Boolean isGETfunc(Xobject xobj) {
    if (xobj.Opcode() == Xcode.FUNCTION_CALL) {

      if (COARRAYGET_GENERIC_NAME.equals(xobj.getName()))
        return true;
    }
    return false;
  }

  public Xobject getObj() {
    return obj;
  }

  public String getName() {
    return name;
  }

  public Xtype getType() {
    return coarray.getType();
  }

  public Ident getIdent() {
    return coarray.getIdent();
  }

  public XMPcoarray getCoarray() {
    return coarray;
  }

  // my mold object corresponding to the coindex object
  //
  public Xobject removeCoindex() {
    this.obj = _removeCoindex(obj);
    return this.obj;
  }

  private Xobject _removeCoindex(Xobject obj) {
    Xobject obj1, obj2, obj_out;

    switch (obj.Opcode()) {
    case CO_ARRAY_REF:              // v[k..] --> v
      obj_out = obj.getArg(0).getArg(0);
      return obj_out;

    case MEMBER_REF:              // remove(..%b%c)    --> remove(..%b)%c
    case F_ARRAY_REF:             // remove(..%b(i..)) --> remove(..%b)(i..)
      obj1 = obj.getArg(0);
      obj2 = obj1.getArg(0);
      obj2 = _removeCoindex(obj2);
      obj1.setArg(0, obj2);
      return obj;

    default:
      break;
    }

    XMP.fatal("INTERNAL: unexpected Opcode (XMPcoindexObj:removeCoindex #3)");
    return obj;
  }


  public XMPenv getEnv() {
    return coarray.getEnv();
  }

  public Block getBlock() {
    return coarray.fblock;
  }

  public BlockList getBlockList() {
    return coarray.fblock.getBody();
  }

  public Xobject getDecls() {
    return getBlockList().getDecls();
  }

  public String toString() {
    return "XMPcoindexObj(" + obj.toString() + ")";
  }


}
