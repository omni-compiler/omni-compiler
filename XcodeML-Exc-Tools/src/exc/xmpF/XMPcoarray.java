package exc.xmpF;

import exc.object.*;
import exc.block.*;
import java.util.*;

/*
 * Madiator for each coarray
 */
public class XMPcoarray {
  // DEBUG
  private final static Boolean _DEBUG_ALIGN_ON = false;    // true or false

  // name of property 
  private final static String XMP_COARRAY_NODES_PROP = "XMP_COARRAY_NODES_PROP";

  // name of library
  public final static String VAR_DESCPOINTER_PREFIX = "xmpf_descptr";
  public final static String VAR_CRAYPOINTER_PREFIX = "xmpf_crayptr";
  public final static String CBLK_COARRAYS_PREFIX   = "xmpf_coarrayvar";
  public final static String VAR_COARRAYCOMM_PREFIX = "xmpf_coarraynodes";    // for COARRAY directive
  final static String XMPF_LCOBOUND = "xmpf_lcobound";
  final static String XMPF_UCOBOUND = "xmpf_ucobound";
  final static String XMPF_COSIZE = "xmpf_cosize";
  //  final static String GET_IMAGE_INDEX_NAME = "xmpf_coarray_get_image_index";
  final static String GET_IMAGE_INDEX_NAME = "xmpf_image_index_generic";
  //  final static String SET_COSHAPE_NAME = "xmpf_coarray_set_coshape";
  final static String SET_CORANK_NAME = "xmpf_coarray_set_corank";
  final static String SET_CODIMENSION_NAME = "xmpf_coarray_set_codim";
  final static String SET_VARNAME_NAME = "xmpf_coarray_set_varname";
  final static String GET_DESCR_ID_NAME = "xmpf_get_descr_id";
  final static String SET_NODES_NAME = "xmpf_coarray_set_nodes";    // for COARRAY directive
  final static String SET_IMAGE_NODES_NAME = "xmpf_coarray_set_image_nodes";  // for IMAGE directive

  final static String COUNT_SIZE_NAME = "xmpf_coarray_count_size";
  final static String ALLOC_STATIC_NAME = "xmpf_coarray_alloc_static";
  final static String REGMEM_STATIC_NAME = "xmpf_coarray_regmem_static";

  // original attributes
  private Ident ident;
  private String name;
  private FindexRange indexRange = null;
  private FindexRange coindexRange = null;
  //private Xtype originalType;
  private Boolean isAllocatable;
  private Boolean isPointer;
  private Boolean isUseAssociated;
  private Boolean _wasMovedFromModule = false;

  // corresponding cray pointer, descriptor and common block names
  private String _crayPtrName = null;
  private Ident crayPtrId = null;
  private String _descPtrName = null;
  private Ident descPtrId = null;
  private String homeBlockCodeName = null;    // see _getHomeBlockCodeName()
  private String _coarrayCommName = null;
  private Ident coarrayCommId = null;
  
  // corresponding nodes by coarray directive
  private String nodesName = null;
  private Ident nodesDescId = null;

  // context
  protected XMPenv env;
  protected XobjectDef def;
  protected FunctionBlock fblock;

  // strategy for each coarray
  /* true (memory allocation in library) if 
   *   memory mannager is Ver.3 or the coarray is derived type,
   * else false (not allocation but only registration in library)
   */
  private Boolean useMalloc = true;

  /**************************
      debugging tools
   ***************************/
  private void _DEBUG_ALIGN(String str) {
    if (_DEBUG_ALIGN_ON)
      System.out.println(str);
  }


  //------------------------------
  //  CONSTRUCTOR
  //------------------------------
  public XMPcoarray(Ident ident, XMPenv env)
  {
    this(ident, env.getCurrentDef(), env);
  }
  public XMPcoarray(Ident ident, FuncDefBlock funcDef, XMPenv env)
  {
    this(ident, funcDef.getDef(), funcDef.getBlock(), env);
  }
  public XMPcoarray(Ident ident, FuncDefBlock funcDef, XMPenv env,
                    String homeBlockCodeName)
  {
    this(ident, funcDef.getDef(), funcDef.getBlock(), env, homeBlockCodeName);
  }
  public XMPcoarray(Ident ident, XobjectDef def, FunctionBlock fblock, XMPenv env)
  {
    this.env = env;
    this.def = def;
    this.fblock = fblock;
    setIdentEtc(ident);
    _setHomeBlockCodeName();
  }
  public XMPcoarray(Ident ident, XobjectDef def, FunctionBlock fblock, XMPenv env,
                    String homeBlockCodeName)
  {
    this.env = env;
    this.def = def;
    this.fblock = fblock;
    setIdentEtc(ident);
    this.homeBlockCodeName = homeBlockCodeName;
  }


  private void _setHomeBlockCodeName()
  {
    String name, code;

    name = ident.getFdeclaredModule();
    if (name != null) {
      code = _getCodeFromName(name);
    }
    else {
      name = def.getName();
      code = _getCodeFromName(name);
      XobjectDef parent_def = def.getParent();
      while (parent_def != null) {
        name = parent_def.getName();
        code = _getCodeFromName(name) + "_" + code;
        parent_def = parent_def.getParent();
      }
    }

    homeBlockCodeName = code;
  }

  private String _getCodeFromName(String name)
  {
    int count = 0;
    for (int idx = name.indexOf("_");
         idx >= 0;
         idx = name.indexOf("_", idx + 1))
      ++count;

    if (count == 0)
      return name;
    return "" + count + name;
  }


  public void setUseMallocWithHint(Boolean useMalloc) {
    // useMalloc (memory manager Ver.3) is specified.
    if (useMalloc) {
      this.useMalloc = true;
      return;
    }
    // derived-type coarray is not supported in Ver.4.
    if (getIdent().Type().getKind() == Xtype.STRUCT) {
      this.useMalloc = true;
      return;
    }
    // otherwise, memory manager Ver.4 is used.
    this.useMalloc = false;
  }

  public Boolean usesMalloc() {
    return useMalloc;
  }

  //------------------------------
  //  semantic analysis:
  //    COARRAY directive
  //------------------------------
  public static void analyzeCoarrayDirective(Xobject coarrayPragma,
                                             XMPenv env, PragmaBlock pb) {

    String nodesName = coarrayPragma.getArg(0).getString();
    XobjList coarrayNameList = (XobjList)coarrayPragma.getArg(1);

    for(Xobject xobj: coarrayNameList){
      String coarrayName = xobj.getString();
      analyzeEachCoarray(coarrayName, nodesName, env, pb);
      if(XMP.hasError()) break;
    }
  }


  private static void analyzeEachCoarray(String name, String nodesName,
                                         XMPenv env, PragmaBlock pb) {

    Ident ident = env.findVarIdent(name, null);

    // error check #1
    if (ident == null || !ident.isCoarray()) {
      XMP.errorAt(pb, "not declared as a coarray variable: " + name);
      return;
    }

    // error check #2
    if (getProp_nodes(ident) != null) {
      XMP.errorAt(pb, "double-declaration in coarray directives: " + name);
      return;
    }

    setProp_nodes(ident, nodesName);
  }


  public static String getProp_nodes(Ident ident) {
    return (String)ident.getProp(XMP_COARRAY_NODES_PROP);
  }

  public static String getProp_nodes(Xobject xobj) {
    return (String)xobj.getProp(XMP_COARRAY_NODES_PROP);
  }

  private static void setProp_nodes(Ident ident, String nodesName) {
    ident.setProp(XMP_COARRAY_NODES_PROP, nodesName);
  }

  private static void setProp_nodes(Xobject xobj, String nodesName) {
    xobj.setProp(XMP_COARRAY_NODES_PROP, nodesName);
  }


  //-----------------------------------------------------
  //  A part of TRANSLATION c.  (for Ver.3) 
  //  A part of TRANSLATION c7. (for Ver.7)
  //  declare cray-pointer variable correspoinding to this.
  //-----------------------------------------------------
  //
  public void genDecl_crayPointer() {
    genDecl_crayPointer(false);
  }
  public void genDecl_crayPointer(Boolean saved) {
    BlockList blist = fblock.getBody();
    String crayPtrName = getCrayPointerName();

    StorageClass sclass;
    if (saved) {
      sclass = StorageClass.FSAVE;
      _setSaveAttrInType(ident.Type());
    } else {
      sclass = StorageClass.FLOCAL;
    }

    // generate declaration of crayPtrId
    Xtype crayPtrType = Xtype.Farray(BasicType.Fint8Type);
    crayPtrType.setIsFcrayPointer(true);
    crayPtrId = blist.declLocalIdent(crayPtrName,
                                     crayPtrType,
                                     sclass,
                                     Xcons.FvarRef(ident));  // ident.Ref() if C
  }


  // declare variable of descriptor pointer corresponding to this.
  //
  public void genDecl_descPointer() {
    if(descPtrId != null) {
      return;
    }

    String descPtrName = getDescPointerName();
    //BlockList blist = fblock.getBody();
    
    descPtrId = env.declInternIdent(descPtrName,
                                    BasicType.Fint8Type);
  }


  //-----------------------------------------------------
  //  A part of TRANSLATION b.
  //  generate "CALL coarray_count_size(count, elem)"
  //-----------------------------------------------------
  //
  public Xobject makeStmt_countCoarrays()
  {
    BlockList blist = fblock.getBody();
    return makeStmt_countCoarrays(blist);
  }

  public Xobject makeStmt_countCoarrays(BlockList blist)
  {
    Xobject elem = getElementLengthExpr_atmost();

    if (elem == null) {
      XMP.error("current restriction: " + 
                "could not find the element length of: "+getName());
    }

    int count = getTotalArraySize();
    Xobject args = Xcons.List(Xcons.IntConstant(count), elem);
    Ident subr = blist.declLocalIdent(COUNT_SIZE_NAME,
                                      BasicType.FexternalSubroutineType);

    return subr.callSubroutine(args);
  }


  //-----------------------------------------------------
  //  A part of TRANSLATION b. (for Ver. 4, 6 and 7/FJ&MPI3)
  //  generate "CALL coarray_regmem_static(descPtr_var, LOC(var), ... )"
  //-----------------------------------------------------
  //
  public Xobject makeStmt_regmemStatic()
  {
    BlockList blist = fblock.getBody();
    return makeStmt_regmemStatic(blist);
  }

  public Xobject makeStmt_regmemStatic(BlockList blist)
  {
    String subrName = REGMEM_STATIC_NAME;
    Ident subrIdent =
      blist.declLocalIdent(subrName, BasicType.FexternalSubroutineType);

    // arg2
    FunctionType ftype = new FunctionType(Xtype.Fint8Type, Xtype.TQ_FINTRINSIC);
    Ident locId = env.declIntrinsicIdent("loc", ftype);
    Xobject locCall = locId.Call(Xcons.List(getIdent()));

    // get args
    Xobject args = _getCommonArgs(locCall);

    // CALL stmt
    return subrIdent.callSubroutine(args);
  }

   
  //-----------------------------------------------------
  //  A part of TRANSLATION b. (for Ver. 3 and 7/GASNet)
  //  generate "CALL coarray_alloc_static(descPtr_var, crayPtr_var, ... )"
  //-----------------------------------------------------
  //
  public Xobject makeStmt_allocStatic()
  {
    BlockList blist = fblock.getBody();
    return makeStmt_allocStatic(blist);
  }

  public Xobject makeStmt_allocStatic(BlockList blist)
  {
    String subrName = ALLOC_STATIC_NAME;
    Ident subrIdent =
      blist.declLocalIdent(subrName, BasicType.FexternalSubroutineType);

    // arg2
    Ident crayPtrId = getCrayPointerId();

    // get args
    Xobject args = _getCommonArgs(crayPtrId);

    // CALL stmt
    return subrIdent.callSubroutine(args);
  }


  // common arguments
  //
  private Xobject _getCommonArgs(Xobject arg2)
  {
    // arg1
    Ident descPtr = getDescPointerId();
    // arg3
    Xobject count = getTotalArraySizeExpr();
    // arg4
    Xobject elem = getElementLengthExpr_atmost();
    if (elem==null)
      XMP.fatal("elem must not be null.");
    // arg6
    String varName = getName();
    Xobject varNameObj = 
      Xcons.FcharacterConstant(Xtype.FcharacterType, varName, null);
    // arg5
    Xobject nameLen = Xcons.IntConstant(varName.length());

    // args
    Xobject args = Xcons.List(descPtr,
                              arg2,
                              count,
                              elem,
                              nameLen,
                              varNameObj);
    if (args.hasNullArg())
      XMP.fatal("INTERNAL: contains null argument");

    return args;
  }


  //-----------------------------------------------------
  //  A part of TRANSLATION m. with XMPenv
  //  generate
  //     "CALL set_corank(descPtr, corank)"
  //     "CALL set_codim (descPtr, 0, clb, cub)"
  //     ...
  //     "CALL set_codim (descPtr, corank-1, clb)"
  //  returns null if it is not allocated
  //-----------------------------------------------------
  //
  /******************************
  public Xobject makeStmt_setCoshape() {
    return makeStmt_setCoshape(env);
  }
  public Xobject makeStmt_setCoshape(XMPenv env) {
    int corank = getCorank();

    Xobject args = Xcons.List(getDescPointerId(),
                              Xcons.IntConstant(corank));
    for (int i = 0; i < corank - 1; i++) {
      args.add(getLcobound(i));
      args.add(getUcobound(i));
    }
    args.add(getLcobound(corank - 1));

    if (args.hasNullArg())
      XMP.fatal("generated null argument " + SET_COSHAPE_NAME +
                "(makeStmt_setCoshape())");

    Ident subr = env.findVarIdent(SET_COSHAPE_NAME, null);
    if (subr == null) {
      subr = env.declExternIdent(SET_COSHAPE_NAME,
                                 BasicType.FexternalSubroutineType);
    }
    Xobject subrCall = subr.callSubroutine(args);
    return subrCall;
  }
  ***********************************/

  public void addStmts_setCoshapeAndName(BlockList list, XMPenv env) {
    addStmts_setCoshape(list, env);
    Xobject subrCall = makeStmt_setVarName();
    list.add(subrCall);
  }

  public void addStmts_setCoshape(BlockList list) {
    addStmts_setCoshape(list, env);
  }
  
  public void addStmts_setCoshape(BlockList list, XMPenv env) {
    Xobject stmt;
    stmt = makeStmt_setCorank(env);
    list.add(stmt);

    int corank = getCorank();
    for (int i = 0; i < corank; i++) {
      stmt = makeStmt_setCodimension(i, env);
      list.add(stmt);
    }
  }    

  public void addStmts_setCoshape(ArrayList<Xobject> list) {
    addStmts_setCoshape(list, env);
  }
  
  public void addStmts_setCoshape(ArrayList<Xobject> list, XMPenv env) {
    Xobject stmt;
    stmt = makeStmt_setCorank(env);
    list.add(stmt);

    int corank = getCorank();
    for (int i = 0; i < corank; i++) {
      stmt = makeStmt_setCodimension(i, env);
      list.add(stmt);
    }
  }    

  public void addStmts_setCoshape(XobjList list) {
    addStmts_setCoshape(list, env);
  }
  
  public void addStmts_setCoshape(XobjList list, XMPenv env) {
    Xobject stmt;
    stmt = makeStmt_setCorank(env);
    list.add(stmt);

    int corank = getCorank();
    for (int i = 0; i < corank; i++) {
      stmt = makeStmt_setCodimension(i, env);
      list.add(stmt);
    }
  }    

  public Xobject makeStmt_setCorank() {
    return makeStmt_setCorank(env);
  }
  public Xobject makeStmt_setCorank(XMPenv env) {
    int corank = getCorank();

    Xobject args = Xcons.List(getDescPointerId(),
                              Xcons.IntConstant(corank));

    Ident subr = env.findVarIdent(SET_CORANK_NAME, null);
    if (subr == null) {
      subr = env.declExternIdent(SET_CORANK_NAME,
                                 BasicType.FexternalSubroutineType);
    }
    Xobject subrCall = subr.callSubroutine(args);
    return subrCall;
  }

  public Xobject makeStmt_setCodimension(int dim) {
    return makeStmt_setCodimension(dim, env);
  }
  public Xobject makeStmt_setCodimension(int dim, XMPenv env) {
    int corank = getCorank();

    Xobject args = Xcons.List(getDescPointerId(),
                              Xcons.IntConstant(dim));

    if (dim < corank - 1) {            // not last dimension
      args.add(getLcobound(dim));
      args.add(getUcobound(dim));
    } else if (dim == corank - 1) {    // last dimension
      args.add(getLcobound(dim));
      args.add(Xcons.IntConstant(-999));  // dummy
    } else {                           // illegal
      XMP.fatal("INTERNAL: dimension number specified larger than corank " +
                SET_CODIMENSION_NAME + "(makeStmt_setCodimension(dim, env))");
    }

    Ident subr = env.findVarIdent(SET_CODIMENSION_NAME, null);
    if (subr == null) {
      subr = env.declExternIdent(SET_CODIMENSION_NAME,
                                 BasicType.FexternalSubroutineType);
    }
    Xobject subrCall = subr.callSubroutine(args);
    return subrCall;
  }


  //-----------------------------------------------------
  //  A part of TRANSLATION m. with static coshape
  //  generate
  //     "CALL set_corank(descPtr, corank)"
  //     "CALL set_codim (descPtr, 0, clb, cub)"
  //     ...
  //     "CALL set_codim (descPtr, corank-1, clb)"
  //-----------------------------------------------------
  //
  /*********************************************
  public Xobject makeStmt_setCoshape(XobjList coshape) {
    int corank = getCorank();
    if (corank != coshape.Nargs()) {
      XMP.fatal("number of codimensions not matched with the declaration:"
                + corank + " and " + coshape.Nargs());
      return null;
    }

    Xobject args = Xcons.List(getDescPointerId(),
                              Xcons.IntConstant(corank));
    for (int i = 0; i < corank - 1; i++) {
      args.add(_getLboundInIndexRange(coshape.getArg(i)));
      args.add(_getUboundInIndexRange(coshape.getArg(i)));
    }
    args.add(_getLboundInIndexRange(coshape.getArg(corank - 1)));
    if (args.hasNullArg())
      XMP.fatal("generated null argument " + SET_COSHAPE_NAME + 
                " (XMPcoarray.makeStmt_setCoshape(coshape))");

    Ident subr = env.findVarIdent(SET_COSHAPE_NAME, null);
    if (subr == null) {
      subr = env.declExternIdent(SET_COSHAPE_NAME,
                                 BasicType.FexternalSubroutineType);
    }
    Xobject subrCall = subr.callSubroutine(args);
    return subrCall;
  }
  *********************************************************/

  public void addStmts_setCoshape(BlockList list, XobjList coshape) {
    Xobject stmt;
    stmt = makeStmt_setCorank(coshape);
    list.add(stmt);

    int corank = getCorank();
    for (int i = 0; i < corank; i++) {
      stmt = makeStmt_setCodimension(i, coshape);
      list.add(stmt);
    }
  }    
  public void addStmts_setCoshape(ArrayList<Xobject> list, XobjList coshape) {
    Xobject stmt;
    stmt = makeStmt_setCorank(coshape);
    list.add(stmt);

    int corank = getCorank();
    for (int i = 0; i < corank; i++) {
      stmt = makeStmt_setCodimension(i, coshape);
      list.add(stmt);
    }
  }    

  public Xobject makeStmt_setCorank(XobjList coshape) {
    int corank = getCorank();
    if (corank != coshape.Nargs()) {
      XMP.fatal("number of codimensions not matched with the declaration:"
                + corank + " and " + coshape.Nargs());
      return null;
    }

    return makeStmt_setCorank(env);
  }


  public Xobject makeStmt_setCodimension(int dim, XobjList coshape) {
    int corank = getCorank();

    Xobject args = Xcons.List(getDescPointerId(),
                              Xcons.IntConstant(dim));

    if (dim < corank - 1) {            // not last dimension
      args.add(_getLboundInIndexRange(coshape.getArg(dim)));
      args.add(_getUboundInIndexRange(coshape.getArg(dim)));
    } else if (dim == corank - 1) {    // last dimension
      args.add(_getLboundInIndexRange(coshape.getArg(dim)));
      args.add(Xcons.IntConstant(-998));  // dummy
    } else {                           // illegal
      XMP.fatal("INTERNAL: dimension number specified larger than corank " +
                SET_CODIMENSION_NAME + "(makeStmt_setCodimension(dim, coshape))");
    }

    Ident subr = env.findVarIdent(SET_CODIMENSION_NAME, null);
    if (subr == null) {
      subr = env.declExternIdent(SET_CODIMENSION_NAME,
                                 BasicType.FexternalSubroutineType);
    }
    Xobject subrCall = subr.callSubroutine(args);
    return subrCall;
  }


  private Xobject _getLboundInIndexRange(Xobject dimension) {
    Xobject lbound;

    if (dimension == null)
      lbound = null;
    else {
      switch (dimension.Opcode()) {
      case F_INDEX_RANGE:
        lbound = dimension.getArg(0);
        break;
      case F_ARRAY_INDEX:
        lbound = null;
        break;
      default:
        lbound = null;
        break;
      }
    }

    if (lbound == null)
      return Xcons.IntConstant(1);

    return lbound.cfold(fblock);
  }


  private Xobject _getUboundInIndexRange(Xobject dimension) {
    Xobject ubound;

    if (dimension == null)
      ubound = null;
    else {
      switch (dimension.Opcode()) {
      case F_INDEX_RANGE:
        ubound = dimension.getArg(1);
        break;
      case F_ARRAY_INDEX:
        ubound = dimension.getArg(0);
        break;
      default:
        ubound = dimension;
      }
    }

    if (ubound == null)
      XMP.fatal("illegal upper bound specified in ALLOCATE statement");

    return ubound.cfold(fblock);
  }


  //-----------------------------------------------------
  //  A part of TRANSLATION n.
  //  generate "CALL set_varname(descPtr, namelen, name)"
  //-----------------------------------------------------
  //
  public Xobject makeStmt_setVarName() {
    return makeStmt_setVarName(env);
  }

  public Xobject makeStmt_setVarName(XMPenv env) {
    String varName = getName();
    Xobject varNameObj = 
      Xcons.FcharacterConstant(Xtype.FcharacterType, varName, null);
    Xobject varNameLen = 
      Xcons.IntConstant(varName.length());
    Xobject args = Xcons.List(getDescPointerId(),
                              varNameLen, varNameObj);
    if (args.hasNullArg())
      XMP.fatal("generated null argument " + SET_VARNAME_NAME +
                "(makeStmt_setVarName)");

    Ident subr = env.findVarIdent(SET_VARNAME_NAME, null);
    if (subr == null) {
      subr = env.declExternIdent(SET_VARNAME_NAME,
                                 BasicType.FexternalSubroutineType);
    }
    Xobject subrCall = subr.callSubroutine(args);
    return subrCall;
  }


  //-----------------------------------------------------
  //  For COARRAY directive in XMPtransPragma
  //  generate and add "CALL xmpf_coarray_set_nodes(descPtr, nodesDesc)"
  //-----------------------------------------------------
  //
  public void build_setMappingNodes(BlockList blist, Block block)
  {
    if (nodesDescId != null)
      blist.add(makeStmt_setMappingNodes(block));
  }

  public Xobject makeStmt_setMappingNodes(Block block)
  {
    // descPtrId must be declarad previously in the coarray pass
    if (descPtrId == null)
      descPtrId = env.findVarIdent(getDescPointerName(), block);

    Xobject args = Xcons.List(descPtrId, nodesDescId);
    Ident subr = env.findVarIdent(SET_NODES_NAME, null);
    if (subr == null) {
      subr = (block == null ) ?
             env.declExternIdent(SET_NODES_NAME,
                                 BasicType.FexternalSubroutineType) :
             block.getBody().declLocalIdent(SET_NODES_NAME,
                                 BasicType.FexternalSubroutineType);
    }
    Xobject subrCall = subr.callSubroutine(args);
    return subrCall;
  }


  //-----------------------------------------------------
  //  For IMAGE directive in XMPtransPragma
  //  generate and add "CALL xmpf_coarray_set_image_nodes(nodesDesc)"
  //-----------------------------------------------------
  //
  public static Xobject makeStmt_setImageNodes(String nodesName, XMPenv env)
  {
    return makeStmt_setImageNodes(nodesName, env, env.getCurrentDef().getBlock());
  }

  public static Xobject makeStmt_setImageNodes(String nodesName, XMPenv env, Block block)
  {
    //    Ident imageNodesId = _getNodesDescIdByName(nodesName, env,
    //                                               env.getCurrentDef().getBlock());
    XMPnodes nodes = env.findXMPnodes(nodesName, block);
    Ident imageNodesId = nodes.getDescId();

    Xobject args = Xcons.List(imageNodesId);
    Ident subr = env.findVarIdent(SET_IMAGE_NODES_NAME, block);
    if (subr == null) {
      subr = (block == null) ?
             env.declExternIdent(SET_IMAGE_NODES_NAME,
                                 BasicType.FexternalSubroutineType) :
             block.getBody().declLocalIdent(SET_IMAGE_NODES_NAME,
                                 BasicType.FexternalSubroutineType);
    }
    Xobject subrCall = subr.callSubroutine(args);
    return subrCall;
  }


  //------------------------------
  //  self error check
  //------------------------------
  public void errorCheck() {

    if (ident.isCoarray()) {  // if it is not converted yet
      if (isPointer()) {
        XMP.error("Coarray variable cannot be a pointer: " + name);
      }
      if (isDummyArg()) {
        if (isScalar() || isExplicitShape() || isAssumedSize() ||
            isAssumedShape() || isAllocatable())
          ;
        else
          XMP.error("Coarray dummy argument must be of explicit shape, assumed size, assumed shape, or allocatable: "
                    + name);
      }
    }

  }


  //------------------------------
  //  IndexRange (to be abolished)
  //------------------------------

  private void _setIndexRange() {
    Xobject[] shape = getShape();
    indexRange = new FindexRange(shape, fblock, env);
  }

  private void _setIndexRange(Block block, XMPenv env) {
    Xobject[] shape = getShape();
    indexRange = new FindexRange(shape, block, env);
  }

  public FindexRange getIndexRange() {
    if (indexRange == null)
      _setIndexRange();
    return indexRange;
  }


  //------------------------------
  //  CoindexRange
  //------------------------------

  private void _setCoindexRange() {
    Xobject[] shape = getCoshape();
    coindexRange = new FindexRange(shape, fblock, env);
  }

  private void _setCoindexRange(Block block, XMPenv env) {
    Xobject[] shape = getCoshape();
    coindexRange = new FindexRange(shape, block, env);
  }

  public FindexRange getCoindexRange() {
    if (coindexRange == null)
      _setCoindexRange();
    return coindexRange;
  }


  //------------------------------
  //  evaluate index
  //------------------------------
  public int getElementLengthOrNot() {
    Xobject elem = getElementLengthExpr_runtime(); 
    if (elem == null || !elem.isIntConstant())
      return -1;
    return elem.getInt();
  }

  public int getElementLength() {
    int elem = getElementLengthOrNot(); 
    if (elem < 0) {
      XMP.fatal("current restriction: " +
                "could not numerically evaluate the element length of: "+name);
    }
    return elem;
  }

  /******************************************************
  public Xobject getElementLengthExpr() {
    ////// SELECTIVE
    return getElementLengthExpr(true);   // statically
  }
  public Xobject getElementLengthExpr(Boolean staticEvaluation) {
    if (staticEvaluation)
      return getElementLengthExpr_atmost();
    else
      return getElementLengthExpr_runtime();
  }
  *********************************************************/

  /* static evaluation of the size of derived-type data element
   *  This result will be equal to or greater than the size that 
   *  the backend compiler will deside.
   */
  public Xobject getElementLengthExpr_atmost() {
    int length = getElementLength_atmost(ident.Type());
    return Xcons.IntConstant(length);
  }

  public int getElementLength_atmost(Xtype type) {
    switch (type.getKind()) {
    case Xtype.F_ARRAY:
      Xtype baseType = type.getBaseRefType();        // type BASIC or STRUCT
      return _getLength_atmost(baseType);

    default:
      break;
    }
    return _getLength_atmost(type);
  }

  private int _getLength_atmost(Xtype type) {
    if (type.isFpointer())
      return _getPointerComponentLength_atmost(type);
    else if (type.isFallocatable())
      return _getAllocatableComponentLength_atmost(type);

    // otherwize
    return _getStaticDataLength_atmost(type);
  }


  /* These values are desided to match gfortran. If they do not match other
   * Fortran compilers, it should be modified.
   *    ----------------------------------
   *       p  p(:) p(:,:) p(:,:,:) ...
   *       8   48    72      96    ...
   *    ----------------------------------
   */
  private int _getPointerComponentLength_atmost(Xtype type) {
    int rank, length;
    rank = getRank(type);
    if (rank == 0)
      length = 8;
    else
      length = 24 * rank + 24;
    return length;
  }

  /* These values are desided to match gfortran. If they do not match other
   * Fortran compilers, it should be modified.
   */
  private int _getAllocatableComponentLength_atmost(Xtype type) {
    return _getPointerComponentLength_atmost(type);
  }

  private int _getStaticDataLength_atmost(Xtype type) {
    switch (type.getKind()) {
    case Xtype.BASIC:
      return type.getElementLength(getFblock());    // see BasicType.java

    case Xtype.F_ARRAY:
      Xtype baseType = type.getBaseRefType();       // type BASIC or STRUCT
      int elemLen = getElementLength_atmost(baseType);
      int size = getTotalArraySize(type);
      return size * elemLen;

    case Xtype.STRUCT:
      _DEBUG_ALIGN("Into DerivedType " + type);
      int length = _getStructLength_atmost(type);
      _DEBUG_ALIGN("Out of DerivedType " + type);
      return length;

    case Xtype.UNION:
    case Xtype.FUNCTION:
    case Xtype.F_COARRAY:
    default:
      XMP.fatal("INTERNAL: unexpected Xtype kind (" + type.getKind() + ")");
      break;
    }

    return 0;   // illegal
  }


  private int _getNumElements(Xtype type, Block block) {
    int size;
    if (type.getKind() == Xtype.F_ARRAY)
      size = getTotalArraySize(type);
    else
      size = 1;

    return size;
  }
    

  private int _getStructLength_atmost(Xtype type) {
    int currentPos = 0;
    int largestBoundary = 1;
    for (Xobject member: type.getMemberList()) {
      Xtype type1 = member.Type();
      int elemLen1, numElems1;
      if (type1.isFpointer()) {
        elemLen1 = _getPointerComponentLength_atmost(type1);
        numElems1 = 1;
        _DEBUG_ALIGN("  pointer member:" + member +
                     ", length=" + elemLen1);
      } else if (type1.isFallocatable()) {
        elemLen1 = _getAllocatableComponentLength_atmost(type1);
        numElems1 = 1;
        _DEBUG_ALIGN("  allocatable member:" + member +
                     ", length=" + elemLen1);
      } else {
        elemLen1 = getElementLength_atmost(type1);
        numElems1 = _getNumElements(type1, getFblock());
        _DEBUG_ALIGN("  static member:" + member +
                     ", element length=" + elemLen1 +
                     ", num elements=" + numElems1);
      }

      // get boundary length for the member
      int boundary1;
      if (elemLen1 > 4)
        boundary1 = 8;
      else if (elemLen1 > 2)
        boundary1 = 4;
      else if (elemLen1 > 1)
        boundary1 = 2;
      else
        boundary1 = 1;

      // alignment for the member (round up)
      if (currentPos % boundary1 != 0) {
        currentPos = (currentPos/boundary1 + 1) * boundary1;
        _DEBUG_ALIGN("  skip for alignment upto "+ currentPos);
      }

      // proceed current position
      currentPos += elemLen1 * numElems1;
      _DEBUG_ALIGN("  proceed upto "+ currentPos);

      if (largestBoundary < boundary1)
        largestBoundary = boundary1;
    }

    // alignment for the structure (round up)
    if (currentPos % largestBoundary != 0) {
      currentPos = (currentPos/largestBoundary + 1) * largestBoundary;
    }
    _DEBUG_ALIGN("  finally proceed upto "+ currentPos);

    return currentPos;
  }


  /* build an expression for the size of the data element that
   * can be evaluated at runtime
   */
  public Xobject getElementLengthExpr_runtime() {
    Xobject lengthExpr = ident.Type().getElementLengthExpr(getFblock());    // see BasicType.java
    if (lengthExpr != null)
      return lengthExpr;

    // for derived type objects
    lengthExpr = _getDerivedTypeLengthExpr_runtime();
    return lengthExpr;
  }

  private Xobject _getDerivedTypeLengthExpr_runtime() {
    int rank = getRank();

    // build reference of the object
    Xobject elemRef;
    if (rank == 0) {          // scalar
      elemRef = Xcons.FvarRef(ident);
    } else {                  // array element eg. a(lb1,lb2,...)
      elemRef = Xcons.FarrayRef(ident.Ref());
      for (int i = 0; i < rank; i++) {
        Xobject lb = getLbound(i);
        Xobject subscr = Xcons.FarrayIndex(lb);
        elemRef.getArg(1).setArg(i, subscr);
      }
    }

    // build an expression to get sizeof elemRef
    Xobject lengthExpr = _buildSizeofExpr(elemRef);
    return lengthExpr;
  }


  /* build expression sizeof(data)
   * PROBLEM in BACKEND: extended intrinsic function sizeof is declared
   * with the attribute EXTERNAL anyway.
   */
  private Xobject _buildSizeofExpr(Xobject data) {
    Ident sizeofId = declIntExtendIntrinsicIdent("sizeof");
    Xobject size = sizeofId.Call(Xcons.List(data));
    return size;
  }

  /* NOT USED
   * tricky and low-performance but standard version
   *    size(transfer(data, (/" "/))
   */
  private Xobject _buildSizeofExpr__OLD__(Xobject data) {
    Ident sizeId = declIntIntrinsicIdent("size");
    Ident transferId = declIntIntrinsicIdent("transfer");
    Xobject arg21 = Xcons.FcharacterConstant(Xtype.FcharacterType, " ", null);
    Xobject arg2 = Xcons.List(Xcode.F_ARRAY_CONSTRUCTOR,
                              _getCharFarrayType(1),
                              arg21);
    Xobject transfer = transferId.Call(Xcons.List(data, arg2));
    Xobject size = sizeId.Call(Xcons.List(transfer));
    return size;
  }


  public int getTotalArraySize() {
    return getTotalArraySize(getIndexRange());
  }

  public int getTotalArraySize(Xtype type) {
    if (type.getKind() != Xtype.F_ARRAY)
      XMP.fatal("INTERNAL ERROR: FarrayType expected here");
    Xobject[] shape = getShape((FarrayType)type);
    FindexRange findexRange = new FindexRange(shape, getFblock(), getEnv());
    return getTotalArraySize(findexRange);
  }

  public int getTotalArraySize(FindexRange findexRange) {
    Xobject sizeExpr = getTotalArraySizeExpr(findexRange);
    if (!sizeExpr.isIntConstant()) {
      XMP.error("current restriction: " +
                "cannot numerically evaluate the total size of: "+name);
      return 0;
    }
    return sizeExpr.getInt();
  }

  public Xobject getTotalArraySizeExpr() {
    FindexRange findexRange = getIndexRange();
    return getTotalArraySizeExpr(findexRange);
  }

  public Xobject getTotalArraySizeExpr(Xtype type) {
    if (type.getKind() != Xtype.F_ARRAY)
      XMP.fatal("INTERNAL ERROR: FarrayType expected here");
    Xobject[] shape = getShape((FarrayType)type);
    FindexRange findexRange = new FindexRange(shape, getFblock(), getEnv());
    return getTotalArraySizeExpr(findexRange);
  }

  public Xobject getTotalArraySizeExpr(FindexRange findexRange) {
    Xobject sizeExpr = findexRange.getTotalArraySizeExpr();
    if (sizeExpr == null)
      XMP.error("current restriction: " +
                "cannot find the total size of: "+name);
    return sizeExpr;
  }


  //------------------------------
  //  inquire in Fortran terminology:
  //   rank, shape, lower/upper bound and size
  //------------------------------

  public int getRank() {
    return ident.Type().getNumDimensions();
  }

  private int getRank(Xtype ftype) {
    if (ftype.getKind() == Xtype.F_ARRAY)
      return ftype.getNumDimensions();
    return 0;
  }

  public Xobject[] getShape() {
    if (getRank() == 0)
      return new Xobject[0];

    FarrayType ftype = (FarrayType)ident.Type();
    return ftype.getFarraySizeExpr();
  }

  private Xobject[] getShape(FarrayType ftype) {
    return ftype.getFarraySizeExpr();
  }
    

  public Xobject getLboundStatic(int i) {
    if (isExplicitShape()) {
      FarrayType ftype = (FarrayType)ident.Type();
      return ftype.getLbound(i, fblock);
    }
    return null;
  }

  public Xobject getUboundStatic(int i) {
    if (isExplicitShape()) {
      FarrayType ftype = (FarrayType)ident.Type();
      return ftype.getUbound(i, fblock);
    }
    return null;
  }

  public Xobject getLbound(int i) {
    Xobject lbound = getLboundStatic(i);
    if (lbound == null) {
      // generate intrinsic function call "lbound(a,dim)"
      Xobject arg1 = Xcons.Symbol(Xcode.VAR, name);
      Xobject arg2 = Xcons.IntConstant(i + 1);
      Ident lboundId = declIntIntrinsicIdent("lbound");
      lbound = lboundId.Call(Xcons.List(arg1, arg2));
    }
    return lbound;
  }

  public Xobject getUbound(int i) {
    Xobject ubound = getUboundStatic(i);
    if (ubound == null) {
      // generate intrinsic function call "ubound(a,dim)"
      Xobject arg1 = Xcons.Symbol(Xcode.VAR, name);
      Xobject arg2 = Xcons.IntConstant(i + 1);
      Ident uboundId = declIntIntrinsicIdent("ubound");
      ubound = uboundId.Call(Xcons.List(arg1, arg2));
    }
    return ubound;
  }


  public Xobject getSizeFromLbUb(Xobject lb, Xobject ub) {
    return getIndexRange().getSizeFromLbUb(lb, ub);
  }

  public Xobject getSizeFromIndexRange(Xobject range) {
    Xobject i1 = range.getArg(0);
    Xobject i2 = range.getArg(1);
    Xobject i3 = range.getArg(2);
    return getIndexRange().getSizeFromTriplet(i1, i2, i3);
  }


  //public Xobject getSizeFromTriplet(Xobject i1, Xobject i2, Xobject i3)
  //{
  //  return getIndexRange().getSizeFromTriplet(i1, i2, i3);
  //}

  public Xobject getSizeFromTriplet(int i, Xobject i1, Xobject i2, Xobject i3) {
    return getIndexRange().getSizeFromTriplet(i, i1, i2, i3);
  }


  //------------------------------
  //  evaluation in Fortran terminology:
  //   corank, coshape, lower/upper cobound and cosize
  //------------------------------

  public int getCorank() {
    return ident.Type().getCorank();
  }
 
  public Xobject[] getCoshape() {
    return ident.Type().getCodimensions();
  }

  public Xobject getLcoboundStatic(int i) {
    FindexRange indexRange = getCoindexRange();
    return (indexRange == null) ? null : indexRange.getLbound(i);
  }

  public Xobject getUcoboundStatic(int i) {
    FindexRange indexRange = getCoindexRange();
    return (indexRange == null) ? null : indexRange.getUbound(i);
  }

  public Xobject getCosizeStatic(int i) {
    FindexRange indexRange = getCoindexRange();
    return (indexRange == null) ? null : indexRange.getExtent(i);
  }

  public Xobject getLcobound(int i) {
    Xobject lcobound = getLcoboundStatic(i);
    if (lcobound == null) {
      // generate intrinsic function call "xmpf_lcobound(serno, dim)"
      Xobject arg1 = descPtrId;
      Xobject arg2 = Xcons.IntConstant(i + 1);
      Ident lcoboundId = getEnv().findVarIdent(XMPF_LCOBOUND, null);
      if (lcoboundId == null)
        lcoboundId = getEnv().declExternIdent(XMPF_LCOBOUND, Xtype.FintFunctionType);
      lcobound = lcoboundId.Call(Xcons.List(arg1, arg2));
    }
    return lcobound;
  }

  public Xobject getUcobound(int i) {
    Xobject ucobound = getUcoboundStatic(i);
    if (ucobound == null) {
      // generate intrinsic function call "xmpf_ucobound(serno, dim)"
      Xobject arg1 = descPtrId;
      Xobject arg2 = Xcons.IntConstant(i + 1);
      Ident ucoboundId = getEnv().findVarIdent(XMPF_UCOBOUND, null);
      if (ucoboundId == null)
        ucoboundId = getEnv().declExternIdent(XMPF_UCOBOUND, Xtype.FintFunctionType);
      ucobound = ucoboundId.Call(Xcons.List(arg1, arg2));
    }
    return ucobound;
  }

  public Xobject getCosize(int i) {
    Xobject cosize = getCosizeStatic(i);
    if (cosize == null) {
      // generate intrinsic function call "xmpf_cosize(serno, dim)"
      Xobject arg1 = descPtrId;
      Xobject arg2 = Xcons.IntConstant(i + 1);
      Ident cosizeId = getEnv().findVarIdent(XMPF_COSIZE, null);
      if (cosizeId == null)
        cosizeId = getEnv().declExternIdent(XMPF_COSIZE, Xtype.FintFunctionType);
      cosize = cosizeId.Call(Xcons.List(arg1, arg2));
    }
    return cosize;
  }


  //------------------------------
  //  evaluation in Fortran terminology:
  //   image index
  //------------------------------
  public Xobject getImageIndex(Xobject baseAddr, Xobject cosubscripts) {
    Xobject imageIndex;

    // try to get better expression without runtime library call
    if (!isAllocatable()) {
      imageIndex = getImageIndex_opt(baseAddr, cosubscripts);
      if (imageIndex != null)   // success
        return imageIndex;
    }

    // default expression calling runtime library
    imageIndex = getImageIndex_default(baseAddr, cosubscripts);
    return imageIndex;
  }


  public Xobject getImageIndex_opt(Xobject baseAddr, Xobject cosubscripts) {
    int ndim = getCorank();
    Xobject imageIndex = null;

    int k;
    Xobject cosubscr, lcobound, cosize;
    Xobject sum;

    /* ndim==1:   cs[0]-lb[0]                                             +1
     * ndim==2:  (cs[1]-lb[1]) *sz[0]+(cs[0]-lb[0])                       +1
     * ndim==3: ((cs[2]-lb[2]) *sz[1]+(cs[1]-lb[1])) *sz[0]+(cs[0]-lb[0]) +1
     * ...
     */
    k = ndim - 1;
    // width[k] = cosubscr[k] - lcobound[k]
    cosubscr = cosubscripts.getArg(k).getArg(0);
    lcobound = getLcoboundStatic(k);
    sum = _calcMinusExprExpr(cosubscr, lcobound);
    if (sum == null)
      return null;

    for (k = ndim - 2; k >= 0; k -= 1) {
      // tmp1[k] = sum[k+1] * cosize[k]
      cosize = getCosizeStatic(k);
      Xobject tmp1 = _calcTimesExprExpr(sum, cosize);
      if (tmp1 == null)
        return null;

      // tmp2[k] = cosubscr[k] - lcobound[k]
      cosubscr = cosubscripts.getArg(k).getArg(0);
      lcobound = getLcoboundStatic(k);
      Xobject tmp2 = _calcMinusExprExpr(cosubscr, lcobound);
      if (tmp2 == null)
        return null;

      // sum[k] = tmp1[k] + tmp2[k]
      sum = _calcPlusExprExpr(tmp1, tmp2);
      if (sum == null)
        return null;
    }

    Xobject image = _calcIncExpr(sum); 
    return image;
  }


  public Xobject getImageIndex_default(Xobject baseAddr, Xobject cosubscripts) {
    String fname = GET_IMAGE_INDEX_NAME;
    Ident fnameId = getEnv().findVarIdent(fname, null);
    if (fnameId == null)
      fnameId = getEnv().declExternIdent(fname, Xtype.FintFunctionType);

    // Old interface for xmpf_coarray_get_image_index
    //    XobjList args = Xcons.List(getDescPointerIdExpr(baseAddr), 
    //                               Xcons.IntConstant(getCorank()));
    // Interface for xmpf_image_index_generic
    XobjList args = Xcons.List(getDescPointerIdExpr(baseAddr));
    for (Xobject cosubs: (XobjList)cosubscripts) {
      args.add(cosubs);
    }
    if (args.hasNullArg())
      XMP.fatal("generated null argument " + fname + "(getImageIndex)");

    return fnameId.Call(args);
  }




  //------------------------------
  //  arithmetic routine
  //------------------------------

  /*  returns (a2-a1+1).cfold()
   */
  Xobject _calcIntDiffExprExpr(Xobject a1, Xobject a2) {
    Xobject b1, b2;

    if (a1 == null || a2 == null)
      return null;
    b1 = a1.cfold(fblock);
    b2 = a2.cfold(fblock);

    // short cut
    if (b2.equals(b1))
      return Xcons.IntConstant(1);

    Xobject result;
    if (b1.isIntConstant()) {
      if (b2.isIntConstant()) {
        // int(b2)-int(b1)+1
        int extent = b2.getInt() - b1.getInt() + 1;
        return Xcons.IntConstant(extent);
      } else {
        // b2-(int(b1)-1)
        Xobject tmp = Xcons.IntConstant(b1.getInt() - 1);
        result = Xcons.binaryOp(Xcode.MINUS_EXPR, b2, tmp);
      }
    } else {
      if (b2.isIntConstant()) {
        // (int(b2)+1)-b1
        Xobject tmp = Xcons.IntConstant(b2.getInt() + 1);
        result = Xcons.binaryOp(Xcode.MINUS_EXPR, tmp, b1);
      } else {
        // (b2-b1)+1
        Xobject tmp = Xcons.binaryOp(Xcode.MINUS_EXPR, b2, b1);
        result = Xcons.binaryOp(Xcode.PLUS_EXPR, tmp, Xcons.IntConstant(1));
      }
    }
    return result.cfold(fblock);
  }


  /*  returns (a1*a2).cfold()
   */
  Xobject _calcTimesExprExpr(Xobject a1, Xobject a2) {
    Xobject b1, b2;

    if (a1 == null || a2 == null)
      return null;
    b1 = a1.cfold(fblock);
    b2 = a2.cfold(fblock);

    Xobject result;
    int n;
    if (b1.isIntConstant()) {
      if (b2.isIntConstant()) {
        // int(b1)*int(b2)
        n = b1.getInt() * b2.getInt();
        return Xcons.IntConstant(n);
      } else {
        // int(b1)*b2
        if ((n = b1.getInt()) == 1)
          return b2;
        else if (n == 0)
          return b1;
      }
    } else {
      if (b2.isIntConstant()) {
        // b1*int(b2)
        if ((n = b2.getInt()) == 1)
          return b1;
        else if (n == 0)
          return b2;
      }
    }

    result = Xcons.binaryOp(Xcode.MUL_EXPR, b1, b2);
    return result.cfold(fblock);
  }


  /*  returns (a1+a2).cfold()
   */
  Xobject _calcPlusExprExpr(Xobject a1, Xobject a2) {
    Xobject b1, b2;

    if (a1 == null || a2 == null)
      return null;
    b1 = a1.cfold(fblock);
    b2 = a2.cfold(fblock);

    Xobject result;
    int n;
    if (b1.isIntConstant()) {
      if (b2.isIntConstant()) {
        // int(b1)+int(b2)
        n = b1.getInt() + b2.getInt();
        return Xcons.IntConstant(n);
      } else {
        // int(b1)+b2
        if ((n = b1.getInt()) == 0)
          return b2;
      }
    } else {
      if (b2.isIntConstant()) {
        // b1+int(b2)
        if ((n = b2.getInt()) == 0)
          return b1;
      }
    }

    result = Xcons.binaryOp(Xcode.PLUS_EXPR, b1, b2);
    return result.cfold(fblock);
  }


  /*  returns (a1-a2).cfold()
   */
  Xobject _calcMinusExprExpr(Xobject a1, Xobject a2) {
    Xobject b1, b2;

    if (a1 == null || a2 == null)
      return null;
    b1 = a1.cfold(fblock);
    b2 = a2.cfold(fblock);

    int n;
    if (b1.isIntConstant()) {
      if (b2.isIntConstant()) {
        // int(b1)-int(b2)
        n = b1.getInt() - b2.getInt();
        return Xcons.IntConstant(n);
      }
    } else {
      if (b2.isIntConstant()) {
        // b1-int(b2)
        if (b2.getInt() == 0)
          return b1;
      }
    }

    Xobject result = Xcons.binaryOp(Xcode.MINUS_EXPR, b1, b2);
    return result.cfold(fblock);
  }


  /*  returns (a1+1).cfold()
   */
  Xobject _calcIncExpr(Xobject a1) {
    Xobject b1;

    if (a1 == null)
      return null;
    b1 = a1.cfold(fblock);

    int n;
    if (b1.isIntConstant()) {
      // int(b1)+1
      n = b1.getInt() + 1;
      return Xcons.IntConstant(n);
    }

    Xobject result = Xcons.binaryOp(Xcode.PLUS_EXPR, b1, Xcons.IntConstant(1));
    return result.cfold(fblock);
  }


  //------------------------------
  //  tool
  //------------------------------
  private Ident declIntIntrinsicIdent(String name) { 
    FunctionType ftype = new FunctionType(Xtype.FintType, Xtype.TQ_FINTRINSIC);
    Ident ident = getEnv().declIntrinsicIdent(name, ftype);
    return ident;
  }

  private Ident declIntExtendIntrinsicIdent(String name) { 
    FunctionType ftype = new FunctionType(Xtype.FintType, Xtype.TQ_FINTRINSIC);
    Ident ident = getEnv().declIntrinsicIdent(name, ftype);
    return ident;
  }


  //------------------------------
  //  get/set Xtype object
  //------------------------------
  public Boolean isScalar() {
    return (ident.Type().getNumDimensions() == 0);
  }

  public Boolean isAllocatable() {
    return isAllocatable;
  }

  public void setAllocatable() {
    ident.Type().setIsFallocatable(true);
  }

  public void resetAllocatable() {
    /**
     * Since TypeQualFlag TQ_FALLOCATABLE may duplicately be set
     * in a type and its children, e.g., in type (of F_ARRAY) and
     * in type.getRef() (of STRUCT).  In order to reset such a 
     * flag, recursive operation is necessary.
     */
    for (Xtype type = ident.Type(); type != null; ) {
      type.setIsFallocatable(false);
      if (type.copied != null) {
        type = type.copied;
        continue;
      }
      if (type.isBasic())
        break;
      else if (type.isStruct())
        break;
      type = type.getRef();
    }
  }

  public void resetSaveAttr() {
    //    Xtype type = ident.Type();
    //    _resetSaveAttrInType(type);
    _resetSaveAttrInType(ident.Type());

    // How various!
    if (ident.getStorageClass() == StorageClass.FSAVE)
      ident.setStorageClass(StorageClass.FLOCAL);
  }

  public void setSaveAttr() {
    _setSaveAttrInType(ident.Type());
    ident.setStorageClass(StorageClass.FSAVE);
  }

  public void setSaveAttrToDescPointer() {
    _setSaveAttrInType(getDescPointerId().Type());
    getDescPointerId().setStorageClass(StorageClass.FSAVE);
  }

  public void setZeroToDescPointer() {
    Xobject zero = Xcons.IntConstant(0, Xtype.intType, "8");
    getDescPointerId().setFparamValue(Xcons.List(zero, null));
  }

  private void _setSaveAttrInType(Xtype type) {
    type.setIsFsave(true);
  }

  private void _resetSaveAttrInType(Xtype type) {
    type.setIsFsave(false);

    if (type.copied != null) 
      _resetSaveAttrInType(type.copied);

    if (type.isArray() || type.isFarray())
      _resetSaveAttrInType(type.getRef());
  }

  public Boolean isPointer() {
    return isPointer;
  }

  public void setPointer() {
    ident.Type().setIsFpointer(true);
  }

  public void resetPointer() {
    for (Xtype type = ident.Type(); type != null; ) {
      type.setIsFpointer(false);
      if (type.copied != null)
        type = type.copied;
      else if (type.isBasic())
        break;
      else
        type = type.getRef();
    }
  }

  public Boolean isDummyArg() {
    if (ident.getStorageClass() == StorageClass.FPARAM)
      return true;
    return false;
  }

  public Boolean isAssumedSize() {
    return ident.Type().isFassumedSize();
  }

  public Boolean isAssumedShape() {
    return ident.Type().isFassumedShape();
  }

  public Boolean isExplicitShape() {
    return (!isAssumedSize() && !isAssumedShape() &&
            !isAllocatable() && !isPointer());
  }

  public Boolean isUseAssociated() {
    return isUseAssociated;
  }



  public Ident getIdent() {
    return ident;
  }

  public void setIdentEtc(Ident ident) {
    this.ident = ident;
    name = ident.getName();

    isAllocatable = ident.Type().isFallocatable();
    isPointer = ident.Type().isFpointer();
    isUseAssociated = (ident.getFdeclaredModule() != null);
    nodesName = getProp_nodes(ident);
    nodesDescId = _getNodesDescIdByName(nodesName);
  }

  private Ident _getNodesDescIdByName(String nodesName) {
    return _getNodesDescIdByName(nodesName, env, fblock);
  }
  private Ident _getNodesDescIdByName(String nodesName,
                                      XMPenv env, FunctionBlock fblock) {
    if (nodesName != null) {
      XMPnodes nodes = env.findXMPnodes(nodesName, (ident.getDeclaredBlock() != null) ?
                                                    ident.getDeclaredBlock().getParent() : fblock);
      return nodes.getDescId();
    }
    return null;
  }

  public XobjectDef getDef() {
    return def;
  }

  public FunctionBlock getFblock() {
    return fblock;
  }

  public XMPenv getEnv() {
    return env;
  }

  public String getHomeBlockCodeName()
  {
    return homeBlockCodeName;
  }

  public String getDescCommonName()
  {
    String descCommonName = 
      VAR_DESCPOINTER_PREFIX + "_" +
      homeBlockCodeName + "_" +
      _getCodeFromName(getName());

    return descCommonName;
  }

  // for case useMalloc
  public String getCrayCommonName()
  {
    String crayCommonName = 
      VAR_CRAYPOINTER_PREFIX + "_" +
      homeBlockCodeName + "_" +
      _getCodeFromName(getName());

    return crayCommonName;
  }

  // for case !useMalloc
  public String getCoarrayCommonName()
  {
    String coarrayCommonName =
      CBLK_COARRAYS_PREFIX + "_" +
      homeBlockCodeName + "_" +
      _getCodeFromName(getName());

    return coarrayCommonName;
  }

  public String getCrayPointerName() {
    if (_crayPtrName == null) {
      _crayPtrName = VAR_CRAYPOINTER_PREFIX + "_" + name;
    }
    return _crayPtrName;
  }

  public Ident getCrayPointerId() {
    if (descPtrId == null) {
      XMP.warning("INTERNAL: illegal null crayPtrId (XMPcoppy.getCrayPointerId)");
      return null;
    }

    return crayPtrId;
  }

  public String getDescPointerName() {
    if (_descPtrName == null) {
      _descPtrName = VAR_DESCPOINTER_PREFIX + "_" + name;
    }

    return _descPtrName;
  }

  public Ident getDescPointerId() {
    Ident ident = env.findVarIdent(_descPtrName, fblock);
    return ident;
  }


  public String getCoarrayCommName() {
    if (_coarrayCommName == null) {
      _coarrayCommName = VAR_COARRAYCOMM_PREFIX + "_" + name;
    }

    return _coarrayCommName;
  }

  public Ident getCoarrayCommId() {
    if (coarrayCommId == null) {
      XMP.warning("INTERNAL: illegal null coarrayCommId (XMPcoppy.getCoarrayCommId)");
      return null;
    }

    return coarrayCommId;
  }


  /*************** should be deleted .....
  ***************************/
  /** No no, this may be used again in Ver.6
  ***/
  public Xobject getDescPointerIdExpr(Xobject baseAddr) {
    if (descPtrId != null)
      return descPtrId;

    Ident funcIdent =
      getEnv().declExternIdent(GET_DESCR_ID_NAME, Xtype.FintFunctionType);
    Xobject descId = funcIdent.Call(Xcons.List(baseAddr));
    return descId;
  }

  public Xobject[] getCodimensions() {
    Xobject[] codims = ident.Type().getCodimensions();
    return codims;
  }

  public void setCodimensions(Xobject[] codimensions) {
    ident.Type().setCodimensions(codimensions);
  }

  public void removeCodimensions() {
    ident.Type().removeCodimensions();
  }

  public void hideCodimensions() {
    ident.Type().setIsCoarray(false);
  }

  public String getName() {
    return ident.getName();
  }

  public Xtype getType() {
    return ident.Type();
  }

  public void setWasMovedFromModule(Boolean bool) {
    _wasMovedFromModule = bool;
  }

  public Boolean wasMovedFromModule() {
    return _wasMovedFromModule;
  }

  public static XMPcoarray findCoarrayInCoarrays(String name,
                                                 ArrayList<XMPcoarray> coarrays) {
    for (XMPcoarray coarray: coarrays) {
      if (coarray.getName() == name) {
        return coarray;
      }
    }

    return null;
  }


  public String toString() {
    String s = 
      "\n  Ident ident = " +  ident +
      "\n  String name = " +  name +
      "\n  FindexRange indexRange = " +  indexRange +
      "\n  FindexRange coindexRange = " +  coindexRange +
      "\n  Boolean isAllocatable = " +  isAllocatable +
      "\n  Boolean isPointer = " +  isPointer +
      "\n  Boolean isUseAssociated = " +  isUseAssociated +
      "\n  Boolean useMalloc = " + useMalloc +
      "\n  Boolean _wasMovedFromModule = " + _wasMovedFromModule +
      "\n  String _crayPtrName = " +  _crayPtrName +
      "\n  Ident crayPtrId = " +  crayPtrId +
      "\n  String _descPtrName = " +  _descPtrName +
      "\n  Ident descPtrId = " +  descPtrId +
      "\n  String homeBlockCodeName = " +  homeBlockCodeName +
      "\n  XMPenv env = " +  env +
      "\n  XobjectDef def = " +  def + ": name=" + def.getName() +
      "\n  FunctionBlock fblock" + ": name=" + def.getName();
    return s;
  }



  //------------------------------
  //  low-level handling (NOT USED)
  //------------------------------
  public Ident unlinkIdent() {
    return unlinkIdent(def);
  }
  public Ident unlinkIdent(XobjectDef def) {
    return unlinkIdent((XobjList)def.getDef());
  }
  public Ident unlinkIdent(XobjList def) {
    XobjArgs args0 = def.getIdentList().getArgs();
    XobjArgs lastArgs = null;
    XobjArgs thisArgs = null;
    for (XobjArgs args = args0; args != null; args = args.nextArgs()) {
      Xobject arg = args.getArg();
      Ident id = (Ident)arg;
      if (id == ident) {
        thisArgs = args;
        break;
      }
      if (id.getName().equals(name)) {
        XMP.fatal("unexpected matching of ident names: " + name);
        thisArgs = args;
        break;
      }
      lastArgs = args;
    }

    if (thisArgs == null)   // not found
      return null;

    // unlink and reconnect
    if (lastArgs == null)
      def.getIdentList().setArgs(thisArgs.nextArgs());
    else
      lastArgs.setNext(thisArgs.nextArgs());

    thisArgs.setNext(null);

    return (Ident)thisArgs.getArg();
  }


  //------------------------------------------------------------
  //  Fortran Type and Kind
  //   ******** under construction *********
  //------------------------------------------------------------

  private Xtype _getXtype() {
    Xtype xtype = ident.Type();
    if (xtype.getKind() == Xtype.F_ARRAY)
      xtype = xtype.getRef();
    return xtype;
  }

  public int getFtypeNumber() {
    return _getXtype().getBasicType();
  }

  public Xobject getFkind() {
    return _getXtype().getFkind();
  }

  /*
   * return a name of Fortran intrinsic function
   */
  public String getFtypeString() {
    return getFtypeString(_getXtype());
  }
  public String getFtypeString(Xtype xtype) {
    Ftype ftype = new Ftype(xtype, fblock);
    return ftype.getNameOfConvFunction();
  }


  private Xtype _getCharFarrayType(int size) {
    Xtype ref = Xtype.FcharacterType;
    Xtype type = Xtype.Farray(ref, Xcons.IntConstant(size));
    return type;
  }

}

