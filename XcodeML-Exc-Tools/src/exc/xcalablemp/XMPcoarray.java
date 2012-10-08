/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;
import exc.block.*;
import java.util.Vector;

public class XMPcoarray {
  public final static int GET = 700;
  public final static int PUT = 701;

  // FIXME supported in coarray fortran???
  public final static int ACC_BIT_XOR	= 1;

  private String		_name;
  private Xtype			_elmtType;
  private int			_varDim;
  private Vector<Long>		_sizeVector;
  private Xobject		_varAddr;
  private Ident			_varId;
  private Ident			_descId;

  public XMPcoarray(String name, Xtype elmtType, int varDim, Vector<Long> sizeVector,
                    Xobject varAddr, Ident varId, Ident descId) {
    _name = name;
    _elmtType = elmtType;
    _varDim = varDim;
    _sizeVector = sizeVector;
    _varAddr = varAddr;
    _varId = varId;
    _descId = descId;
  }

  public String getName() {
    return _name;
  }

  public Xtype getElmtType() {
    return _elmtType;
  }

  public int getVarDim() {
    return _varDim;
  }

  public int getSizeAt(int index) {
    return _sizeVector.get(index).intValue();
  }

  public Xobject getVarAddr() {
    return _varAddr;
  }

  public Ident getVarId() {
    return _varId;
  }

  public Ident getDescId() {
    return _descId;
  }

  public static void translateCoarray(XobjList coarrayDecl, XMPglobalDecl globalDecl,
                                      boolean isLocalPragma, XMPsymbolTable localXMPsymbolTable) throws XMPexception {

    String coarrayName = coarrayDecl.getArg(0).getString();

    // Fix me: Now only one dimensional node of coarray is supported
    int num_coarray_node_dim = XMPutil.countElmts((XobjList)coarrayDecl.getArg(1));
    if(num_coarray_node_dim > 1){
      throw new XMPexception("number of coarray " + coarrayName + " node dimension is " + num_coarray_node_dim + ".\n" +
			     "Now support only one coarray node dimension.");
    }

    if(globalDecl.getXMPcoarray(coarrayName) != null) {
      throw new XMPexception("coarray " + coarrayName + " is already declared");
    }

    if (globalDecl.getXMPalignedArray(coarrayName) != null) {
      throw new XMPexception("an aligned array cannot be declared as a coarray");
    }

    Ident varId = globalDecl.findVarIdent(coarrayName);
    if (varId == null) {
      throw new XMPexception("coarray '" + coarrayName + "' is not declared");
    }

    boolean isArray = false;
    int varDim = 0;
    Xtype elmtType = null;
    Xtype varType = varId.Type();
    Xobject varAddr = null;
    long num_of_elemt = 1;
    if (varType.getKind() == Xtype.ARRAY) {
      isArray = true;
      varDim = varType.getNumDimensions();
      elmtType = varType.getArrayElementType();
      varAddr = varId.Ref();
      num_of_elemt = XMPutil.getArrayElmtCount(varType);
    } else {
      varDim = 1;
      elmtType = varType;
      varAddr = varId.getAddr();
    }

    Xobject elmtTypeRef = null;
    if (elmtType.getKind() == Xtype.BASIC) {
      elmtTypeRef = XMP.createBasicTypeConstantObj(elmtType);
    } else {
      elmtTypeRef = Xcons.IntConstant(XMP.NONBASIC_TYPE);
    }

    XobjList coarrayDimSizeList = (XobjList)coarrayDecl.getArg(1);
    int coarrayDim = coarrayDimSizeList.Nargs();
    if (coarrayDim > XMP.MAX_DIM) {
      throw new XMPexception("coarray dimension should be less than " + (XMP.MAX_DIM + 1));
    }

    String initDescFuncName = null;
    initDescFuncName = new String("_XMP_coarray_malloc"); 
    //    if (coarrayDecl.getArg(1).getArg(coarrayDim - 1) == null) {
    //      initDescFuncName = new String("_XMP_init_coarray_DYNAMIC");
    //    } else {
    //      initDescFuncName = new String("_XMP_init_coarray_STATIC");
    //    }
    
    // init descriptor
    Ident descId = globalDecl.declStaticIdent(XMP.COARRAY_DESC_PREFIX_ + coarrayName, Xtype.voidPtrType);
    Ident addrId = globalDecl.declStaticIdent(XMP.COARRAY_ADDR_PREFIX_ + coarrayName, new PointerType(elmtType));
    //    XobjList initDescFuncArgs = Xcons.List(descId.getAddr(), varAddr, elmtTypeRef, Xcons.SizeOf(elmtType),
    //                                           Xcons.IntConstant(coarrayDim));
    
    // Fix me : How to create unsigned long long constant ?
    XobjList initDescFuncArgs = Xcons.List(descId.getAddr(), addrId.getAddr(), Xcons.LongLongConstant(0, num_of_elemt),
					   Xcons.SizeOf(elmtType));
    
    //    for (Xobject coarrayDimSize : coarrayDimSizeList) {
    //      if (coarrayDimSize != null) {
    //        initDescFuncArgs.add(Xcons.Cast(Xtype.intType, coarrayDimSize));
    //      }
    //    }

    // call init desc function
    globalDecl.addGlobalInitFuncCall(initDescFuncName, initDescFuncArgs);

    // call init comm function
    //    XobjList initCommFuncArgs = Xcons.List(descId.Ref(), Xcons.IntConstant(varDim));
    Vector<Long> sizeVector = new Vector<Long>(varDim);
    
    if (isArray) {
      for (int i = 0; i < varDim; i++, varType = varType.getRef()) {
	long dimSize = varType.getArraySize();
	if ((dimSize == 0) || (dimSize == -1)) {
	  throw new XMPexception("array size should be declared statically");
	}
	sizeVector.add(new Long(dimSize));
	//        initCommFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.LongLongConstant(0, dimSize)));
      }
    } else {
      sizeVector.add(new Long(1));
      //      initCommFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
    }
    
    //    globalDecl.addGlobalInitFuncCall("_XMP_init_coarray_comm", initCommFuncArgs);
    
    // call finalize desc function
    //		globalDecl.addGlobalFinalizeFuncCall("_XMP_finalize_coarray", Xcons.List(descId.Ref()));
    
    XMPcoarray coarrayEntry = new XMPcoarray(coarrayName, elmtType, varDim, sizeVector, varAddr, varId, descId);
    globalDecl.putXMPcoarray(coarrayEntry);
  }
}
