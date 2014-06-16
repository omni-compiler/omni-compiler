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
  public final static int ASTERISK = -1;

  // FIXME supported in coarray fortran???
  public final static int ACC_BIT_XOR	= 1;

  private String		_name;
  private Xtype			_elmtType;
  private int			_varDim;
  private Vector<Integer>	_sizeVector;
  private int                   _imageDim;
  private Vector<Integer>       _imageVector;
  private Xobject		_varAddr;
  private Ident			_varId;
  private Ident			_descId;

  public XMPcoarray(String name, Xtype elmtType, int varDim, Vector<Integer> sizeVector, int imageDim, Vector<Integer> imageVector,
                    Xobject varAddr, Ident varId, Ident descId) {
    _name        = name;
    _elmtType    = elmtType;
    _varDim      = varDim;
    _sizeVector  = sizeVector;
    _imageDim    = imageDim;
    _imageVector = imageVector;
    _varAddr     = varAddr;
    _varId       = varId;
    _descId      = descId;
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

  public int getImageDim() {
    return _imageDim;
  }

  public int getImageAt(int index) {
    return _imageVector.get(index);
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

    int imageDim = XMPutil.countElmts((XobjList)coarrayDecl.getArg(1));
    boolean isArray = false;
    int varDim = 0;
    Xtype elmtType = null;
    Xtype varType = varId.Type();
    Xobject varAddr = null;

    if (varType.getKind() == Xtype.ARRAY) {
      isArray = true;
      varDim = varType.getNumDimensions();
      elmtType = varType.getArrayElementType();
      varAddr = varId.Ref();
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
    if(coarrayDim > XMP.MAX_DIM) {
      throw new XMPexception("coarray dimension should be less than " + (XMP.MAX_DIM + 1));
    }

    // _XMP_coarray_malloc_set()
    String funcName = new String("_XMP_coarray_malloc_set");
    XobjList funcArgs = Xcons.List(Xcons.SizeOf(elmtType), Xcons.IntConstant(varDim), Xcons.IntConstant(imageDim));
    globalDecl.addGlobalInitFuncCall(funcName, funcArgs);

    // _XMP_coarray_malloc_array_info()
    funcName = new String("_XMP_coarray_malloc_array_info");
    Vector<Integer> sizeVector = new Vector<Integer>(varDim);
    if(isArray){
      for(int i=0;i<varDim;i++,varType=varType.getRef()){
        int dimSize = (int)varType.getArraySize();
        if((dimSize == 0) || (dimSize == -1)) {
          throw new XMPexception("array size should be declared statically");
        }
	funcArgs = Xcons.List(Xcons.IntConstant(i), Xcons.IntConstant(dimSize));
	globalDecl.addGlobalInitFuncCall(funcName, funcArgs);
        sizeVector.add(new Integer(dimSize));
      }
    }
    else{
      funcArgs = Xcons.List(Xcons.IntConstant(0), Xcons.IntConstant(1));
      globalDecl.addGlobalInitFuncCall(funcName, funcArgs);
      sizeVector.add(new Integer(1));
    }  

    // _XMP_coarray_malloc_image_info()
    funcName = new String("_XMP_coarray_malloc_image_info");
    Vector<Integer> imageVector = new Vector<Integer>(imageDim);
    for(int i=0;i<imageDim-1;i++){
      funcArgs = Xcons.List(Xcons.IntConstant(i), ((XobjList)coarrayDecl.getArg(1)).getArg(i));
      globalDecl.addGlobalInitFuncCall(funcName, funcArgs);
      imageVector.add(((XobjList)coarrayDecl.getArg(1)).getArg(i).getInt());
    }
    imageVector.add(XMPcoarray.ASTERISK);

    // _XMP_coarray_malloc_do()
    funcName = new String("_XMP_coarray_malloc_do");
    Ident descId = globalDecl.declStaticIdent(XMP.COARRAY_DESC_PREFIX_ + coarrayName, Xtype.voidPtrType);
    Ident addrId = globalDecl.declStaticIdent(XMP.COARRAY_ADDR_PREFIX_ + coarrayName, new PointerType(elmtType));
    funcArgs = Xcons.List(descId.getAddr(), addrId.getAddr());
    globalDecl.addGlobalInitFuncCall(funcName, funcArgs);
    
    XMPcoarray coarrayEntry = new XMPcoarray(coarrayName, elmtType, varDim, sizeVector, 
                                             imageDim, imageVector, varAddr, varId, descId);

    globalDecl.putXMPcoarray(coarrayEntry);
  }
}
