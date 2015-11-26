package exc.xcalablemp;

import exc.object.*;
import exc.block.*;
import xcodeml.IXobject;
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
  private Vector<Long>          _sizeVector;
  private int                   _imageDim;
  private Vector<Integer>       _imageVector;
  private Xobject		_varAddr;
  private Ident			_varId;
  private Ident			_descId;

  public XMPcoarray(String name, Xtype elmtType, int varDim, Vector<Long> sizeVector, int imageDim, Vector<Integer> imageVector,
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

  public long getSizeAt(int index) {
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


  /*
   *  for XMP/C V1.0 format
   */
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

    XobjList codimensions = (XobjList)coarrayDecl.getArg(1);
    translateCoarray_core(varId, coarrayName, codimensions,
                          globalDecl, isLocalPragma);
  }


  public static void translateCoarray_core(Ident varId, String name,
                                           XobjList codimensions,
                                           XMPglobalDecl globalDecl,
                                           boolean isLocal) throws XMPexception {
    boolean is_output = true;
    if(varId.getStorageClass() == StorageClass.EXTERN)
      is_output = false;

    int imageDim = XMPutil.countElmts(codimensions);
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

    XobjList coarrayDimSizeList = codimensions;
    int coarrayDim = coarrayDimSizeList.Nargs();
    if(coarrayDim > XMP.MAX_DIM) {
      throw new XMPexception("coarray dimension should be less than " + (XMP.MAX_DIM + 1));
    }

    // _XMP_coarray_malloc_info_X()
    String funcName = new String("_XMP_coarray_malloc_info_");
    funcName += Integer.toString(varDim);
    XobjList funcArgs = Xcons.List();
    XobjList lockFuncArgs = Xcons.List();  // This variable may be used for _xmp_lock_initialize()
    Vector<Long> sizeVector = new Vector<Long>(varDim);

    if(!isArray){
      sizeVector.add(new Long(1));
      Xobject arg = Xcons.Cast(Xtype.unsignedType, Xcons.LongLongConstant(0, 1));
      funcArgs.add(arg);
      lockFuncArgs.add(arg);
    }
    else{
      for(int i=0;i<varDim;i++,varType=varType.getRef()){
        long dimSize = (long)varType.getArraySize();
        if((dimSize == 0) || (dimSize == -1)) {
          throw new XMPexception("array size should be declared statically");
        }
        sizeVector.add(new Long(dimSize));
        Xobject arg = Xcons.Cast(Xtype.unsignedType, Xcons.LongLongConstant(0, dimSize));
	funcArgs.add(arg);
        lockFuncArgs.add(arg);
      }
    }
    
    funcArgs.add(Xcons.SizeOf(elmtType));
    if(is_output){
      globalDecl.addGlobalInitFuncCall(funcName, funcArgs);
    }

    // _XMP_coarray_malloc_image_info_X()
    if(is_output){
      funcName = new String("_XMP_coarray_malloc_image_info_");
      funcName = funcName + Integer.toString(imageDim);
      funcArgs = Xcons.List();
      for(int i=0;i<imageDim-1;i++){
        funcArgs.add(codimensions.getArg(i));
      }
      globalDecl.addGlobalInitFuncCall(funcName, funcArgs);
    }

    Vector<Integer> imageVector = new Vector<Integer>(imageDim);
    for(int i=0;i<imageDim-1;i++){
      imageVector.add(codimensions.getArg(i).getInt());
    }
    imageVector.add(XMPcoarray.ASTERISK);

    // _XMP_coarray_malloc_do()
    funcName = new String("_XMP_coarray_malloc_do");
    Ident descId, addrId;
    if(varId.getStorageClass() == StorageClass.EXTERN){
      descId = globalDecl.declExternIdent(XMP.COARRAY_DESC_PREFIX_ + name, Xtype.voidPtrType);
      addrId = globalDecl.declExternIdent(XMP.COARRAY_ADDR_PREFIX_ + name, new PointerType(elmtType));
    }
    else if(varId.getStorageClass() == StorageClass.STATIC){
      descId = globalDecl.declStaticIdent(XMP.COARRAY_DESC_PREFIX_ + name, Xtype.voidPtrType);
      addrId = globalDecl.declStaticIdent(XMP.COARRAY_ADDR_PREFIX_ + name, new PointerType(elmtType));
    }
    else if(varId.getStorageClass() == StorageClass.EXTDEF){
      descId = globalDecl.declGlobalIdent(XMP.COARRAY_DESC_PREFIX_ + name, Xtype.voidPtrType);
      addrId = globalDecl.declGlobalIdent(XMP.COARRAY_ADDR_PREFIX_ + name, new PointerType(elmtType));
    }
    else {
      throw new XMPexception("cannot coarray '" + name + "', wrong storage class");
    }

    funcArgs = Xcons.List(descId.getAddr(), addrId.getAddr());

    if(is_output){
      globalDecl.addGlobalInitFuncCall(funcName, funcArgs);
    }
    
    // _xmp_lock_initialize()
    XobjectFile _env = globalDecl.getEnv();
    Ident id = _env.findIdent("xmp_lock_t", IXobject.FINDKIND_TAGNAME);
    if(id != null){
      if(id.Type() == elmtType){
        funcName = new String("_XMP_lock_initialize_");
        funcName += Integer.toString(varDim);
        lockFuncArgs.insert(addrId);
        
        if(is_output){
          globalDecl.addGlobalInitFuncCall(funcName, lockFuncArgs);
        }
      }
    }
    XMPcoarray coarrayEntry = new XMPcoarray(name, elmtType, varDim, sizeVector, 
                                             imageDim, imageVector, varAddr, varId, descId);

    globalDecl.putXMPcoarray(coarrayEntry);
  }
}
