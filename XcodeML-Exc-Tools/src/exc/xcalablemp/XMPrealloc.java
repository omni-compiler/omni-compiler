package exc.xcalablemp;

import exc.object.*;
import java.util.Iterator; 

public class XMPrealloc implements XobjectDefVisitor {
  private XMPglobalDecl _globalDecl;

  public XMPrealloc(XMPglobalDecl globalDecl) {
    _globalDecl = globalDecl;
  }

  public void doDef(XobjectDef def) {
    try {
      alignArrayRealloc(def);
      originalCoarrayDelete(def);
    } catch (XMPexception e) {
      XMP.error(def.getLineNo(), e.getMessage());
    }
  }

  private void originalCoarrayDelete(XobjectDef def) throws XMPexception {
    if (def.isVarDecl()) {
      String varName = def.getName();
      XMPcoarray coarray = _globalDecl.getXMPcoarray(varName);

      if (coarray != null){
	def.setDef(Xcons.List(Xcode.TEXT,
			      Xcons.String("/* array '" + varName + "' is removed by XcalableMP coarray directive */")));
      }
    }
  }

  private void insertReallocFunction(XobjectDef def, String varName, Ident structId) throws XMPexception {
    XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(varName);
    if(alignedArray == null)     return;
    if(alignedArray.isPointer()) return;

    Boolean isStructure = (structId != null);
    if(alignedArray.realloc()){
      XobjList allocFuncArgs = null;
      if(isStructure){
	Xobject x = Xcons.memberAddr(structId.getAddr(), alignedArray.getAddrId().getName());
	allocFuncArgs = Xcons.List(Xcons.Cast(Xtype.Pointer(Xtype.voidPtrType), x), alignedArray.getDescId().Ref());
      }
      else
	allocFuncArgs = Xcons.List(alignedArray.getAddrIdVoidAddr(), alignedArray.getDescId().Ref());
      
      if(alignedArray.getAddrId().getStorageClass() != StorageClass.EXTERN)
        allocFuncArgs.add(Xcons.IntConstant(1));
      
      for(int i = alignedArray.getDim() - 1; i >= 0; i--)
        allocFuncArgs.add(Xcons.Cast(Xtype.Pointer(Xtype.unsignedlonglongType),
                                     alignedArray.getAccIdAt(i).getAddr()));
      
      if(alignedArray.getAddrId().getStorageClass() == StorageClass.EXTERN)
        _globalDecl.addGlobalInitFuncCall("_XMP_alloc_array_EXTERN", allocFuncArgs);
      else{
        _globalDecl.addGlobalInitFuncCall("_XMP_alloc_array",                allocFuncArgs);
        _globalDecl.insertGlobalFinalizeFuncCall("_XMP_finalize_array_desc", Xcons.List(alignedArray.getDescId().Ref()));
        _globalDecl.insertGlobalFinalizeFuncCall("_XMP_dealloc_array",       Xcons.List(alignedArray.getDescId().Ref()));
      }

      if(! isStructure)
	def.setDef(Xcons.List(Xcode.TEXT, Xcons.String("/* array '" + varName + "' is removed by XMP align directive */")));
    }
    else{
      Xobject addrIdVoidAddr = alignedArray.getAddrIdVoidAddr();
      Xobject arrayIdRef     = alignedArray.getArrayId().Ref();
      Xobject arrayDescRef   = alignedArray.getDescId().Ref();
      XobjList allocFuncArgs = Xcons.List(addrIdVoidAddr, arrayIdRef, arrayDescRef);
      
      for(int i=alignedArray.getDim()-1; i>=0; i--)
        allocFuncArgs.add(Xcons.Cast(Xtype.Pointer(Xtype.unsignedlonglongType),
                                     alignedArray.getAccIdAt(i).getAddr()));
      
      _globalDecl.addGlobalInitFuncCall("_XMP_init_array_addr", allocFuncArgs);
    }
   }
  
  private void alignArrayRealloc(XobjectDef def) throws XMPexception {
    if(def.isVarDecl() == false) return;

    String varName   = def.getName();
    Ident varId      = _globalDecl.findIdent(varName);
    Boolean isStructure = (varId.Type().getKind() == Xtype.STRUCT);
    if(isStructure){
      String structName   = varName;
      XobjList memberList = _globalDecl.findIdent(varName).Type().getMemberList();
      for(Xobject x : memberList){
	Ident arrayId = (Ident)x;
	if(arrayId.isMemberAligned()){
	  String orgName = x.getName().replaceAll("^"+XMP.ADDR_PREFIX_, "");
	  String arrayName = XMP.STRUCT + varName + "_" + orgName;
	  insertReallocFunction(def, arrayName, varId);
	}
      }
    }
    else
      insertReallocFunction(def, varName, null);
  }
}
