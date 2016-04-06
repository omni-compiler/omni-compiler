/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

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

  private void alignArrayRealloc(XobjectDef def) throws XMPexception {
    if (def.isVarDecl()) {
      String varName = def.getName();
      XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(varName);
      if (alignedArray != null && !alignedArray.isPointer()) {
        if (alignedArray.realloc()) {
          XobjList allocFuncArgs = Xcons.List(alignedArray.getAddrIdVoidAddr(), alignedArray.getDescId().Ref());

	  if (alignedArray.getAddrId().getStorageClass() != StorageClass.EXTERN)
	    allocFuncArgs.add(Xcons.IntConstant(1));

          for (int i = alignedArray.getDim() - 1; i >= 0; i--) {
            allocFuncArgs.add(Xcons.Cast(Xtype.Pointer(Xtype.unsignedlonglongType),
                                         alignedArray.getAccIdAt(i).getAddr()));
          }

          if (alignedArray.getAddrId().getStorageClass() == StorageClass.EXTERN) {
            _globalDecl.addGlobalInitFuncCall("_XMP_alloc_array_EXTERN", allocFuncArgs);
          }
          else {
            _globalDecl.addGlobalInitFuncCall("_XMP_alloc_array", allocFuncArgs);
          }

	  def.setDef(Xcons.List(Xcode.TEXT,
				Xcons.String("/* array '" + varName + "' is removed by XcalableMP align directive */")));
        } else {
          XobjList allocFuncArgs = Xcons.List(alignedArray.getAddrIdVoidAddr(),
                                              alignedArray.getArrayId().Ref(),
                                              alignedArray.getDescId().Ref());
          for (int i = alignedArray.getDim() - 1; i >= 0; i--) {
            allocFuncArgs.add(Xcons.Cast(Xtype.Pointer(Xtype.unsignedlonglongType),
                                         alignedArray.getAccIdAt(i).getAddr()));
          }

          _globalDecl.addGlobalInitFuncCall("_XMP_init_array_addr", allocFuncArgs);
        }
      }
    }
  }
}
