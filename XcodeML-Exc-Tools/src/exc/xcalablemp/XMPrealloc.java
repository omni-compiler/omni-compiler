package exc.xcalablemp;

import exc.object.*;
import java.util.Iterator; 

public class XMPrealloc implements XobjectDefVisitor {
  private XMPglobalDecl _globalDecl;
  private XMPobjectTable _globalObjectTable;

  public XMPrealloc(XMPglobalDecl globalDecl) {
    _globalDecl = globalDecl;
    _globalObjectTable = globalDecl.getGlobalObjectTable();
  }

  public void doDef(XobjectDef def) {
    realloc(def);
  }

  public void realloc(XobjectDef def) {
    if (def.isVarDecl()) {
      String varName = def.getName();
      XMPalignedArray alignedArray = _globalObjectTable.getAlignedArray(varName);
      if(alignedArray != null) {
        if (!(alignedArray.realloc())) return;

        def.setDef(null);
        if(alignedArray.getAddrId().getStorageClass() != StorageClass.EXTERN) {
          XobjList allocFuncArgs = Xcons.List(alignedArray.getAddrId().getAddr(),
                                              alignedArray.getDescId().Ref(),
                                              Xcons.SizeOf(alignedArray.getType()));
          for (int i = alignedArray.getDim() - 1; i >= 0; i--)
            allocFuncArgs.add(Xcons.Cast(Xtype.unsignedlonglongType,
                                         alignedArray.getGtolAccIdAt(i).getAddr()));

          _globalDecl.addGlobalInitFuncCall("_XCALABLEMP_alloc_array", allocFuncArgs);
        }
      }
    }
  }
}
