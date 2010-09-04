package exc.xcalablemp;

import exc.block.*;
import exc.object.*;

public class XMPrewriteExpr {
  private XMPglobalDecl		_globalDecl;
  private XMPobjectTable	_globalObjectTable;

  public XMPrewriteExpr(XMPglobalDecl globalDecl) {
    _globalDecl = globalDecl;
    _globalObjectTable = globalDecl.getGlobalObjectTable();
  }

  public void rewrite(FuncDefBlock def) {
    FunctionBlock fb = def.getBlock();
    if (fb == null) return;

    // rewrite expr
    XMPobjectTable localObjectTable = XMPlocalDecl.getObjectTable(fb);

    BasicBlockExprIterator iter = new BasicBlockExprIterator(fb);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();

      try {
        rewriteExpr(expr, localObjectTable);
      } catch (XMPexception e) {
        XMP.error(expr.getLineNo(), e.getMessage());
      }
    }

    // create local object descriptors, constructors and desctructors
    XMPlocalDecl.setupObjectId(fb);
    XMPlocalDecl.setupConstructor(fb);
    XMPlocalDecl.setupDestructor(fb);

    def.Finalize();
  }

  public void rewriteExpr(Xobject expr, XMPobjectTable localObjectTable) throws XMPexception {
    if (expr == null) return;

    bottomupXobjectIterator iter = new bottomupXobjectIterator(expr);
    iter.init();
    while (!iter.end()) {
      Xobject newExpr = null;
      Xobject myExpr = iter.getXobject();
      if (myExpr == null) {
        iter.next();
        continue;
      }

      switch (myExpr.Opcode()) {
        case ARRAY_REF:
          {
            String arrayName = myExpr.getSym();
            XMPalignedArray alignedArray = _globalObjectTable.getAlignedArray(arrayName);
            if (alignedArray == null) {
              if (localObjectTable != null)
                alignedArray = localObjectTable.getAlignedArray(arrayName);
            }

            if (alignedArray != null) {
              if (alignedArray.checkRealloc()) {
                iter.next();
                rewriteAlignedArrayExpr(iter, alignedArray);
                break;
              }
            }

            iter.next();
            break;
          }
        default:
          iter.next();
      }
    }
  }

  private void rewriteAlignedArrayExpr(bottomupXobjectIterator iter,
                                       XMPalignedArray alignedArray) throws XMPexception {
    XobjList getAddrFuncArgs = Xcons.List(alignedArray.getAddrId().Ref());
    parseArrayExpr(iter, alignedArray, 0, getAddrFuncArgs);
  }

  private void parseArrayExpr(bottomupXobjectIterator iter,
                              XMPalignedArray alignedArray, int arrayDimCount, XobjList args) throws XMPexception {
    String syntaxErrMsg = "syntax error on array expression, cannot rewrite distributed array";
    Xobject prevExpr = iter.getPrevXobject();
    Xcode prevExprOpcode = prevExpr.Opcode();
    Xobject myExpr = iter.getXobject();
    Xobject parentExpr = iter.getParent();
    switch (myExpr.Opcode()) {
      case PLUS_EXPR:
        {
          switch (prevExprOpcode) {
            case ARRAY_REF:
              {
                if (arrayDimCount != 0)
                  throw new XMPexception(syntaxErrMsg);

                break;
              }
            case POINTER_REF:
              break;
            default:
              throw new XMPexception(syntaxErrMsg);
          }

          if (parentExpr.Opcode() == Xcode.POINTER_REF) {
            args.add(getCalcIndexFuncRef(alignedArray, arrayDimCount, myExpr.right()));
            iter.next();
            parseArrayExpr(iter, alignedArray, arrayDimCount + 1, args);
          }
          else {
            if (alignedArray.getDim() == arrayDimCount) {
              Xobject funcCall = createRewriteAlignedArrayFunc(alignedArray, arrayDimCount, args, Xcode.POINTER_REF);
              myExpr.setLeft(funcCall);
            }
            else {
              args.add(getCalcIndexFuncRef(alignedArray, arrayDimCount, myExpr.right()));
              Xobject funcCall = createRewriteAlignedArrayFunc(alignedArray, arrayDimCount + 1, args, Xcode.PLUS_EXPR);
              iter.setXobject(funcCall);
            }

            iter.next();
          }

          return;
        }
      case POINTER_REF:
        {
          switch (prevExprOpcode) {
            case PLUS_EXPR:
              break;
            default:
              throw new XMPexception(syntaxErrMsg);
          }

          iter.next();
          parseArrayExpr(iter, alignedArray, arrayDimCount, args);
          return;
        }
      default:
        {
          switch (prevExprOpcode) {
            case ARRAY_REF:
              {
                if (arrayDimCount != 0)
                  throw new XMPexception(syntaxErrMsg);

                break;
              }
            case PLUS_EXPR:
            case POINTER_REF:
              break;
            default:
              throw new XMPexception(syntaxErrMsg);
          }

          Xobject funcCall = createRewriteAlignedArrayFunc(alignedArray, arrayDimCount, args, prevExprOpcode);
          iter.setPrevXobject(funcCall);
          return;
        }
    }
  }

  private Xobject createRewriteAlignedArrayFunc(XMPalignedArray alignedArray, int arrayDimCount,
                                                XobjList getAddrFuncArgs, Xcode opcode) throws XMPexception {
    int arrayDim = alignedArray.getDim();
    Ident getAddrFuncId = null;
    if (arrayDim == arrayDimCount) {
      getAddrFuncId = XMP.getMacroId("_XCALABLEMP_M_GET_ADDR_E_" + arrayDim);
      for (int i = 0; i < arrayDim - 1; i++)
        getAddrFuncArgs.add(alignedArray.getGtolAccIdAt(i).Ref());
    }
    else {
      getAddrFuncId = XMP.getMacroId("_XCALABLEMP_M_GET_ADDR_" + arrayDimCount);
      for (int i = 0; i < arrayDimCount; i++)
        getAddrFuncArgs.add(alignedArray.getGtolAccIdAt(i).Ref());
    }

    Xobject retObj = getAddrFuncId.Call(getAddrFuncArgs);
    switch (opcode) {
      case ARRAY_REF:
      case PLUS_EXPR:
        return retObj;
      case POINTER_REF:
        return Xcons.List(Xcode.POINTER_REF, retObj.Type(), retObj);
      default:
        throw new XMPexception("unknown operation in exc.xcalablemp.XMPrewrite.createRewriteAlignedArrayFunc()");
    }
  }

  private Xobject getCalcIndexFuncRef(XMPalignedArray alignedArray, int index, Xobject indexRef) throws XMPexception {
    int distManner = alignedArray.getDistMannerAt(index);
    switch (distManner) {
      case XMPalignedArray.NO_ALIGN:
      case XMPtemplate.DUPLICATION:
        return indexRef;
      case XMPtemplate.BLOCK:
        if (alignedArray.hasShadow()) {
          XMPshadow shadow = alignedArray.getShadowAt(index);
          switch (shadow.getType()) {
            case XMPshadow.SHADOW_NONE:
              {
                XobjList args = Xcons.List(indexRef,
                                           alignedArray.getGtolTemp0IdAt(index).Ref());
                return XMP.getMacroId("_XCALABLEMP_M_CALC_INDEX_BLOCK").Call(args);
              }
            case XMPshadow.SHADOW_FULL:
              return indexRef;
            case XMPshadow.SHADOW_NORMAL:
              {
                XobjList args = Xcons.List(indexRef,
                                           alignedArray.getGtolTemp0IdAt(index).Ref(),
                                           shadow.getLo());
                return XMP.getMacroId("_XCALABLEMP_M_CALC_INDEX_BLOCK_W_SHADOW").Call(args);
              }
            default:
              throw new XMPexception("unknown shadow type");
          }
        }
        else {
          XobjList args = Xcons.List(indexRef,
                                     alignedArray.getGtolTemp0IdAt(index).Ref());
          return XMP.getMacroId("_XCALABLEMP_M_CALC_INDEX_BLOCK").Call(args);
        }
      case XMPtemplate.CYCLIC:
        if (alignedArray.hasShadow()) {
          XMPshadow shadow = alignedArray.getShadowAt(index);
          switch (shadow.getType()) {
            case XMPshadow.SHADOW_NONE:
              {
                XobjList args = Xcons.List(indexRef,
                                           alignedArray.getGtolTemp0IdAt(index).Ref());
                return XMP.getMacroId("_XCALABLEMP_M_CALC_INDEX_CYCLIC").Call(args);
              }
            case XMPshadow.SHADOW_FULL:
              return indexRef;
            case XMPshadow.SHADOW_NORMAL:
              throw new XMPexception("only block distribution allows shadow");
            default:
              throw new XMPexception("unknown shadow type");
          }
        }
        else {
          XobjList args = Xcons.List(indexRef,
                                     alignedArray.getGtolTemp0IdAt(index).Ref());
          return XMP.getMacroId("_XCALABLEMP_M_CALC_INDEX_CYCLIC").Call(args);
        }
      default:
        throw new XMPexception("unknown distribute manner for array '" + alignedArray.getName()  + "'");
    }
  }
}
