/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.block.*;
import exc.object.*;
import java.util.Vector;
import java.util.Iterator;

public class XMPalignedArray {
  // defined in xmp_constant.h
  public final static int NOT_ALIGNED	= 200;
  public final static int DUPLICATION	= 201;
  public final static int BLOCK		= 202;
  public final static int CYCLIC	= 203;

  private String		_name;
  private Xtype			_type;
  private int			_dim;
  private Vector<Long>		_sizeVector;
  private Vector<XMPshadow>	_shadowVector;
  private Vector<Integer>	_alignMannerVector;
  private Vector<Ident>		_accIdVector;
  private Vector<Ident>		_gtolTemp0IdVector;
  private Vector<Integer>	_alignSubscriptIndexVector;
  private Vector<Xobject>	_alignSubscriptExprVector;
  private Ident			_arrayId;
  private Ident			_descId;
  private Ident			_addrId;
  private boolean		_hasShadow;
  private boolean		_reallocChecked;
  private boolean		_realloc;
  private XMPtemplate		_alignedTemplate;

  public static int convertDistMannerToAlignManner(int distManner) throws XMPexception {
    switch (distManner) {
      case XMPtemplate.DUPLICATION:
        return DUPLICATION;
      case XMPtemplate.BLOCK:
        return BLOCK;
      case XMPtemplate.CYCLIC:
        return CYCLIC;
      default:
        throw new XMPexception("unknown dist manner");
    }
  }

  public XMPalignedArray(String name, Xtype type, int dim,
                         Vector<Long> sizeVector, Vector<Ident> accIdVector,
                         Ident arrayId, Ident descId, Ident addrId,
                         XMPtemplate alignedTemplate) {
    _name = name;
    _type = type;
    _dim = dim;
    _sizeVector = sizeVector;
    _shadowVector = new Vector<XMPshadow>(XMP.MAX_DIM);
    _alignMannerVector = new Vector<Integer>(XMP.MAX_DIM);
    _accIdVector = accIdVector;
    _gtolTemp0IdVector = new Vector<Ident>(XMP.MAX_DIM);
    _alignSubscriptIndexVector = new Vector<Integer>(XMP.MAX_DIM);
    _alignSubscriptExprVector = new Vector<Xobject>(XMP.MAX_DIM);
    for (int i = 0; i < dim; i++) {
      _shadowVector.add(new XMPshadow(XMPshadow.SHADOW_NONE, null, null));
      _alignMannerVector.add(null);
      _gtolTemp0IdVector.add(null);
      _alignSubscriptIndexVector.add(null);
      _alignSubscriptExprVector.add(null);
    }
    _arrayId = arrayId;
    _descId = descId;
    _addrId = addrId;
    _hasShadow = false;
    _reallocChecked = false;
    _alignedTemplate = alignedTemplate;
  }

  public String getName() {
    return _name;
  }

  public Xtype getType() {
    return _type;
  }

  public int getDim() {
    return _dim;
  }

  public void setAlignMannerAt(int manner, int index) {
    _alignMannerVector.setElementAt(new Integer(manner), index);
  }

  public int getAlignMannerAt(int index) {
    return _alignMannerVector.get(index).intValue();
  }

  public String getAlignMannerStringAt(int index) throws XMPexception {
    switch (getAlignMannerAt(index)) {
      case NOT_ALIGNED:
        return new String("NOT_ALIGNED");
      case DUPLICATION:
        return new String("DUPLICATION");
      case BLOCK:
        return new String("BLOCK");
      case CYCLIC:
        return new String("CYCLIC");
      default:
        throw new XMPexception("unknown align manner");
    }
  }

  public Vector<Ident> getAccIdVector() {
    return _accIdVector;
  }

  public Ident getAccIdAt(int index) {
    return _accIdVector.get(index);
  }

  // temp0 is
  // block distribution:	parallel/serial lower	| _XMP_gtol_lower_<array_name>_<array_dim>
  // cyclic distribution:	nodes size		| _XMP_gtol_cycle_<array_name>_<array_dim>
  public void setGtolTemp0IdAt(Ident temp0Id, int index) {
    _gtolTemp0IdVector.setElementAt(temp0Id, index);
  }

  public Ident getGtolTemp0IdAt(int index) {
    return _gtolTemp0IdVector.get(index);
  }

  public void setAlignSubscriptIndexAt(int alignSubscriptIndex, int alignSourceIndex) {
    _alignSubscriptIndexVector.setElementAt(new Integer(alignSubscriptIndex), alignSourceIndex);
  }

  public Integer getAlignSubscriptIndexAt(int alignSourceIndex) {
    return _alignSubscriptIndexVector.get(alignSourceIndex);
  }

  public void setAlignSubscriptExprAt(Xobject alignSubscriptExpr, int alignSourceIndex) {
    _alignSubscriptExprVector.setElementAt(alignSubscriptExpr, alignSourceIndex);
  }

  public Xobject getAlignSubscriptExprAt(int alignSourceIndex) {
    return _alignSubscriptExprVector.get(alignSourceIndex);
  }

  public Ident getArrayId() {
    return _arrayId;
  }

  public Ident getDescId() {
    return _descId;
  }

  public Ident getAddrId() {
    return _addrId;
  }

  public void setHasShadow() {
    _hasShadow = true;
  }

  public boolean hasShadow() {
    return _hasShadow;
  }

  public void setShadowAt(XMPshadow shadow, int index) {
    _shadowVector.setElementAt(shadow, index);
  }

  public XMPshadow getShadowAt(int index) {
    return _shadowVector.get(index);
  }

  public XMPtemplate getAlignedTemplate() {
    return _alignedTemplate;
  }

  public boolean checkRealloc() throws XMPexception {
    if (_reallocChecked) return _realloc;

    if (_hasShadow) {
      for (int i = 0; i < _dim; i++) {
        switch (getAlignMannerAt(i)) {
          case NOT_ALIGNED:
          case DUPLICATION:
            break;
          case BLOCK:
          case CYCLIC:
            {
              XMPshadow shadow = getShadowAt(i);
              switch (shadow.getType()) {
                case XMPshadow.SHADOW_FULL:
                  break;
                case XMPshadow.SHADOW_NONE:
                case XMPshadow.SHADOW_NORMAL:
                  {
                    _reallocChecked = true;
                    _realloc = true;
                    return _realloc;
                  }
                default:
                  throw new XMPexception("unknown shadow type");
              }
            } break;
          default:
            throw new XMPexception("unknown align manner");
        }
      }

      _reallocChecked = true;
      _realloc = false;
      return _realloc;
    }
    else {
      for (int i = 0; i < _dim; i++) {
        switch (getAlignMannerAt(i)) {
          case NOT_ALIGNED:
          case DUPLICATION:
            break;
          case BLOCK:
          case CYCLIC:
            {
              _reallocChecked = true;
              _realloc = true;
              return _realloc;
            }
          default:
            throw new XMPexception("unknown align manner");
        }
      }

      _reallocChecked = true;
      _realloc = false;
      return _realloc;
    }
  }

  public boolean realloc() throws XMPexception {
    if (_reallocChecked) return _realloc;
    else                 return checkRealloc();
  }

  public static void translateAlign(XobjList alignDecl, XMPglobalDecl globalDecl,
                                    boolean isLocalPragma, PragmaBlock pb) throws XMPexception {
    // local parameters
    BlockList funcBlockList = null;
    XMPsymbolTable localXMPsymbolTable = null;
    if (isLocalPragma) {
      funcBlockList = XMPlocalDecl.findParentFunctionBlock(pb).getBody();
      localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
    }

    // get array information
    String arrayName = alignDecl.getArg(0).getString();
    XMPalignedArray alignedArray = null;
    if (isLocalPragma) {
      alignedArray = localXMPsymbolTable.getXMPalignedArray(arrayName);
    }
    else {
      alignedArray = globalDecl.getXMPalignedArray(arrayName);
    }

    if (alignedArray != null) {
      throw new XMPexception("array '" + arrayName + "' is already aligned");
    }

    Ident arrayId = null;
    if (isLocalPragma) {
      arrayId = funcBlockList.findLocalIdent(arrayName);
      if (arrayId != null) {
        if (arrayId.getStorageClass() != StorageClass.PARAM) {
          throw new XMPexception("array '" + arrayName + "' is not a parameter");
        }
      }
    }
    else {
      arrayId = globalDecl.findVarIdent(arrayName);
    }

    if (arrayId == null) {
      throw new XMPexception("array '" + arrayName + "' is not declared");
    }

    Xtype arrayType = arrayId.Type();
    if (arrayType.getKind() != Xtype.ARRAY) {
      throw new XMPexception(arrayName + " is not an array");
    }

    Xtype arrayElmtType = arrayType.getArrayElementType();
    Xobject arrayElmtTypeRef = null;
    if (arrayElmtType.getKind() == Xtype.BASIC) {
      arrayElmtTypeRef = XMP.createBasicTypeConstantObj(arrayElmtType);
    }
    else {
      arrayElmtTypeRef = Xcons.IntConstant(XMP.NONBASIC_TYPE);
    }

    // get template information
    String templateName = alignDecl.getArg(2).getString();
    XMPtemplate templateObj = null;
    if (isLocalPragma) {
      templateObj = XMPlocalDecl.getXMPtemplate(templateName, localXMPsymbolTable, globalDecl);
    }
    else {
      templateObj = globalDecl.getXMPtemplate(templateName);
    }

    if (templateObj == null) {
      throw new XMPexception("template '" + templateName + "' is not declared");
    }

    if (!templateObj.isFixed()) {
      throw new XMPexception("template '" + templateName + "' is not fixed");
    }

    if (!(templateObj.isDistributed())) {
      throw new XMPexception("template '" + templateName + "' is not distributed");
    }

    int templateDim = templateObj.getDim();

    // declare array address pointer, array descriptor
    Ident arrayAddrId = null;
    Ident arrayDescId = null;
    if (isLocalPragma) {
      arrayAddrId = XMPlocalDecl.addObjectId(XMP.ADDR_PREFIX_ + arrayName, Xtype.Pointer(arrayElmtType), pb);

      arrayDescId = XMPlocalDecl.addObjectId(XMP.DESC_PREFIX_ + arrayName, pb);
    }
    else {
      if (arrayId.getStorageClass() == StorageClass.EXTERN) {
        arrayAddrId = globalDecl.declExternIdent(XMP.ADDR_PREFIX_ + arrayName, Xtype.Pointer(arrayElmtType));
      }
      else if (arrayId.getStorageClass() == StorageClass.STATIC) {
        arrayAddrId = globalDecl.declStaticIdent(XMP.ADDR_PREFIX_ + arrayName, Xtype.Pointer(arrayElmtType));
      }
      else if (arrayId.getStorageClass() == StorageClass.EXTDEF) {
        arrayAddrId = globalDecl.declGlobalIdent(XMP.ADDR_PREFIX_ + arrayName, Xtype.Pointer(arrayElmtType));
      }
      else {
        throw new XMPexception("cannot align array '" + arrayName +  ", wrong storage class");
      }

      arrayDescId = globalDecl.declStaticIdent(XMP.DESC_PREFIX_ + arrayName, Xtype.voidPtrType);
    }

    int arrayDim = arrayType.getNumDimensions();
    if (arrayDim > XMP.MAX_DIM) {
      throw new XMPexception("array dimension should be less than " + (XMP.MAX_DIM + 1));
    }

    XobjList initArrayDescFuncArgs = Xcons.List(arrayDescId.getAddr(),
                                                templateObj.getDescId().Ref(),
                                                Xcons.IntConstant(arrayDim),
                                                arrayElmtTypeRef,
                                                Xcons.SizeOf(arrayElmtType));

    Vector<Long> arraySizeVector = new Vector<Long>(arrayDim);
    Vector<Ident> accIdVector = new Vector<Ident>(arrayDim);
    for (int i = 0; i < arrayDim; i++, arrayType = arrayType.getRef()) {
      long dimSize = arrayType.getArraySize();
      if (dimSize == 0) {
        throw new XMPexception("array size cannot be omitted");
      }
      else if (dimSize == -1) {
        if (isLocalPragma) {
          // FIXME allow dynamic array
          throw new XMPexception("dynamic array is not allowed yet");
        }
        else {
          // FIXME possible error in global scope???
          throw new XMPexception("array size should be fixed");
        }
      }

      arraySizeVector.add(new Long(dimSize));
      initArrayDescFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.LongLongConstant(0, dimSize)));

      Ident accId = null;
      if (isLocalPragma) {
        accId = XMPlocalDecl.addObjectId(XMP.GTOL_PREFIX_ + "acc_" + arrayName + "_" + i,
                                         Xtype.unsignedlonglongType, pb);
      }
      else {
        accId = globalDecl.declStaticIdent(XMP.GTOL_PREFIX_ + "acc_" + arrayName + "_" + i,
                                           Xtype.unsignedlonglongType);
      }

      accIdVector.add(accId);
    }

    alignedArray = new XMPalignedArray(arrayName, arrayElmtType, arrayDim,
                                       arraySizeVector, accIdVector,
                                       arrayId, arrayDescId, arrayAddrId,
                                       templateObj);

    if (isLocalPragma) {
      XMPlocalDecl.addConstructorCall("_XMP_init_array_desc", initArrayDescFuncArgs, globalDecl, pb);
      XMPlocalDecl.insertDestructorCall("_XMP_finalize_array_desc", Xcons.List(arrayDescId.Ref()), globalDecl, pb);
      localXMPsymbolTable.putXMPalignedArray(alignedArray);
    }
    else {
      globalDecl.addGlobalInitFuncCall("_XMP_init_array_desc", initArrayDescFuncArgs);
      globalDecl.putXMPalignedArray(alignedArray);
    }

    // check <align-source> list, <align-subscrip> list
    XobjList alignSourceList = (XobjList)alignDecl.getArg(1);
    XobjList alignSubscriptList = (XobjList)alignDecl.getArg(3);
    XobjList alignSubscriptVarList = (XobjList)alignSubscriptList.left();
    XobjList alignSubscriptExprList = (XobjList)alignSubscriptList.right();

    // check <align-source> list
    if (XMPutil.countElmts(alignSourceList) != arrayDim) {
      throw new XMPexception("the number of <align-source>s is not the same with array dimension");
    }
    else if (XMPutil.countElmts(alignSourceList, XMP.ASTERISK) == arrayDim) {
      throw new XMPexception("array " + arrayName + " is not aligned on any dimension");
    }

    // check <align-subscript> list
    if (XMPutil.countElmts(alignSubscriptVarList) != templateDim) {
      throw new XMPexception("the number of <align-subscript>s is not the same with template dimension");
    }

    // check ':' source/subscript
    if (XMPutil.countElmts(alignSourceList, XMP.COLON) !=
        XMPutil.countElmts(alignSubscriptVarList, XMP.COLON)) {
      throw new XMPexception("the number of ':' in <align-source> list is not the same with <align-subscript> list");
    }

    // create align function calls
    int alignSourceIndex = 0;
    for (XobjArgs i = alignSourceList.getArgs(); i != null; i = i.nextArgs()) {
      String alignSource = i.getArg().getString();

      if (alignSource.equals(XMP.ASTERISK)) {
        declNotAlignFunc(alignedArray, alignSourceIndex, globalDecl, isLocalPragma, pb);
      }
      else if (alignSource.equals(XMP.COLON)) {
        if (!XMPutil.hasElmt(alignSubscriptVarList, XMP.COLON)) {
          throw new XMPexception("cannot find ':' in <align-subscript> list");
        }

        int alignSubscriptIndex = XMPutil.getLastIndex(alignSubscriptVarList, XMP.COLON);
        alignSubscriptVarList.setArg(alignSubscriptIndex, null);

        declAlignFunc(alignedArray, alignSourceIndex, templateObj, alignSubscriptIndex, null, globalDecl, isLocalPragma, pb);
      }
      else {
        if (XMPutil.countElmts(alignSourceList, alignSource) != 1) {
          throw new XMPexception("multiple '" + alignSource + "' indicated in <align-source> list");
        }

        if (XMPutil.hasElmt(alignSubscriptVarList, alignSource)) {
          if (XMPutil.countElmts(alignSubscriptVarList, alignSource) != 1) {
            throw new XMPexception("multiple '" + alignSource + "' indicated in <align-subscript> list");
          }

          int alignSubscriptIndex = XMPutil.getFirstIndex(alignSubscriptVarList, alignSource);
          declAlignFunc(alignedArray, alignSourceIndex, templateObj, alignSubscriptIndex,
                        alignSubscriptExprList.getArg(alignSubscriptIndex), globalDecl, isLocalPragma, pb);
        }
        else {
          throw new XMPexception("cannot find '" + alignSource + "' in <align-subscript> list");
        }
      }

      alignSourceIndex++;
    }

    // check alignSubscriptVarList
    for (XobjArgs i = alignSubscriptVarList.getArgs(); i != null; i = i.nextArgs()) {
      String alignSubscript = i.getArg().getString();

      if (alignSubscript.equals(XMP.ASTERISK) || alignSubscript.equals(XMP.COLON)) {
        break;
      }

      if (XMPutil.hasElmt(alignSourceList, alignSubscript)) {
        if (XMPutil.countElmts(alignSourceList, alignSubscript) != 1) {
          throw new XMPexception("no/multiple '" + alignSubscript + "' indicated in <align-source> list");
        }
      }
      else {
        throw new XMPexception("cannot find '" + alignSubscript + "' in <align-source> list");
      }
    }

    // init array communicator
    XobjList initArrayCommFuncArgs = Xcons.List(alignedArray.getDescId().Ref());
    for (XobjArgs i = alignSubscriptVarList.getArgs(); i != null; i = i.nextArgs()) {
      String alignSubscript = i.getArg().getString();

      if (alignSubscript.equals(XMP.ASTERISK)) {
        initArrayCommFuncArgs.add(Xcons.IntConstant(1));
      }
      else {
        initArrayCommFuncArgs.add(Xcons.IntConstant(0));
      }
    }

    if (isLocalPragma) {
      XMPlocalDecl.addConstructorCall("_XMP_init_array_comm", initArrayCommFuncArgs, globalDecl, pb);

      // init array address
      XobjList initArrayAddrFuncArgs = Xcons.List(arrayAddrId.getAddr(),
                                                  arrayId.getAddr(),
                                                  arrayDescId.Ref());
      for (int i = arrayDim - 1; i >= 0; i--) {
        initArrayAddrFuncArgs.add(Xcons.Cast(Xtype.unsignedlonglongType,
                                             alignedArray.getAccIdAt(i).getAddr()));
      }

      XMPlocalDecl.addConstructorCall("_XMP_init_array_addr", initArrayAddrFuncArgs, globalDecl, pb);
    }
    else {
      globalDecl.addGlobalInitFuncCall("_XMP_init_array_comm", initArrayCommFuncArgs);
    }
  }

  private static void declNotAlignFunc(XMPalignedArray alignedArray, int alignSourceIndex,
                                       XMPglobalDecl globalDecl, boolean isLocalPragma, PragmaBlock pb) throws XMPexception {
    XobjList alignFuncArgs = Xcons.List(alignedArray.getDescId().Ref(),
                                        Xcons.IntConstant(alignSourceIndex));

    alignedArray.setAlignMannerAt(XMPalignedArray.NOT_ALIGNED, alignSourceIndex);

    if (isLocalPragma) {
      XMPlocalDecl.addConstructorCall("_XMP_align_array_NOT_ALIGNED", alignFuncArgs, globalDecl, pb);
    }
    else {
      globalDecl.addGlobalInitFuncCall("_XMP_align_array_NOT_ALIGNED", alignFuncArgs);
    }
  }

  private static void declAlignFunc(XMPalignedArray alignedArray, int alignSourceIndex,
                                    XMPtemplate templateObj, int alignSubscriptIndex,
                                    Xobject alignSubscriptExpr,
                                    XMPglobalDecl globalDecl, boolean isLocalPragma, PragmaBlock pb) throws XMPexception {
    XobjList alignFuncArgs = Xcons.List(alignedArray.getDescId().Ref(),
                                        Xcons.IntConstant(alignSourceIndex),
                                        Xcons.IntConstant(alignSubscriptIndex));

    if (alignSubscriptExpr == null) {
      alignFuncArgs.add(Xcons.IntConstant(0));
    }
    else {
      alignFuncArgs.add(alignSubscriptExpr);
    }

    int distManner = templateObj.getDistMannerAt(alignSubscriptIndex);
    alignedArray.setAlignMannerAt(XMPalignedArray.convertDistMannerToAlignManner(distManner), alignSourceIndex);

    alignedArray.setAlignSubscriptIndexAt(alignSubscriptIndex, alignSourceIndex);
    alignedArray.setAlignSubscriptExprAt(alignSubscriptExpr, alignSourceIndex);

    switch (distManner) {
      case XMPtemplate.DUPLICATION: // FIXME how implement???
        break;
      case XMPtemplate.BLOCK:
      case XMPtemplate.CYCLIC:
        {
          Ident gtolTemp0Id = null;
          if (isLocalPragma) {
            gtolTemp0Id = XMPlocalDecl.addObjectId(XMP.GTOL_PREFIX_ + "temp0_" + alignedArray.getName() + "_" + alignSourceIndex,
                                                   Xtype.intType, pb);
          }
          else {
            gtolTemp0Id = globalDecl.declStaticIdent(XMP.GTOL_PREFIX_ + "temp0_" + alignedArray.getName() + "_" + alignSourceIndex,
                                                     Xtype.intType);
          }

          alignedArray.setGtolTemp0IdAt(gtolTemp0Id, alignSourceIndex);
          alignFuncArgs.add(gtolTemp0Id.getAddr());

          break;
        }
      default:
        throw new XMPexception("unknown distribute manner");
    }

    if (isLocalPragma) {
      XMPlocalDecl.addConstructorCall("_XMP_align_array_" + templateObj.getDistMannerStringAt(alignSubscriptIndex),
                                      alignFuncArgs, globalDecl, pb);
    }
    else {
      globalDecl.addGlobalInitFuncCall("_XMP_align_array_" + templateObj.getDistMannerStringAt(alignSubscriptIndex),
                                       alignFuncArgs);
    }
  }
}
