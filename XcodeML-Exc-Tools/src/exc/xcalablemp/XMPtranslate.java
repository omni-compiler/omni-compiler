/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.block.*;
import exc.object.*;
import xcodeml.util.XmOption;

/**
 * XcalableMP AST translator
 */
public class XMPtranslate implements XobjectDefVisitor {
  private XMPglobalDecl			_globalDecl;
  private XMPtranslateGlobalPragma	_translateGlobalPragma;
  private XMPtranslateLocalPragma	_translateLocalPragma;
  private XMPrewriteExpr		_rewriteExpr;
  private boolean                       _all_profile;
  private boolean                       _selective_profile;

  public XMPtranslate(XMPglobalDecl globalDecl) {
    // FIXME current implementation only supports C language
    if (!XmOption.isLanguageC())
      XMP.fatal("current version only supports C language.");

    _globalDecl = globalDecl;
    _translateGlobalPragma = new XMPtranslateGlobalPragma(globalDecl);
    _translateLocalPragma = new XMPtranslateLocalPragma(globalDecl);
    _rewriteExpr = new XMPrewriteExpr(globalDecl);

    _all_profile = false;
    _selective_profile = false;
  }

  public void finalize() {
    _globalDecl.finalize();
  }

  public void doDef(XobjectDef def) {
    translate(def);
  }

  private void translate(XobjectDef def) {
    if (!def.isFuncDef()) {
      Xobject x = def.getDef();
      if (x.Opcode() == Xcode.XMP_PRAGMA) _translateGlobalPragma.translate(x);
      return;
    }
        
    FuncDefBlock fd = new FuncDefBlock(def);

    // translate directives
    _translateLocalPragma.translate(fd);

    // rewrite expressions
		_rewriteExpr.rewrite(fd);
  }

  public void set_all_profile(){
      _all_profile = true;
      _translateLocalPragma.set_all_profile();
  }

  public void set_selective_profile(){
      _selective_profile = true;
      _translateLocalPragma.set_selective_profile();
  }

  public void setScalascaEnabled(boolean v) {
      _translateLocalPragma.setScalascaEnabled(v);
  }

  public void setTlogEnabled(boolean v) {
      _translateLocalPragma.setTlogEnabled(v);
  }
}
