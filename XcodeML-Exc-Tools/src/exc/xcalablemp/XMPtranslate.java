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

  public XMPtranslate(XMPglobalDecl globalDecl) {
    // FIXME current implementation only supports C language
    if (!XmOption.isLanguageC())
      XMP.fatal("current version only supports C language.");

    _globalDecl = globalDecl;
    _translateGlobalPragma = new XMPtranslateGlobalPragma(globalDecl);
    _translateLocalPragma = new XMPtranslateLocalPragma(globalDecl);
    _rewriteExpr = new XMPrewriteExpr(globalDecl);
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
        
    if (def.getName().equals("main")) replaceMain(def);

    FuncDefBlock fd = new FuncDefBlock(def);

    // translate directives
    _translateLocalPragma.translate(fd);

    // rewrite expressions
    _rewriteExpr.rewrite(fd);
  }

  private void replaceMain(XobjectDef def) {
    Ident id = _globalDecl.findVarIdent("main");
    id.setName("_XMP_main");
    def.setName("_XMP_main");
  }
}
