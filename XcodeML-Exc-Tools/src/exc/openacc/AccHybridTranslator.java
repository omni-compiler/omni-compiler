package exc.openacc;

import exc.block.FuncDefBlock;
import exc.block.FunctionBlock;
import exc.object.*;
import xcodeml.util.XmOption;
import exc.block.*;

public class AccHybridTranslator implements XobjectDefVisitor {
	// private final ACCglobalDecl _globalDecl;
	// private final AccRewriter _rewrite;

	public AccHybridTranslator(XobjectFile xobjFile) {
		if (!XmOption.isLanguageC()) {
			ACC.fatal("current version only supports C language.");
		}

		// _globalDecl = new ACCglobalDecl(xobjFile);
		// _rewrite = new AccRewriter(_globalDecl);
	}

	// AccTranslator から
	@Override
	public void doDef(XobjectDef def) {

		String fname = def.getName();
		System.out.println("Func name is " + fname);

		XobjectIterator i = new topdownXobjectIterator(def.getFuncBody());
		for (i.init(); !i.end(); i.next()) {
			Xobject x = i.getXobject();
			if (x != null && (x.isVariable() || x.isVarAddr()))
				System.out.println("Variable '" + x.getName() + "' is referenced from Function '" + fname + "'");
		}

		if (!def.isFuncDef()) {
			return;
		}

		// Block fb = Bcons.buildFunctionBlock(def);
		// BlockIterator j = new topdownBlockIterator(fb);
		// for(j.init(); !j.end(); j.next()){
		// fb = j.getBlock();
		// // System.out.println("block: " + fb.

		// BasicBlock bb = fb.getBasicBlock();
		// if(bb == null) continue;
		// for(Statement s = bb.getHead(); s != null; s = s.getNext()){
		// Xobject x = s.getExpr();
		// // System.out.println("statement: " + x.getName());

		// // 式xに対する操作 ...
		// }
		// }

		FuncDefBlock fd = new FuncDefBlock(def);
		FunctionBlock fb = fd.getBlock();
		String funcName = fb.getName();
		if (funcName == "main") {
			fb.remove();
			return;
		}

		// XMPrewriteExpr より
		topdownBlockIterator bIter = new topdownBlockIterator(fb);

		for (bIter.init(); !bIter.end(); bIter.next()) {
			Block block = bIter.getBlock();

			if (block.Opcode() == Xcode.ACC_PRAGMA) {
				PragmaBlock pragmaBlock = ((PragmaBlock) block);
				Xobject clauses = pragmaBlock.getClauses();
				if (clauses == null) {
					// エラーにしたい

					// BlockList newBody = Bcons.emptyBody();
					// rewriteACCClauses(clauses, pragmaBlock, fb, localXMPsymbolTable, newBody);
					// if (!newBody.isEmpty()) {
					// bIter.setBlock(Bcons.COMPOUND(newBody));
					// newBody.add(block);
					// }
				}

				if (pragmaBlock.getPragma().equals("DATA")) {
					System.out.println("DATA ディレクティブ！！！");
					BlockList body = pragmaBlock.getBody();
					if (body.getDecls() != null) {
						BlockList newBody = Bcons.emptyBody(body.getIdentList().copy(), body.getDecls().copy());
						body.setIdentList(null);
						body.setDecls(null);
						// newBody.add(Bcons.PRAGMA(Xcode.ACC_PRAGMA, pragmaBlock.getPragma(),
						// pragmaBlock.getClauses(), body));
						pragmaBlock.replace(Bcons.COMPOUND(newBody));
					}
				}
			}
		}

		// if (def.isFuncDef()) {
		// FuncDefBlock fd = new FuncDefBlock(def);
		// FunctionBlock fb = fd.getBlock();
		// doFuncDef(fb);
		// fd.finalizeBlock();
		// } else {
		// Xobject x = def.getDef();
		// doNonFuncDef(x);
		// }
	}

	// private void doFuncDef(FunctionBlock fb){
	// _rewrite.doFuncDef(fb);
	// ACC.exitByError();
	// }

	// private void doNonFuncDef(Xobject x){
	// _rewrite.doNonFuncDef(x);
	// ACC.exitByError();
	// }

	// AccProcessor から
	// private void doFuncDef(FunctionBlock fb) {
	// BlockIterator blockIterator;
	// // if (_isTopdown) {
	// blockIterator = new topdownBlockIterator(fb);
	// // } else {
	// // blockIterator = new bottomupBlockIterator(fb);
	// // }

	// for (blockIterator.init(); !blockIterator.end(); blockIterator.next()) {
	// Block b = blockIterator.getBlock();
	// switch (b.Opcode()) {
	// case ACC_PRAGMA:
	// try {
	// doLocalAccPragma((PragmaBlock) b);
	// } catch (ACCexception e) {
	// ACC.error(b.getLineNo(), e.getMessage());
	// }
	// break;
	// case PRAGMA_LINE:
	// if (_warnUnknownPragma) {
	// ACC.warning(b.getLineNo(), "unknown pragma : " + b);
	// }
	// break;
	// default:
	// }
	// }
	// ACC.exitByError();
	// }

	// // AccProcessor から
	// private void doNonFuncDef(Xobject x) {
	// switch (x.Opcode()) {
	// case ACC_PRAGMA:
	// try {
	// doGlobalAccPragma(x);
	// } catch (ACCexception e) {
	// ACC.error(x.getLineNo(), e.getMessage());
	// }
	// break;
	// case PRAGMA_LINE:
	// if (_warnUnknownPragma) {
	// ACC.warning(x.getLineNo(), "unknown pragma : " + x);
	// }
	// break;
	// default:
	// }
	// ACC.exitByError();
	// }

	// void doLocalAccPragma(PragmaBlock pb) throws ACCexception {
	// doAccPragma(pb);
	// }

	// void doGlobalAccPragma(Xobject def) throws ACCexception {
	// doAccPragma(def);
	// }

	// void doAccPragma(PropObject po) throws ACCexception {
	// Object obj = po.getProp(AccDirective.prop);
	// if (obj == null)
	// return;
	// AccDirective dire = (AccDirective) obj;
	// dire.rewrite();
	// }

	// void rewrite() throws ACCexception {
	// if (isDisabled()) {
	// _pb.replace(Bcons.COMPOUND(_pb.getBody()));
	// return;
	// }

	// // build
	// BlockList beginBody = Bcons.emptyBody();
	// for (Block b : initBlockList)
	// beginBody.add(b);
	// for (Block b : copyinBlockList)
	// beginBody.add(b);
	// BlockList endBody = Bcons.emptyBody();
	// for (Block b : copyoutBlockList)
	// endBody.add(b);
	// for (Block b : finalizeBlockList)
	// endBody.add(b);

	// Block beginBlock = Bcons.COMPOUND(beginBody);
	// Block endBlock = Bcons.COMPOUND(endBody);

	// BlockList kernelsBody = Bcons.emptyBody();
	// for (Block b : _kernelBlocks) {
	// kernelsBody.add(b);
	// }
	// Block kernelsBlock = Bcons.COMPOUND(kernelsBody);

	// BlockList resultBody = Bcons.emptyBody();
	// for (Xobject x : idList) {
	// resultBody.addIdent((Ident) x);
	// }

	// Xobject ifExpr = _info.getIntExpr(ACCpragma.IF);
	// boolean isEnabled = (ifExpr == null || (ifExpr.isIntConstant() &&
	// !ifExpr.isZeroConstant()));
	// if (isEnabled) {
	// resultBody.add(beginBlock);
	// resultBody.add(kernelsBlock);
	// resultBody.add(endBlock);
	// } else {
	// Ident condId = resultBody.declLocalIdent("_ACC_DATA_IF_COND", Xtype.charType,
	// StorageClass.AUTO, ifExpr);
	// resultBody.add(Bcons.IF(condId.Ref(), beginBlock, null));
	// resultBody.add(Bcons.IF(condId.Ref(), kernelsBlock,
	// Bcons.COMPOUND(_pb.getBody())));
	// resultBody.add(Bcons.IF(condId.Ref(), endBlock, null));
	// }

	// _pb.replace(Bcons.COMPOUND(resultBody));
	// }

	public void finish() {
		// ヘッダーを出力する？

		// ACCgpuDecompiler gpuDecompiler = new ACCgpuDecompiler();
		// gpuDecompiler.decompile(_globalDecl);

	}
}