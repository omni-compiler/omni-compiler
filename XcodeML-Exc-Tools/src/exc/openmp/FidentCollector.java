package exc.openmp;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import exc.block.Block;
import exc.block.BlockList;
import exc.block.LabelBlock;
import exc.block.Statement;
import exc.block.StatementIterator;
import exc.block.topdownBlockIterator;
import exc.object.Ident;
import exc.object.Xcode;
import exc.object.Xcons;
import exc.object.XobjList;
import exc.object.Xobject;
import exc.object.Xtype;
import exc.object.topdownXobjectIterator;

/**
 * Fortran identifier collector.
 */
class FidentCollector
{
    private BlockList orgBody;
    private Map<String, Ident> nameMap;
    
    FidentCollector(BlockList orgBody, Map<String, Ident> nameMap)
    {
        this.orgBody = orgBody;
        this.nameMap = nameMap;
    }
    
    /** collect identifier */
    private void collectIdent(BlockList body)
    {
        if(body == null)
            return;
        
        topdownBlockIterator ite = new topdownBlockIterator(body);
        for(ite.init(); !ite.end(); ite.next()) {
            Block b = ite.getBlock();
            if(b == null)
                continue;
            if(b.getBasicBlock() != null) {
                for(StatementIterator site = b.getBasicBlock().statements(); site.hasNext(); ) {
                    Statement s = site.next();
                    if(s != null)
                        collectIdent(s.getExpr());
                }
            }
        }
    }
    
    /** collect identifiers in identifer's declaration */
    void collectIdentInIdent(Ident id)
    {
        if(id == null)
            return;
        collectIdentInType(id.Type());
        collectIdent(id.getFparamValue());
    }

    /** collect variable identifiers */
    void collectIdentName(Xobject x, OMPinfo i)
    {
        if(x == null)
            return;
        
        String name = null;
        if(x instanceof Ident) {
            name = x.getName();
        } else {
            switch(x.Opcode()) {
            case IDENT:
            case VAR:
            case VAR_ADDR:
                name = x.getName();
                break;
            }
        }
        
        if(name != null) {
            Ident id = orgBody.findLocalIdent(name);
            if(id == null) {
                id = i.getIdList().findVarIdent(name);
                if(id == null && name.startsWith(OMPtransPragma.LOCAL_VAR_PREFIX)) {
                    // find original name of private variable.
                    id = orgBody.findLocalIdent(
                        name.substring(OMPtransPragma.LOCAL_VAR_PREFIX.length(), name.length()));
                }
            }
            
            if(id != null) {
                nameMap.put(name, id);
                collectIdentInIdent(id);
                if(id.Type().isStruct()) {
                    Ident stid = orgBody.findLocalIdent(id.Type());
                    if(stid != null)
                        nameMap.put(stid.getName(), stid);
                }
            }
        }
        
        if(x instanceof XobjList) {
            for(Xobject a : (XobjList)x) {
                collectIdentName(a, i);
            }
        }
    }
    
    /** collect identifiers related to variable list(common block/namelist) */
    private void collectIdentInVarList(Xobject varListList)
    {
        for(Xobject varList : (XobjList)varListList) {
            collectIdent(varList.getArg(1));
            Xobject n = varList.getArgOrNull(0);
            if(n != null) {
                String name = n.getName();
                Ident id = orgBody.findLocalIdent(name);
                if(id != null && !nameMap.containsKey(name)) {
                    nameMap.put(name, id);
                    collectIdentInIdent(id);
                }
            }
        }
    }

    /** collect identifiers in type specifier
     * (array subscription, kind parameter, len parameter) */
    private void collectIdentInType(Xtype t)
    {
        if(t == null)
            return;
        collectIdent(t.getFkind());
        collectIdent(t.getFlen());
        if(t.isFarray()) {
            for(Xobject s : t.getFarraySizeExpr())
                collectIdent(s);
        }
    }

    /** collect identifiers in specified node */
    private void collectIdent(Xobject x)
    {
        if(x == null)
            return;
        
        topdownXobjectIterator ite = new topdownXobjectIterator(x);
        for(ite.init(); !ite.end(); ite.next()) {
            Xobject v = ite.getXobject();
            if(v == null || !v.isVarRef())
                continue;
            String name = v.getName();
            Ident id = orgBody.findLocalIdent(name);
            if(id == null || id.isFmoduleVar())
                continue;
            Ident id0 = nameMap.get(name);
            if(id0 != null)
                continue;
            
            if(id.isDeclared()) {
                id = (Ident)id.copy();
                id.setIsDeclared(false);
            }
            nameMap.put(name, id);
            collectIdentInIdent(id);
        }
    }

    /** copy declarations from original block to parallel region */
    void copyDecls(XobjList dest_id_list, Xobject dest_decls,
        BlockList body, OMPinfo i, boolean forWrapper)
    {
        if(forWrapper || !nameMap.isEmpty()) {
            Map<String, Ident> nameMapBak = nameMap;
            Map<String, Ident> bodyNameMap = new HashMap<String, Ident>();
            
            nameMap = bodyNameMap;
            for(Xobject a : dest_id_list)
                collectIdentInType(a.Type());
            collectIdent(orgBody);
            nameMap = nameMapBak;
            
            Xobject orgDecls = orgBody.getDecls();
    
            for(Xobject x : (XobjList)orgDecls) {
                if(x == null)
                    continue;
                String name = null;
                Ident id;
                
                switch(x.Opcode()) {
                case F_USE_DECL:
                case F_USE_ONLY_DECL:
                    dest_decls.add(x);
                    break;
                case F_COMMON_DECL:
                case F_NAMELIST_DECL:
                    if(isInVarList(x, nameMap) || isInVarList(x, bodyNameMap)) {
                        collectIdentInVarList(x);
                        dest_decls.add(x);
                    }
                    break;
                case VAR_DECL:
                    if(forWrapper)
                        break;
                    name = x.getArg(0).getName();
                    if(nameMap.containsKey(name))
                        dest_decls.add(x);
                    break;
                case F_STRUCT_DECL:
                    dest_decls.add(x);
                    id = orgBody.findLocalIdent(x.getArg(0).getName());
                    if(id != null && !nameMap.containsKey(id.getName())) {
                        nameMap.put(id.getName(), id);
                    }
                    break;
                case F_INTERFACE_DECL:
                    if(forWrapper)
                        break;
                    name = (x.getArg(0) != null) ? x.getArg(0).getName() : null;
                    if(nameMap.containsKey(name)) {
                        dest_decls.add(x);
                        break;
                    }
                    if(x.getArgOrNull(3) != null) {
                        List<Xobject> nameList = new ArrayList<Xobject>();
                        boolean exists = false;
                        
                        for(Xobject y : (XobjList)x.getArg(3)) {
                            if(y != null && y.Opcode() == Xcode.FUNCTION_DECL) {
                                nameList.add(y.getArg(0));
                                name = y.getArg(0).getName();
                                if(nameMap.containsKey(name))
                                    exists = true;
                            }
                        }
                        
                        if(exists) {
                            dest_decls.add(x);
                            for(Xobject v : nameList) {
                                id = orgBody.findLocalIdent(v.getName());
                                if(id != null && !nameMap.containsKey(id.getName())) {
                                    nameMap.put(id.getName(), id);
                                }
                            }
                            break;
                        }
                    }
                    break;
                }
            }
            
            Ident[] ids = nameMap.values().toArray(new Ident[nameMap.size()]);
            
            for(Ident id : ids) {
                collectIdentInIdent(id);
            }
            
            for(Ident id : nameMap.values()) {
                if(!dest_id_list.has(id))
                    dest_id_list.add(id);
            }
        }

        if(!forWrapper && body != null) {
            // copy format decls with statement labels in original body.
            List<Xobject> fmtList = new ArrayList<Xobject>();
            Set<Integer> numSet = new HashSet<Integer>();
            
            collectFormatNumsInIOStatement(body, numSet);
            removeFormatNumsInLine(body, numSet);
            collectFormatDecls(fmtList, numSet);
            for(Xobject x : fmtList)
                body.add(x);
        }
    }

    /** collect format identifier used in I/O statement */
    private void collectFormatNumsInIOStatement(BlockList body, Set<Integer> numSet)
    {
        topdownBlockIterator ite = new topdownBlockIterator(body);
        for(ite.init(); !ite.end(); ite.next()) {
            Block b = ite.getBlock();
            if(b == null)
                continue;
            if(b.getBasicBlock() != null) {
                for(StatementIterator site = b.getBasicBlock().statements(); site.hasNext(); ) {
                    Statement s = site.next();
                    if(s == null || s.getExpr() == null)
                        continue;
                    switch(s.getExpr().Opcode()) {
                    default:
                        continue;
                    case F_READ_STATEMENT:
                    case F_WRITE_STATEMENT:
                        break;
                    }
                    Xobject namedValueList = s.getExpr().getArgOrNull(0);
                    if(namedValueList == null)
                        continue;
                    for(Xobject nv : (XobjList)namedValueList) {
                        Xobject sym = nv.getArg(0);
                        if(!sym.getName().equals("fmt"))
                            continue;
                        Xobject val = nv.getArg(1);
                        if(val == null || val.Opcode() != Xcode.INT_CONSTANT)
                            continue;
                        numSet.add(val.getInt());
                    }
                }
            }
        }
    }
    
    /** remove numbers from numSet, which is already in line */
    private void removeFormatNumsInLine(BlockList body, Set<Integer> numSet)
    {
        topdownBlockIterator ite = new topdownBlockIterator(body);
        Xobject lastLabel = null;
        for(ite.init(); !ite.end(); ite.next()) {
            Block b = ite.getBlock();
            if(b == null)
                continue;
            if(b instanceof LabelBlock && b.getLabel() != null)
                lastLabel = b.getLabel();
            else if(b.getBasicBlock() != null) {
                for(StatementIterator site = b.getBasicBlock().statements(); site.hasNext(); ) {
                    Statement s = site.next();
                    if(s == null || s.getExpr() == null)
                        continue;
                    if(lastLabel == null || s.getExpr().Opcode() != Xcode.F_FORMAT_DECL)
                        continue;
                    int num = Integer.parseInt(lastLabel.getName());
                    if(numSet.contains(num)) {
                        numSet.remove(num);
                        lastLabel = null;
                    }
                }
            }
        }
    }
    
    /** collect format declarations which have format identifier in numSet */
    private void collectFormatDecls(List<Xobject> fmtList, Set<Integer> numSet)
    {
        topdownBlockIterator ite = new topdownBlockIterator(orgBody);
        Xobject lastLabel = null;
        for(ite.init(); !ite.end(); ite.next()) {
            Block b = ite.getBlock();
            if(b == null)
                continue;
            if(b instanceof LabelBlock && b.getLabel() != null)
                lastLabel = b.getLabel();
            else if(b.getBasicBlock() != null) {
                for(StatementIterator site = b.getBasicBlock().statements(); site.hasNext(); ) {
                    Statement s = site.next();
                    if(s == null || s.getExpr() == null)
                        continue;
                    if(lastLabel == null || s.getExpr().Opcode() != Xcode.F_FORMAT_DECL)
                        continue;
                    int num = Integer.parseInt(lastLabel.getName());
                    if(numSet.contains(num)) {
                        fmtList.add(Xcons.List(Xcode.STATEMENT_LABEL, lastLabel));
                        fmtList.add(s.getExpr());
                        lastLabel = null;
                    }
                }
            }
        }
    }

    /** return if nameMap contains one of the name in varListList */
    private boolean isInVarList(Xobject varListList, Map<String, Ident> nameMap)
    {
        for(Xobject varList : (XobjList)varListList) {
            XobjList vs = (XobjList)varList.getArgOrNull(1);
            if(vs == null)
                continue;
            for(Xobject v : vs) {
                if(v.Opcode() != Xcode.F_VAR_REF)
                    continue;
                String name = v.getArg(0).getName();
                if(nameMap.containsKey(name))
                    return true;
            }
        }
        
        return false;
    }
}
