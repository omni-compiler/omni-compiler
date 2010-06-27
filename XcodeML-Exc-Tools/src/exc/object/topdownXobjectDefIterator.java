/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

import java.util.LinkedList;

public class topdownXobjectDefIterator
{
    private LinkedList<XobjectDef> defQueue = new LinkedList<XobjectDef>();
    private XobjectDef curDef;
    
    public topdownXobjectDefIterator(XobjectDef def)
    {
        defQueue.add(def);
    }
    
    public topdownXobjectDefIterator(XobjectDefEnv env)
    {
        for(XobjectDef def : env) {
            defQueue.add(def);
        }
    }
    
    public void init()
    {
        next();
    }
    
    public void next()
    {
        if(defQueue.isEmpty()) {
            curDef = null;
            return;
        }
        curDef = defQueue.removeFirst();
        
        if(curDef.hasChildren()) {
            for(XobjectDef def : curDef.getChildren())
                defQueue.add(def);
        }
    }
    
    public XobjectDef getDef()
    {
        return curDef;
    }
    
    public boolean end()
    {
        return curDef == null && defQueue.isEmpty();
    }
}
