package exc.object;

import java.util.Stack;

// Warning !
// topdownXobjectIterator.setXobject cannot be applied to the topXobject.

public class topdownXobjectIterator extends XobjectIterator
{
    Stack<Xobject> obj_stack;
    Stack<XobjArgs> arg_stack;

    public topdownXobjectIterator(Xobject x)
    {
        this(x, false);
    }

    public topdownXobjectIterator(Xobject x, boolean trimBS)
    {
        topXobject = x;
        trimBlockStmt = trimBS;
    }

    @Override
    public void init(Xobject x)
    {
        obj_stack = new Stack<Xobject>();
        arg_stack = new Stack<XobjArgs>();
        if(x == null ||
           !trimBlockStmt || x.Opcode() != Xcode.F_BLOCK_STATEMENT)
            currentXobject = x;
    }

    @Override
    public void init()
    {
        init(topXobject);
    }

    @Override
    public void next()
    {
        if(currentXobject != null && currentXobject.Opcode() != Xcode.ID_LIST
            && currentXobject instanceof XobjList && currentXobject.getArgs() != null) {
            // save current top
            obj_stack.push(currentXobject);
            arg_stack.push(currentArgs);
            currentArgs = currentXobject.getArgs();
            currentXobject = currentArgs.getArg();
            if(currentXobject == null ||
               !trimBlockStmt || currentXobject.Opcode() != Xcode.F_BLOCK_STATEMENT)
                return;
        }
        while(!obj_stack.empty()) {
            while((currentArgs = currentArgs.nextArgs()) != null) {
                if((currentXobject = currentArgs.getArg()) == null ||
                    !trimBlockStmt || currentXobject.Opcode() != Xcode.F_BLOCK_STATEMENT)
                    return;
                else
                    continue;
            }
            obj_stack.pop();
            currentArgs = arg_stack.pop();
        }
        currentXobject = null;
        return;
    }

    public void setXobject(Xobject x)
    {
      // fix me
      //      if(currentArgs == null){
      //	currentArgs = new XobjArgs(null, null);
      //      }

      currentArgs.setArg(x);
    }

    @Override
    public boolean end()
    {
        return currentXobject == null && obj_stack.empty();
    }

    @Override
    public Xobject getParent()
    {
        if(obj_stack.empty())
            return null;
        else
            return obj_stack.peek();
    }
}
